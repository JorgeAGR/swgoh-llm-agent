"""Tools the agent uses to ground its answers in game data."""

from collections import defaultdict
from typing import Literal

from langchain_core.tools import BaseTool, tool

from swgoh_agent.catalog import Catalog, GlossaryEntry
from swgoh_agent.knowledge.index import KnowledgeIndex
from swgoh_agent.models import Unit

Alignment = Literal["Light Side", "Dark Side", "Neutral"]
Role = Literal["Attacker", "Support", "Tank", "Healer"]
CombatType = Literal["character", "ship"]


def _did_you_mean(query: str, candidates: list[Unit]) -> str:
    if not candidates:
        return f"No unit matches '{query}'. Try search_units with a description."
    names = "\n".join(f"- {u.summary_line()}" for u in candidates[:10])
    return f"'{query}' is ambiguous or inexact. Did you mean:\n{names}"


def render_unit(unit: Unit, catalog: Catalog) -> str:
    lines = [
        f"# {unit.name} ({unit.base_id})",
        f"{unit.combat_type.title()} | {unit.alignment} | {unit.role}"
        + (" | Galactic Legend" if unit.is_galactic_legend else "")
        + (" | Capital Ship" if unit.is_capital_ship else ""),
        f"Categories: {', '.join(unit.categories)}",
    ]
    if unit.description:
        lines.append(f"Overview: {unit.description}")
    if unit.ship and (ship := catalog.get(unit.ship)):
        lines.append(f"Pilots/crews ship: {ship.name}")
    if unit.crew:
        crew = [c.name for c in map(catalog.get, unit.crew) if c]
        lines.append(f"Crew: {', '.join(crew)}")
    lines.append(f"Ability classes: {', '.join(unit.ability_classes)}")
    if unit.base_stats:
        stats = ", ".join(f"{k}: {v}" for k, v in unit.base_stats.items())
        lines.append(f"Base stats (swgoh.gg reference): {stats}")
    lines.append("\n## Abilities")
    lines.extend(f"### {a.render()}\n" for a in unit.abilities)
    lines.append(f"swgoh.gg page: {unit.url}")
    if unit.combat_type == "character":
        lines.append(
            f"Top-player mod data: {unit.url}data/mods/?filter_type=guilds_100_gp"
        )
    return "\n".join(lines)


def build_tools(catalog: Catalog, index: KnowledgeIndex) -> list[BaseTool]:
    @tool
    def get_unit(name: str) -> str:
        """Get a unit's full kit: categories, every ability's exact text with
        Zeta/Omicron markers, ship/crew links and base stats. Accepts full names,
        partial names or community acronyms (e.g. "SLKR", "JMK", "GAS")."""
        resolution = catalog.resolve(name)
        if resolution.unit is None:
            return _did_you_mean(name, resolution.candidates)
        text = render_unit(resolution.unit, catalog)
        if namesakes := catalog.namesakes(resolution.unit):
            others = ", ".join(u.name for u in namesakes)
            text += f"\n\nNote: different units with similar names exist: {others}."
        return text

    @tool
    def search_units(
        query: str,
        alignment: Alignment | None = None,
        combat_type: CombatType | None = None,
        keyword: str | None = None,
    ) -> str:
        """Semantic search over every unit's abilities and profile. Use it to find
        units by what they do ("dispels Taunt", "revives allies", "counters
        Bounty Hunters"). Set `keyword` to require an exact game term in the
        ability text (e.g. "Shatterpoint", "Payout")."""
        hits = index.search_units(
            query, alignment=alignment, combat_type=combat_type, keyword=keyword
        )
        if not hits:
            return "No matching units."
        lines = []
        for hit in hits:
            unit = catalog.get(hit.base_id)
            matched = f" | matching abilities: {', '.join(hit.abilities[:3])}"
            lines.append(f"- {unit.summary_line()}{matched if hit.abilities else ''}")
        return "\n".join(lines)

    @tool
    def list_units(
        categories: list[str] | None = None,
        alignment: Alignment | None = None,
        role: Role | None = None,
        combat_type: CombatType | None = None,
        leaders_only: bool = False,
    ) -> str:
        """List every unit matching ALL given filters. Use for faction rosters and
        team building, e.g. categories=["Bounty Hunter"], leaders_only=True."""
        resolved = []
        for name in categories or []:
            category = catalog.match_category(name)
            if category is None:
                suggestions = ", ".join(catalog.suggest_categories(name)) or "none"
                return f"Unknown category '{name}'. Closest: {suggestions}."
            resolved.append(category)
        units = catalog.filter(resolved, alignment, role, combat_type, leaders_only)
        if not units:
            return "No units match those filters."
        return f"{len(units)} units:\n" + "\n".join(
            f"- {u.summary_line()}" + (" [GL]" if u.is_galactic_legend else "")
            for u in units
        )

    @tool
    def list_categories() -> str:
        """List all faction/category tags and how many units carry each."""
        counts: dict[str, int] = defaultdict(int)
        for unit in catalog.units:
            for category in unit.categories:
                counts[category] += 1
        return "\n".join(f"{c} ({counts[c]})" for c in catalog.categories)

    @tool
    def lookup_effect(term: str) -> str:
        """Explain a unit-specific status effect or mechanic (e.g. "Shatterpoint",
        "Payout", "Resilient Defense") as defined in kits, and list the units
        whose abilities mention it."""
        key = term.strip().lower()
        # Group identical definitions so reprinted effects are shown once.
        definitions: dict[str, list[GlossaryEntry]] = defaultdict(list)
        for entry in catalog.kit_glossary:
            if entry.term.lower() == key:
                definitions[entry.definition].append(entry)
        mentions = [
            u.name
            for u in catalog.units
            if any(key in a.description.lower() for a in u.abilities)
        ]
        if not definitions and not mentions:
            return f"No kit mentions '{term}'. It may be a general mechanic."
        lines = []
        if definitions:
            lines.append(
                f"'{term}' as defined in kits. In a definition, \"this unit\" means "
                "whichever unit carries the effect; the ability text shows who "
                "applies it to whom."
            )
        for definition, entries in list(definitions.items())[:4]:
            first = entries[0]
            also = sorted({e.unit for e in entries[1:]} - {first.unit})
            lines += [
                f"\n- Definition: {definition}",
                f'  Source: {first.unit}\'s "{first.ability}"'
                + (f" (also {', '.join(also[:3])})" if also else ""),
                f"  Ability text: {first.usage[:500]}",
            ]
        if mentions:
            shown = ", ".join(mentions[:25])
            more = f" and {len(mentions) - 25} more" if len(mentions) > 25 else ""
            lines.append(
                f"\nUnits whose kits mention it ({len(mentions)}): {shown}{more}"
            )
        return "\n".join(lines)

    return [get_unit, search_units, list_units, list_categories, lookup_effect]
