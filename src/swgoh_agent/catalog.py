"""In-memory catalog of SWGOH units with name resolution and filtering."""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from swgoh_agent.models import Dataset, Unit

# Community shorthand that initials alone can't produce or would get wrong.
ALIASES = {
    "gas": "General Skywalker",
    "dr": "Darth Revan",
    "jtr": "Rey (Jedi Training)",
    "gg": "General Grievous",
    "vader": "Darth Vader",
    "palp": "Emperor Palpatine",
    "palpatine": "Emperor Palpatine",
    "sidious": "Darth Sidious",
    "yoda": "Grand Master Yoda",
    "gl rey": "Rey",
    "gl leia": "Leia Organa",
    "gl ahsoka": "Ahsoka Tano",
    "gl jabba": "Jabba the Hutt",
    "gl hondo": "Pirate King Hondo Ohnaka",
    "pkho": "Pirate King Hondo Ohnaka",
}

_STOPWORDS = {"the", "of", "a", "an"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def initialisms(name: str) -> set[str]:
    words = normalize(name).split()
    full = "".join(w[0] for w in words)
    short = "".join(w[0] for w in words if w not in _STOPWORDS)
    return {i for i in (full, short) if len(i) >= 2}


@dataclass
class Resolution:
    unit: Unit | None
    candidates: list[Unit] = field(default_factory=list)


@dataclass
class GlossaryEntry:
    term: str
    definition: str
    unit: str
    ability: str
    usage: str  # the ability's main text, showing who applies the effect to whom


class Catalog:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.units = dataset.units
        self._by_id = {u.base_id: u for u in self.units}
        self._by_name = {normalize(u.name): u for u in self.units}

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        if not path.exists():
            raise FileNotFoundError(
                f"No unit data at {path}. Run `swgoh-agent data refresh` first."
            )
        return cls(Dataset.model_validate_json(path.read_text()))

    def get(self, base_id: str) -> Unit | None:
        return self._by_id.get(base_id)

    # --- name resolution -------------------------------------------------

    @cached_property
    def _initials(self) -> dict[str, list[Unit]]:
        index: dict[str, list[Unit]] = {}
        for unit in self.units:
            for key in initialisms(unit.name):
                index.setdefault(key, []).append(unit)
        return index

    def resolve(self, query: str) -> Resolution:
        """Match a name, base id, acronym (e.g. "SLKR") or partial name to a unit."""
        key = normalize(query)
        if not key:
            return Resolution(None)
        if unit := self._by_name.get(key) or self._by_id.get(query.strip().upper()):
            return Resolution(unit)
        if alias := ALIASES.get(key):
            return Resolution(self._by_name[normalize(alias)])

        if matches := self._initials.get(key.replace(" ", "")):
            if len(matches) == 1:
                return Resolution(matches[0])
            return Resolution(None, matches)

        pattern = re.compile(rf"\b{re.escape(key)}\b")
        substring = [u for n, u in self._by_name.items() if pattern.search(n)]
        if len(substring) == 1:
            return Resolution(substring[0])
        if substring:
            return Resolution(None, substring)

        close = difflib.get_close_matches(key, self._by_name, n=5, cutoff=0.6)
        if len(close) == 1:
            return Resolution(self._by_name[close[0]])
        return Resolution(None, [self._by_name[n] for n in close])

    def namesakes(self, unit: Unit) -> list[Unit]:
        """Other units whose names contain this unit's name, e.g. Mace Windu ->
        Jedi Master Mace Windu. Helps catch picking the wrong variant."""
        pattern = re.compile(rf"\b{re.escape(normalize(unit.name))}\b")
        return [
            u for n, u in self._by_name.items() if u is not unit and pattern.search(n)
        ]

    # --- filtering -------------------------------------------------------

    @cached_property
    def categories(self) -> list[str]:
        return sorted({c for u in self.units for c in u.categories})

    def match_category(self, name: str) -> str | None:
        lookup = {normalize(c): c for c in self.categories}
        key = normalize(name)
        if key in lookup:
            return lookup[key]
        # Tolerate plurals like "Bounty Hunters" or "Jawas".
        return lookup.get(key.removesuffix("s")) or lookup.get(key.removesuffix("es"))

    def suggest_categories(self, name: str) -> list[str]:
        lookup = {normalize(c): c for c in self.categories}
        close = difflib.get_close_matches(normalize(name), lookup, n=5, cutoff=0.5)
        return [lookup[c] for c in close]

    def filter(
        self,
        categories: list[str] = (),
        alignment: str | None = None,
        role: str | None = None,
        combat_type: str | None = None,
        leaders_only: bool = False,
    ) -> list[Unit]:
        wanted = set(categories)
        return [
            u
            for u in self.units
            if wanted <= set(u.categories)
            and (not alignment or normalize(u.alignment) == normalize(alignment))
            and (not role or normalize(u.role) == normalize(role))
            and (not combat_type or u.combat_type == combat_type)
            and (not leaders_only or u.is_leader)
        ]

    # --- glossary --------------------------------------------------------

    @cached_property
    def kit_glossary(self) -> list[GlossaryEntry]:
        """Effects defined inline in kits, e.g. "Shatterpoint: Receiving damage..."."""
        entries = []
        for unit in self.units:
            for ability in unit.abilities:
                usage, *lines = ability.description.split("\n")
                for line in lines:
                    match = _DEFINITION.match(line.strip())
                    if match and not match.group(1).startswith(_CONDITIONAL):
                        entries.append(
                            GlossaryEntry(
                                term=match.group(1),
                                definition=match.group(2),
                                unit=unit.name,
                                ability=ability.name,
                                usage=usage,
                            )
                        )
        return entries


_DEFINITION = re.compile(r"^([A-Z][\w'\- ]{1,40}):\s+(.+)$")
_CONDITIONAL = ("While ", "If ", "When ", "During ", "At ", "Against ", "Instead")
