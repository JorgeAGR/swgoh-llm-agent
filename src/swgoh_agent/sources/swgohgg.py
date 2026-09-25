"""Fetch unit data from swgoh.gg.

The public JSON API provides units, categories and full ability kits. Base stats
are only published on each unit's HTML page, so those are scraped optionally.
swgoh.gg sits behind Cloudflare, so requests impersonate a real browser.
"""

import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from bs4 import BeautifulSoup
from curl_cffi import requests

from swgoh_agent.models import Ability, Dataset, Unit

log = logging.getLogger(__name__)

BASE_URL = "https://swgoh.gg"

# Ability "type" field in the API. Type 4 covers unique, hardware and contract
# abilities, so the ability_id prefix is used to tell those apart.
_KIND_BY_PREFIX = {
    "basicability": "Basic",
    "specialability": "Special",
    "leaderability": "Leader",
    "uniqueability": "Unique",
    "ultimateability": "Ultimate",
    "hardwareability": "Hardware",
    "contractability": "Contract",
}
_CREW_ABILITY_TYPE = 5

_OMICRON_BATTLE_TYPES = {
    "TERRITORY_WAR_BATTLE": "Territory Wars",
    "TERRITORY_TOURNAMENT_BATTLE": "Grand Arena",
    "TERRITORY_STRIKE_BATTLE": "Territory Battles",
    "TERRITORY_COVERT_BATTLE": "Territory Battles",
    "GUILD_RAID": "Raids",
    "CONQUEST": "Conquest",
}
# omicron_mode values from the game data, used when battle types are absent.
_OMICRON_MODES = {
    4: "Raids",
    5: "Territory Battles",
    6: "Territory Battles",
    7: "Territory Battles",
    8: "Territory Wars",
    9: "Grand Arena",
    11: "Conquest",
    12: "Galactic Challenges",
    14: "Grand Arena (3v3)",
    15: "Grand Arena (5v5)",
}


class SwgohGGClient:
    def __init__(self, delay: float = 1.0, timeout: float = 30.0):
        self.session = requests.Session(impersonate="chrome")
        self.delay = delay
        self.timeout = timeout

    def _get(self, path: str) -> requests.Response:
        response = self.session.get(f"{BASE_URL}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response

    def get_json(self, endpoint: str) -> list[dict]:
        return self._get(f"/api/{endpoint}/").json()

    def get_base_stats(self, unit_url: str) -> dict[str, str]:
        path = unit_url.removeprefix(BASE_URL)
        soup = BeautifulSoup(self._get(path).text, "html.parser")
        stats = {}
        for row in soup.select(".stat-table-data__entry"):
            label = row.select_one(".stat-table-data__entry-primary-label")
            value = row.select_one(".stat-table-data__entry-primary-value")
            if label and value:
                stats[label.get_text(strip=True)] = value.get_text(strip=True)
        return stats


def _clean_text(text: str) -> str:
    # Descriptions arrive with escaped newlines ("\\n") separating effect glossaries.
    return text.replace("\\n", "\n").strip()


def _omicron_modes(raw: dict) -> list[str]:
    if not raw["is_omicron"]:
        return []
    modes = [_OMICRON_BATTLE_TYPES.get(t, t) for t in raw["omicron_battle_types"]]
    if not modes and raw["omicron_mode"] in _OMICRON_MODES:
        modes = [_OMICRON_MODES[raw["omicron_mode"]]]
    return list(dict.fromkeys(modes))


def parse_ability(raw: dict) -> Ability:
    prefix = raw["ability_id"].split("_")[0]
    kind = _KIND_BY_PREFIX.get(prefix, prefix.removesuffix("ability").title())
    if raw["type"] == _CREW_ABILITY_TYPE:
        kind = "Crew"
    return Ability(
        base_id=raw["base_id"],
        name=raw["name"].strip(),
        kind=kind,
        description=_clean_text(raw["description"]),
        is_zeta=raw["is_zeta"],
        is_omicron=raw["is_omicron"],
        omicron_modes=_omicron_modes(raw),
    )


def build_units(
    characters: list[dict], ships: list[dict], abilities: list[dict]
) -> list[Unit]:
    kits: dict[str, list[Ability]] = {}
    for raw in abilities:
        owner = raw["character_base_id"] or raw["ship_base_id"]
        if owner:
            kits.setdefault(owner, []).append(parse_ability(raw))

    crews: dict[str, list[str]] = {}
    for raw in characters:
        if raw.get("ship"):
            crews.setdefault(raw["ship"], []).append(raw["base_id"])

    units = []
    for raw, combat_type in [
        *((c, "character") for c in characters),
        *((s, "ship") for s in ships),
    ]:
        units.append(
            Unit(
                base_id=raw["base_id"],
                name=raw["name"].strip(),
                url="https:" + raw["url"],
                combat_type=combat_type,
                alignment=raw["alignment"],
                role=raw["role"],
                description=raw.get("description", ""),
                categories=raw["categories"],
                ability_classes=raw["ability_classes"],
                abilities=kits.get(raw["base_id"], []),
                ship=raw.get("ship") or None,
                crew=crews.get(raw["base_id"], []),
                is_capital_ship=raw.get("capital_ship", False),
            )
        )
    return sorted(units, key=lambda u: u.name.lower())


def fetch_base_stats(
    client: SwgohGGClient,
    units: Iterable[Unit],
    on_progress: Callable[[Unit], None] | None = None,
) -> None:
    """Populate `base_stats` in place by scraping each unit page."""
    for unit in units:
        try:
            unit.base_stats = client.get_base_stats(unit.url)
        except Exception as exc:  # one bad page shouldn't abort a long crawl
            log.warning("Could not fetch stats for %s: %s", unit.name, exc)
        if on_progress:
            on_progress(unit)
        time.sleep(client.delay)


def fetch_dataset(client: SwgohGGClient) -> Dataset:
    units = build_units(
        client.get_json("characters"),
        client.get_json("ships"),
        client.get_json("abilities"),
    )
    return Dataset(
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=BASE_URL,
        units=units,
    )
