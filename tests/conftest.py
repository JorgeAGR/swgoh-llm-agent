import pytest

from swgoh_agent.catalog import Catalog
from swgoh_agent.models import Dataset
from swgoh_agent.sources.swgohgg import build_units


def _ability(owner, ability_id, name, type_, description, **flags):
    return {
        "base_id": ability_id.upper(),
        "ability_id": ability_id,
        "name": name,
        "description": description,
        "type": type_,
        "is_zeta": flags.get("zeta", False),
        "is_omicron": flags.get("omicron", False),
        "omicron_mode": flags.get("omicron_mode", 1),
        "omicron_battle_types": flags.get("battle_types", []),
        "character_base_id": owner if not flags.get("ship") else None,
        "ship_base_id": owner if flags.get("ship") else None,
    }


def _unit(base_id, name, categories, role="Attacker", alignment="Light Side", **kw):
    return {
        "base_id": base_id,
        "name": name,
        "url": f"//swgoh.gg/units/{base_id.lower()}/",
        "description": "",
        "alignment": alignment,
        "role": role,
        "categories": categories,
        "ability_classes": ["Dispel"],
        **kw,
    }


CHARACTERS = [
    _unit("MACEWINDU", "Mace Windu", ["Leader", "Jedi"], ship="MACESHIP"),
    _unit(
        "JEDIMASTERMACEWINDU",
        "Jedi Master Mace Windu",
        ["Leader", "Jedi", "Galactic Republic"],
    ),
    _unit(
        "SUPREMELEADERKYLOREN",
        "Supreme Leader Kylo Ren",
        ["Leader", "First Order", "Galactic Legend"],
        alignment="Dark Side",
    ),
    _unit("PADME", "Padmé Amidala", ["Leader", "Galactic Republic"], role="Support"),
    _unit("BOSSK", "Bossk", ["Leader", "Bounty Hunter"], role="Tank"),
    _unit("DASHRENDAR", "Dash Rendar", ["Scoundrel"]),
    _unit("DARTHREVAN", "Darth Revan", ["Leader", "Sith"], alignment="Dark Side"),
]
SHIPS = [_unit("MACESHIP", "Mace's Starfighter", ["Jedi"], capital_ship=False)]
ABILITIES = [
    _ability("MACEWINDU", "leaderability_macewindu", "Take a Seat", 3, "Jedi buff."),
    _ability(
        "MACEWINDU",
        "uniqueability_macewindu02",
        "Sense Weakness",
        4,
        "Inflict Shatterpoint.\\n\\nShatterpoint: Receiving damage dispels it."
        "\\n\\nWhile in Territory Wars: Extra effects.",
        zeta=True,
        omicron=True,
        omicron_mode=8,
        battle_types=["TERRITORY_WAR_BATTLE"],
    ),
    _ability("BOSSK", "specialability_bossk01", "Payout", 2, "Bossk gets paid."),
    _ability(
        "SUPREMELEADERKYLOREN",
        "ultimateability_slkr",
        "I Am All The Sith",
        6,
        "Ultimate.",
    ),
    _ability("MACESHIP", "specialability_maceship01", "Crew Strike", 5, "Pew.", ship=1),
]


@pytest.fixture
def units():
    return build_units(CHARACTERS, SHIPS, ABILITIES)


@pytest.fixture
def catalog(units):
    return Catalog(
        Dataset(fetched_at="2026-09-24T00:00:00", source="test", units=units)
    )
