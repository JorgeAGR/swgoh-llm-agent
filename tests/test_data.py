from swgoh_agent.agent.tools import render_unit
from swgoh_agent.knowledge.index import unit_documents


def by_name(units, name):
    return next(u for u in units if u.name == name)


def test_build_units_links_kits_ships_and_crew(units):
    mace = by_name(units, "Mace Windu")
    assert [a.name for a in mace.abilities] == ["Take a Seat", "Sense Weakness"]
    assert mace.url == "https://swgoh.gg/units/macewindu/"
    ship = by_name(units, "Mace's Starfighter")
    assert ship.combat_type == "ship"
    assert ship.crew == ["MACEWINDU"]
    assert ship.abilities[0].kind == "Crew"


def test_parse_ability_kinds_and_omicron_modes(units):
    mace = by_name(units, "Mace Windu")
    sense = mace.abilities[1]
    assert sense.kind == "Unique"
    assert sense.omicron_modes == ["Territory Wars"]
    assert "\n\nShatterpoint:" in sense.description  # escaped newlines restored
    assert "[Zeta] [Omicron: Territory Wars]" in sense.render()
    slkr = by_name(units, "Supreme Leader Kylo Ren")
    assert slkr.abilities[0].kind == "Ultimate"
    assert slkr.is_galactic_legend


def test_render_unit_mentions_ship_and_mod_link(catalog):
    text = render_unit(catalog.get("MACEWINDU"), catalog)
    assert "Pilots/crews ship: Mace's Starfighter" in text
    assert "data/mods/" in text


def test_unit_documents_one_per_ability_plus_profile(units):
    docs = unit_documents(by_name(units, "Mace Windu"))
    assert [d.metadata["kind"] for d in docs] == ["profile", "ability", "ability"]
    assert all("Unit: Mace Windu" in d.page_content for d in docs)
