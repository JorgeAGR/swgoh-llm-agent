import pytest


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Mace Windu", "Mace Windu"),
        ("jedi master mace windu", "Jedi Master Mace Windu"),
        ("SLKR", "Supreme Leader Kylo Ren"),
        ("JMMW", "Jedi Master Mace Windu"),
        ("padme", "Padmé Amidala"),  # accents are ignored
        ("DR", "Darth Revan"),  # curated alias beats the Dash Rendar collision
        ("bosk", "Bossk"),  # typo
        ("macewindu", "Mace Windu"),  # base id
    ],
)
def test_resolve(catalog, query, expected):
    assert catalog.resolve(query).unit.name == expected


def test_resolve_ambiguous_returns_candidates(catalog):
    result = catalog.resolve("Mace")
    assert result.unit is None
    assert {u.name for u in result.candidates} >= {
        "Mace Windu",
        "Jedi Master Mace Windu",
    }


def test_namesakes_flag_variants(catalog):
    mace = catalog.resolve("Mace Windu").unit
    assert [u.name for u in catalog.namesakes(mace)] == ["Jedi Master Mace Windu"]


def test_match_category_tolerates_case_and_plurals(catalog):
    assert catalog.match_category("bounty hunters") == "Bounty Hunter"
    assert catalog.match_category("nope") is None


def test_filter(catalog):
    names = {u.name for u in catalog.filter(["Jedi"], combat_type="character")}
    assert names == {"Mace Windu", "Jedi Master Mace Windu"}
    dark_leaders = catalog.filter(alignment="dark side", leaders_only=True)
    assert {u.name for u in dark_leaders} == {"Darth Revan", "Supreme Leader Kylo Ren"}


def test_kit_glossary_skips_conditional_clauses(catalog):
    [entry] = catalog.kit_glossary
    assert (entry.term, entry.unit, entry.ability) == (
        "Shatterpoint",
        "Mace Windu",
        "Sense Weakness",
    )
    assert entry.usage == "Inflict Shatterpoint."
