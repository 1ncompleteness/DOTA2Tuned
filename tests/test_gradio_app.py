import polars as pl

from dota2tuned.ui.gradio_app import _hero_lookup, _parse_heroes


def test_parse_heroes_accepts_names_and_ids():
    heroes = pl.DataFrame(
        [
            {"hero_id": 1, "hero_name": "Anti-Mage"},
            {"hero_id": 44, "hero_name": "Phantom Assassin"},
            {"hero_id": 30, "hero_name": "Witch Doctor"},
        ]
    )
    lookup, _ = _hero_lookup(heroes)

    hero_ids, unknown = _parse_heroes("Phantom Assassin, Witch Doctor, 1", lookup)

    assert hero_ids == [44, 30, 1]
    assert unknown == []


def test_parse_heroes_reports_unknown_names():
    heroes = pl.DataFrame([{"hero_id": 1, "hero_name": "Anti-Mage"}])
    lookup, _ = _hero_lookup(heroes)

    hero_ids, unknown = _parse_heroes("Anti-Mage, Banana King", lookup)

    assert hero_ids == [1]
    assert unknown == ["Banana King"]
