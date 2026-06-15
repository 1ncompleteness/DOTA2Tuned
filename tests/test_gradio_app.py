import polars as pl

from dota2tuned.ui.gradio_app import (
    APP_CSS,
    _dropdown_js,
    _format_item_time,
    _hero_aliases,
    _hero_choices,
    _hero_lookup,
    _parse_heroes,
    _selected_hero_html,
)


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


def test_parse_heroes_accepts_common_aliases():
    heroes = pl.DataFrame(
        [
            {"hero_id": 1, "hero_name": "Anti-Mage"},
            {"hero_id": 5, "hero_name": "Crystal Maiden"},
            {"hero_id": 44, "hero_name": "Phantom Assassin"},
        ]
    )
    lookup, _ = _hero_lookup(heroes)

    hero_ids, unknown = _parse_heroes("PA, CM, AM", lookup)

    assert hero_ids == [44, 5, 1]
    assert unknown == []


def test_aliases_drop_single_letter_fallbacks():
    assert "M" not in _hero_aliases("Mirana")
    assert "M" not in _hero_aliases("Meepo")
    assert "M" not in _hero_aliases("Medusa")
    assert "Z" not in _hero_aliases("Zeus")
    assert "NS" not in _hero_aliases("Naga Siren")
    assert "VS" not in _hero_aliases("Void Spirit")
    assert "AM" in _hero_aliases("Anti-Mage")
    assert "QoP" in _hero_aliases("Queen of Pain")


def test_hero_choice_labels_do_not_show_single_letter_aliases():
    heroes = pl.DataFrame(
        [
            {"hero_id": 9, "hero_name": "Mirana", "roles": "Carry,Support"},
            {"hero_id": 39, "hero_name": "Queen of Pain", "roles": "Carry,Nuker,Escape"},
        ]
    )

    labels = [label for label, _ in _hero_choices(heroes)]

    assert labels == [
        "Mirana · Carry, Support",
        "Queen of Pain (QoP) · Carry, Nuker, Escape",
    ]


def test_selected_hero_preview_is_image_only_card():
    html = _selected_hero_html(
        [1],
        {1: {"name": "Anti-Mage", "icon": "https://example.test/am.png", "roles": "Carry"}},
    )

    assert "hero-card" in html
    assert "<img" in html
    assert "<strong>" not in html
    assert "entity-tag" not in html


def test_dropdown_js_does_not_reinject_topbar():
    js = _dropdown_js({"Anti-Mage · Carry": {"src": "https://example.test/am.png", "kind": "hero"}})

    assert "choiceIcons" in js
    assert "ensureTopbar" not in js
    assert "dota-choice-tag" in js


def test_css_uses_dota_font_without_global_red_buttons():
    assert 'font-family: "Radiance", "Noto Sans", sans-serif' in APP_CSS
    assert ".gradio-container button {" not in APP_CSS
    assert ".app-main button {" in APP_CSS


def test_parse_heroes_accepts_dropdown_values():
    heroes = pl.DataFrame([{"hero_id": 44, "hero_name": "Phantom Assassin"}])
    lookup, _ = _hero_lookup(heroes)

    hero_ids, unknown = _parse_heroes([44], lookup)

    assert hero_ids == [44]
    assert unknown == []


def test_parse_heroes_reports_unknown_names():
    heroes = pl.DataFrame([{"hero_id": 1, "hero_name": "Anti-Mage"}])
    lookup, _ = _hero_lookup(heroes)

    hero_ids, unknown = _parse_heroes("Anti-Mage, Banana King", lookup)

    assert hero_ids == [1]
    assert unknown == ["Banana King"]


def test_format_item_time_labels_pre_game_buys():
    assert _format_item_time(-90) == "pre-game"
    assert _format_item_time(180) == "3.0 min"
