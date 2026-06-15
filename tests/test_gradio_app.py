import gradio as gr
import polars as pl

from dota2tuned.ui.gradio_app import (
    APP_CSS,
    APP_HEAD,
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


def test_selected_hero_preview_has_text_tags_and_remove_button():
    html = _selected_hero_html(
        [1],
        {1: {"name": "Anti-Mage", "icon": "https://example.test/am.png", "roles": "Carry"}},
        "draft-allies-dropdown",
    )

    assert "hero-card" in html
    assert "<img" in html
    assert "<strong>Anti-Mage</strong>" in html
    assert "entity-tag" in html
    assert "hero-card-remove" in html
    assert "data-dota-target='draft-allies-dropdown'" in html


def test_selected_hero_preview_ignores_transient_null_values():
    html = _selected_hero_html(
        [None, "", "1", "bad"],
        {1: {"name": "Anti-Mage", "icon": "https://example.test/am.png", "roles": "Carry"}},
        "draft-allies-dropdown",
    )

    assert "<strong>Anti-Mage</strong>" in html
    assert "Hero None" not in html
    assert "Hero bad" not in html


def test_hero_multiselect_tolerates_gradio_scroll_null_payload():
    heroes = pl.DataFrame([{"hero_id": 1, "hero_name": "Anti-Mage"}])
    lookup, _ = _hero_lookup(heroes)
    dropdown = gr.Dropdown(
        choices=[("Anti-Mage", 1)],
        multiselect=True,
        allow_custom_value=True,
    )

    payload = dropdown.preprocess([None, 1])
    hero_ids, unknown = _parse_heroes(payload, lookup)

    assert hero_ids == [1]
    assert unknown == []


def test_dropdown_js_does_not_reinject_topbar():
    js = _dropdown_js({"Anti-Mage · Carry": {"src": "https://example.test/am.png", "kind": "hero"}})

    assert "choiceIcons" in js
    assert "ensureTopbar" not in js
    assert "dota-choice-tag" in js
    assert "openFromChevron" in js
    assert "removeSelectedHero" in js
    assert "syncSidebarView" in js
    assert "closestElement" in js
    assert "syncDropdownMenus" in js
    assert "isDropdownMenuScroll" in js


def test_css_uses_trajan_font_without_global_red_buttons():
    assert 'font-family: "Trajan Pro", "Goudy Trajan", "Noto Sans", serif' in APP_CSS
    assert ".gradio-container button {" not in APP_CSS
    assert ".app-main button {" in APP_CSS
    assert "text-transform: uppercase" in APP_CSS
    assert ".hero-dropdown .token" in APP_CSS
    assert "max-width: calc(100vw - 16px)" in APP_CSS
    assert "overscroll-behavior: contain" in APP_CSS


def test_css_keeps_dropdown_chevrons_inside_fields():
    assert ".dota-dropdown .icon-wrap" in APP_CSS
    assert "right: 8px !important" in APP_CSS
    assert "padding-right: 32px !important" in APP_CSS
    assert ".dota-dropdown .secondary-wrapper" in APP_CSS
    assert "background: transparent !important" in APP_CSS
    assert ".hero-dropdown input[autocomplete=\"off\"]" not in APP_CSS
    assert "min-width: 100% !important" not in APP_CSS


def test_css_shows_full_selected_hero_cards():
    assert ".hero-card > img" in APP_CSS
    assert "object-fit: contain" in APP_CSS
    assert ".hero-card strong" in APP_CSS
    assert "white-space: normal" in APP_CSS
    assert "text-overflow: clip" in APP_CSS


def test_loader_and_sidebar_logo_use_shared_red_flash():
    assert "d2-logo-red-flash" in APP_HEAD
    assert "#d2-startup-loader" in APP_HEAD
    assert "dota2tunedHideLoader" in APP_HEAD
    assert "DOTA2Tuned" in APP_HEAD
    assert "4.2s ease-in-out infinite" in APP_HEAD
    assert "html.d2-loading .gradio-container" in APP_HEAD
    assert ".app-sidebar-brand img" in APP_CSS
    assert "animation: d2-logo-red-flash 4.2s ease-in-out infinite" in APP_CSS


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


def test_css_has_dark_accessibility_and_perf_hardening():
    # Palette is a single source of truth; app is dark-only and AA-targeted.
    assert ":root {" in APP_CSS
    assert "--d2-fg" in APP_CSS
    # WCAG 2.2: visible focus + reduced motion.
    assert ":focus-visible" in APP_CSS
    assert "prefers-reduced-motion" in APP_CSS
    # Dark scrollbar is styled for both engines.
    assert "scrollbar-color:" in APP_CSS
    assert "::-webkit-scrollbar-thumb" in APP_CSS
    # Perf: webfonts no longer block first paint; no whole-tree font recalc.
    assert "font-display: swap" in APP_CSS
    assert ".gradio-container * {" not in APP_CSS


def test_dropdown_js_decorates_without_deleting_svelte_nodes():
    js = _dropdown_js(
        {"Anti-Mage · Carry": {"src": "https://example.test/am.png", "kind": "hero"}}
    )
    # Root cause of the red Error toast: we used to wipe Gradio/Svelte-owned nodes.
    assert 'option.textContent = ""' not in js
    assert "dota-decorated-rich" in js
    assert "decorateOption" in js
    # Stale Gradio-5 selectors removed.
    assert "svelte-select-list" not in js
    # Perf: debounced observer + throttled scroll; no blanket keyup re-decorate.
    assert "requestAnimationFrame" in js
    assert "scheduleDecorate" in js
    assert '"keyup"' not in js


def test_dropdown_js_refreshes_reused_search_option_nodes():
    js = _dropdown_js(
        {"Anti-Mage · Carry": {"src": "https://example.test/am.png", "kind": "hero"}}
    )

    assert "readOptionLabel" in js
    assert "resetOptionDecoration" in js
    assert "dotaChoiceLabel" in js
    assert 'attributeFilter: ["aria-label"]' in js
    assert "characterData: true" in js


def test_app_head_preconnects_without_theme_redirect():
    assert "preconnect" in APP_HEAD
    assert "cdn.steamstatic.com" in APP_HEAD
    # Dark mode is a single CSS palette now; no ?__theme=dark URL redirect.
    assert "__theme" not in APP_HEAD
    assert "__DOTA_LOGO_URL__" not in APP_HEAD


def test_css_forces_single_dark_palette():
    # Gradio's semantic theme vars are pinned to dark values unconditionally.
    assert "color-scheme: dark" in APP_CSS
    assert "--background-fill-primary: var(--neutral-950)" in APP_CSS
    assert "--body-text-color: var(--neutral-100)" in APP_CSS


def test_sidebar_js_toggles_view_class_without_inline_display():
    js = _dropdown_js({})
    # Views are always rendered; JS toggles the .d2-active class (CSS hides the
    # rest). It must NOT set inline display/hidden, which fought Svelte and left
    # a freshly-selected tab blank until it was re-selected.
    assert "view.style.display" not in js
    assert "view.hidden" not in js
    assert 'classList.toggle("d2-active"' in js
    assert 'setAttribute("aria-hidden"' in js
    # CSS owns hiding the inactive views.
    assert ".app-view:not(.d2-active)" in APP_CSS
