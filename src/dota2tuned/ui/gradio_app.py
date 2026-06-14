from __future__ import annotations

import json
import os
import re
from urllib.parse import quote

import gradio as gr
import polars as pl

from dota2tuned.config import get_settings
from dota2tuned.rag import Retriever
from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput
from dota2tuned.storage import read_parquet
from dota2tuned.train_predictor import predict_draft_win

ASSET_BASE_URL = "https://cdn.cloudflare.steamstatic.com"
DOTA_LOGO_URL = f"{ASSET_BASE_URL}/apps/dota2/images/dota_react/global/dota2_logo_symbol.png"


def _svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def _badge_icon(body: str, accent: str = "#d9b166") -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<rect x='2' y='2' width='28' height='28' rx='6' fill='#151817'/>"
        f"<rect x='2.75' y='2.75' width='26.5' height='26.5' rx='5.25' "
        f"fill='none' stroke='{accent}' stroke-opacity='.42' stroke-width='1.5'/>"
        f"<g fill='none' stroke='{accent}' stroke-width='2.35' "
        f"stroke-linecap='round' stroke-linejoin='round'>{body}</g>"
        "</svg>"
    )
    return _svg_data_uri(svg)


ROLE_ICON_BODIES = {
    "Carry": (
        "#d9b166",
        (
            "<path d='M10 7l12 18'/>",
            "<path d='M22 7L10 25'/>",
            "<path d='M8 7h5'/>",
            "<path d='M19 7h5'/>",
        ),
    ),
    "Support": (
        "#8fd9b6",
        (
            "<circle cx='16' cy='16' r='9'/>",
            "<path d='M16 9v14'/>",
            "<path d='M9 16h14'/>",
        ),
    ),
    "Nuker": (
        "#e69b68",
        (
            "<path d='M16 5l2.5 7 7-2.5-4 6.5 6.5 3-7.5 1.4-1 7.6"
            "-4.5-5.8-4.5 5.8-1-7.6L4 20l6.5-3-4-6.5 7 2.5z'/>",
        ),
    ),
    "Disabler": (
        "#b7a6ff",
        (
            "<path d='M12 13l8 8'/>",
            "<path d='M10 18a5 5 0 010-7l2-2a5 5 0 017 0'/>",
            "<path d='M22 14a5 5 0 010 7l-2 2a5 5 0 01-7 0'/>",
        ),
    ),
    "Jungler": (
        "#9edb74",
        (
            "<path d='M8 22c10-1 15-8 16-17-8 1-15 7-16 17z'/>",
            "<path d='M8 22c3-4 7-7 12-10'/>",
        ),
    ),
    "Durable": (
        "#a9c7e8",
        ("<path d='M16 5l10 4v7c0 6-4 10-10 12C10 26 6 22 6 16V9z'/>",),
    ),
    "Escape": (
        "#8fc8ff",
        (
            "<path d='M8 22L24 6'/>",
            "<path d='M15 6h9v9'/>",
            "<path d='M7 12v13h13'/>",
        ),
    ),
    "Pusher": (
        "#e0c47b",
        (
            "<path d='M10 26h12'/>",
            "<path d='M12 26V12h8v14'/>",
            "<path d='M10 12h12'/>",
            "<path d='M12 12V8h8v4'/>",
            "<path d='M14 8V5h4v3'/>",
        ),
    ),
    "Initiator": (
        "#ee8f7d",
        (
            "<path d='M10 27V6'/>",
            "<path d='M10 7h13l-3 5 3 5H10'/>",
        ),
    ),
}

ROLE_ICON_URLS = {
    role: _badge_icon("".join(paths), accent)
    for role, (accent, paths) in ROLE_ICON_BODIES.items()
}

ROLE_OPTIONS = [
    ("Position 1 Carry", "carry"),
    ("Position 2 Mid", "mid"),
    ("Position 3 Offlane", "offlane"),
    ("Position 4 Soft support", "soft support"),
    ("Position 5 Hard support", "hard support"),
]

SCOPE_OPTIONS = [
    ("Pro matches", "pro"),
    ("High-rank pubs", "high-rank"),
    ("Public matches", "public"),
]

NAV_OPTIONS = [
    "Draft Coach",
    "Hero Meta",
    "Tuned Model",
    "Match Predictor",
    "Builds",
    "Draft Lab",
    "Data Freshness",
]

NAV_ICON_BODIES = {
    "Draft Coach": (
        "#d9b166",
        (
            "<path d='M8 8h16v16H8z'/>",
            "<path d='M8 13h16'/>",
            "<path d='M13 8v16'/>",
            "<path d='M18 8v16'/>",
        ),
    ),
    "Hero Meta": (
        "#e08a62",
        (
            "<path d='M16 5l3.2 6.5 7.2 1-5.2 5 1.2 7.1L16 21.2l-6.4 3.4 1.2-7.1-5.2-5 7.2-1z'/>",
        ),
    ),
    "Tuned Model": (
        "#9fc7ff",
        (
            "<path d='M10 12a6 6 0 0112 0v8a4 4 0 01-4 4h-4a4 4 0 01-4-4z'/>",
            "<path d='M13 10v14'/>",
            "<path d='M19 10v14'/>",
            "<path d='M8 16h16'/>",
        ),
    ),
    "Match Predictor": (
        "#8fd19e",
        (
            "<path d='M7 23V9'/>",
            "<path d='M13 23V13'/>",
            "<path d='M19 23V6'/>",
            "<path d='M25 23H5'/>",
        ),
    ),
    "Builds": (
        "#d7a36f",
        (
            "<path d='M10 22l12-12'/>",
            "<path d='M18 6l6 6'/>",
            "<path d='M8 24l-2 2'/>",
            "<path d='M12 20l-4 4'/>",
        ),
    ),
    "Draft Lab": (
        "#c09cff",
        (
            "<path d='M12 6h8'/>",
            "<path d='M14 6v7l-5 9a4 4 0 003.5 6h7a4 4 0 003.5-6l-5-9V6'/>",
            "<path d='M11 22h10'/>",
        ),
    ),
    "Data Freshness": (
        "#86d8d0",
        (
            "<path d='M24 12a8 8 0 00-14-4l-2 2'/>",
            "<path d='M8 5v5h5'/>",
            "<path d='M8 20a8 8 0 0014 4l2-2'/>",
            "<path d='M24 27v-5h-5'/>",
        ),
    ),
}

NAV_ICON_URLS = {
    label: _badge_icon("".join(paths), accent)
    for label, (accent, paths) in NAV_ICON_BODIES.items()
}

NAV_ICON_CSS = "\n".join(
    f".app-sidebar .app-nav label:nth-of-type({index})::before "
    f"{{ background-image: url('{NAV_ICON_URLS[label]}'); }}"
    for index, label in enumerate(NAV_OPTIONS, start=1)
)

ROLE_DROPDOWN_ICONS = {
    "carry": ROLE_ICON_URLS["Carry"],
    "mid": ROLE_ICON_URLS["Nuker"],
    "offlane": ROLE_ICON_URLS["Durable"],
    "soft support": ROLE_ICON_URLS["Support"],
    "hard support": ROLE_ICON_URLS["Support"],
}

SCOPE_ICON_BODIES = {
    "pro": (
        "#e1c16e",
        (
            "<path d='M10 7h12v4a6 6 0 01-12 0z'/>",
            "<path d='M9 9H6a4 4 0 004 4'/>",
            "<path d='M23 9h3a4 4 0 01-4 4'/>",
            "<path d='M16 17v5'/>",
            "<path d='M12 25h8'/>",
        ),
    ),
    "high-rank": (
        "#8fc8ff",
        (
            "<path d='M7 23l6-6 4 4 8-11'/>",
            "<path d='M18 10h7v7'/>",
        ),
    ),
    "public": (
        "#b7d99a",
        (
            "<path d='M12 16a4 4 0 100-8 4 4 0 000 8z'/>",
            "<path d='M20 17a3.5 3.5 0 100-7 3.5 3.5 0 000 7z'/>",
            "<path d='M5 25c1-4 4-6 7-6s6 2 7 6'/>",
            "<path d='M16 24c1-3 3-5 6-5 2 0 4 1 5 4'/>",
        ),
    ),
}

SCOPE_ICON_URLS = {
    scope: _badge_icon("".join(paths), accent)
    for scope, (accent, paths) in SCOPE_ICON_BODIES.items()
}

COMMON_HERO_ALIASES = {
    "Anti-Mage": ["AM"],
    "Ancient Apparition": ["AA"],
    "Bounty Hunter": ["BH"],
    "Brewmaster": ["Brew"],
    "Bristleback": ["BB"],
    "Centaur Warrunner": ["CW", "Centaur"],
    "Chaos Knight": ["CK"],
    "Clockwerk": ["Clock"],
    "Crystal Maiden": ["CM"],
    "Dark Seer": ["DS"],
    "Dark Willow": ["DW"],
    "Dawnbreaker": ["DB"],
    "Death Prophet": ["DP"],
    "Dragon Knight": ["DK"],
    "Drow Ranger": ["Drow", "DR"],
    "Earth Spirit": ["ES"],
    "Earthshaker": ["ES"],
    "Elder Titan": ["ET"],
    "Ember Spirit": ["Ember", "ES"],
    "Faceless Void": ["FV", "Void"],
    "Keeper of the Light": ["KOTL"],
    "Kunkka": ["Boat"],
    "Legion Commander": ["LC"],
    "Lifestealer": ["LS", "Naix"],
    "Lone Druid": ["LD"],
    "Monkey King": ["MK"],
    "Nature's Prophet": ["NP", "Furion"],
    "Necrophos": ["Necro"],
    "Night Stalker": ["NS"],
    "Nyx Assassin": ["Nyx", "NA"],
    "Outworld Devourer": ["OD"],
    "Phantom Assassin": ["PA"],
    "Phantom Lancer": ["PL"],
    "Queen of Pain": ["QoP"],
    "Sand King": ["SK"],
    "Shadow Demon": ["SD"],
    "Shadow Fiend": ["SF"],
    "Shadow Shaman": ["SS"],
    "Skywrath Mage": ["Sky", "SM"],
    "Spirit Breaker": ["SB", "Bara"],
    "Storm Spirit": ["Storm", "SS"],
    "Templar Assassin": ["TA"],
    "Treant Protector": ["Treant", "TP"],
    "Underlord": ["Pitlord"],
    "Vengeful Spirit": ["VS"],
    "Venomancer": ["Veno"],
    "Windranger": ["WR"],
    "Winter Wyvern": ["WW"],
    "Witch Doctor": ["WD"],
    "Wraith King": ["WK", "SK"],
}

AMBIGUOUS_GENERATED_ALIASES = {"NS", "VS"}

APP_CSS = """
@font-face {
  font-family: "Radiance";
  src: url("https://cdn.steamstatic.com/apps/dota2/fonts/radiance.woff") format("woff");
  font-weight: 400;
  font-style: normal;
}
@font-face {
  font-family: "Radiance";
  src: url("https://cdn.steamstatic.com/apps/dota2/fonts/radiance-semibold.woff") format("woff");
  font-weight: 700;
  font-style: normal;
}
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  color: #dcdedf;
  font-family: "Radiance", "Noto Sans", sans-serif !important;
  background:
    radial-gradient(circle at 18% -8%, rgba(255, 96, 70, 0.18), transparent 32rem),
    radial-gradient(circle at 92% 3%, rgba(171, 140, 59, 0.13), transparent 28rem),
    linear-gradient(180deg, #05060a 0%, #111318 46%, #090a0d 100%);
}
.gradio-container .contain,
.gradio-container main,
.gradio-container .main {
  max-width: none !important;
  width: 100% !important;
}
.gradio-container * {
  font-family: "Radiance", "Noto Sans", sans-serif !important;
  letter-spacing: 0 !important;
}
.gradio-container label,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: #dcdedf !important;
}
.gradio-container .form,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .input-container,
.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap-inner {
  border-color: rgba(103, 112, 123, 0.42) !important;
  background: rgba(22, 22, 24, 0.70) !important;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.035) !important;
}
.gradio-container textarea,
.gradio-container input {
  color: #efe5bb !important;
}
.gradio-container button {
  border: 1px solid rgba(255, 96, 70, 0.42) !important;
  border-radius: 3px !important;
  background:
    linear-gradient(180deg, rgba(149, 46, 70, 0.92), rgba(83, 31, 28, 0.94)) !important;
  color: #fff !important;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.34) !important;
  transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease !important;
}
.gradio-container button:hover {
  border-color: rgba(255, 96, 70, 0.74) !important;
  background:
    linear-gradient(180deg, rgba(234, 105, 83, 0.98), rgba(126, 43, 34, 0.98)) !important;
  transform: translateY(-1px);
}
.gradio-container button:active {
  transform: translateY(0);
}
.app-title {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.app-title strong {
  color: rgba(252, 245, 230, 0.94);
  font-size: 18px;
  line-height: 20px;
  letter-spacing: 0;
}
.app-title span {
  color: rgba(245, 245, 235, 0.62);
  font-size: 12px;
  line-height: 15px;
}
.app-sidebar {
  border-right: 1px solid rgba(103, 112, 123, 0.36) !important;
  background:
    linear-gradient(180deg, rgba(35, 38, 46, 0.98), rgba(5, 6, 10, 0.99)) !important;
  box-shadow: 12px 0 36px rgba(0, 0, 0, 0.28);
}
.app-sidebar .block,
.app-sidebar .form,
.app-sidebar .panel,
.app-sidebar .wrap-inner {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.app-sidebar-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 62px;
  padding: 8px 4px 16px;
  margin: 0 0 10px;
  border-bottom: 1px solid rgba(235, 207, 135, 0.18);
}
.app-sidebar-brand img {
  width: 34px;
  height: 34px;
  object-fit: contain;
  filter: drop-shadow(0 0 12px rgba(255, 96, 70, 0.36));
}
.app-sidebar-brand strong {
  display: block;
  color: #efe5bb;
  font-size: 17px;
  line-height: 20px;
  letter-spacing: 0;
}
.app-sidebar-brand span {
  display: block;
  color: #8b929a;
  font-size: 12px;
  line-height: 15px;
}
.app-sidebar-nav-section {
  padding-top: 3px;
}
.app-sidebar .app-nav {
  margin-top: 0;
}
.app-sidebar .app-nav .wrap,
.app-sidebar .app-nav .options,
.app-sidebar .app-nav .radio-group {
  gap: 2px !important;
}
.app-sidebar .app-nav input[type="radio"] {
  display: none !important;
}
.app-sidebar .app-nav label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  min-height: 38px !important;
  width: 100% !important;
  padding: 7px 4px !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  color: #dcdedf !important;
  font-size: 13px !important;
  line-height: 16px !important;
  cursor: pointer !important;
}
.app-sidebar .app-nav label::before {
  content: "";
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: contain;
  opacity: 0.72;
}
.app-sidebar .app-nav label:has(input:checked) {
  background: transparent !important;
  color: #ff6046 !important;
}
.app-sidebar .app-nav label:has(input:checked)::before {
  opacity: 1;
  filter: drop-shadow(0 0 8px rgba(255, 96, 70, 0.45));
}
__NAV_ICON_CSS__
.app-main {
  max-width: none !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 0 0 0 !important;
  box-sizing: border-box;
}
.app-view {
  gap: 10px;
  width: 100% !important;
  padding: 0 !important;
  box-sizing: border-box;
}
.app-view > .form,
.app-view > .block {
  width: 100%;
}
.dota-dropdown {
  position: relative;
  z-index: 20;
}
.dota-dropdown:focus-within {
  z-index: 2500;
}
[role="listbox"],
.options,
.dropdown-options,
.svelte-select-list {
  z-index: 4000 !important;
  pointer-events: auto !important;
  border: 1px solid rgba(103, 112, 123, 0.55) !important;
  background: linear-gradient(180deg, #36363e 0%, #23262e 100%) !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.58) !important;
}
[role="option"],
.option,
.dropdown-option,
.svelte-select-list div {
  color: #dcdedf !important;
}
[role="option"]:hover,
.option:hover,
.dropdown-option:hover,
.svelte-select-list div:hover {
  background: rgba(255, 96, 70, 0.16) !important;
  color: #fff !important;
}
.dota-choice-option {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
}
.dota-choice-option .dota-choice-icon {
  width: 28px;
  height: 28px;
  object-fit: contain;
  border-radius: 2px;
  flex: 0 0 28px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.22);
}
.dota-choice-option.dota-hero-option .dota-choice-icon {
  object-fit: cover;
}
.hero-strip, .item-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(72px, 82px));
  gap: 8px;
  margin-top: 6px;
  pointer-events: none;
}
.item-strip {
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
}
.entity-chip {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 42px;
  padding: 7px 8px;
  border: 1px solid rgba(103, 112, 123, 0.42);
  border-radius: 3px;
  background: linear-gradient(180deg, rgba(45, 45, 51, 0.78), rgba(22, 22, 24, 0.78));
}
.entity-chip img {
  width: 32px;
  height: 32px;
  object-fit: contain;
  border-radius: 5px;
}
.hero-card {
  min-height: 50px;
  padding: 4px;
  justify-content: center;
}
.hero-card img {
  width: 100%;
  height: 46px;
  object-fit: cover;
  border-radius: 2px;
}
.entity-chip strong {
  display: block;
  font-size: 13px;
  line-height: 16px;
}
.entity-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 3px;
}
.entity-tag {
  display: inline-flex;
  align-items: center;
  min-height: 18px;
  padding: 1px 6px;
  border-radius: 2px;
  background: rgba(171, 140, 59, 0.14);
  border: 1px solid rgba(235, 207, 135, 0.22);
  color: #d2bd6f;
  font-size: 11px;
  line-height: 14px;
}
.role-icon {
  width: 17px;
  height: 17px;
  margin-right: 4px;
  border-radius: 50%;
  background: rgba(245, 245, 235, 0.10);
  object-fit: contain;
  padding: 1px;
}
.entity-chip .role-icon {
  width: 17px;
  height: 17px;
  border-radius: 50%;
}
.empty-strip {
  color: #8b929a;
  font-size: 13px;
  padding: 7px 0;
}
""".replace("__NAV_ICON_CSS__", NAV_ICON_CSS)


def _dropdown_js(choice_icon_by_label: dict[str, dict[str, str]]) -> str:
    return f"""
() => {{
  const choiceIcons = {json.dumps(choice_icon_by_label)};
  const decorate = () => {{
    const optionSelector = [
      '[role="option"]',
      '.option',
      '.dropdown-option',
      'li',
      '.svelte-select-list div'
    ].join(', ');
    const options = document.querySelectorAll(optionSelector);
    options.forEach((option) => {{
      if (option.dataset && option.dataset.dotaChoiceIconDecorated === "1") return;
      const label = (option.textContent || "").trim().replace(/\\s+/g, " ");
      const icon = choiceIcons[label];
      if (!icon || !icon.src) return;
      option.classList.add("dota-choice-option");
      if (icon.kind) option.classList.add(`dota-${{icon.kind}}-option`);
      const img = document.createElement("img");
      img.className = "dota-choice-icon";
      img.src = icon.src;
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";
      option.prepend(img);
      if (option.dataset) option.dataset.dotaChoiceIconDecorated = "1";
    }});
  }};
  const observer = new MutationObserver(decorate);
  observer.observe(document.body, {{ childList: true, subtree: true }});
  document.addEventListener("click", () => setTimeout(decorate, 0), true);
  document.addEventListener("keyup", () => setTimeout(decorate, 0), true);
  decorate();
}}
"""


def _parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.replace(",", " ").split():
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _normalize_hero_ref(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _initial_aliases(value: str) -> set[str]:
    words = [word for word in re.split(r"[^A-Za-z0-9]+", value) if word]
    if len(words) < 2:
        return set()
    aliases = {"".join(word[0] for word in words)}
    non_stopwords = [
        word for word in words if word.lower() not in {"of", "the", "and", "s"}
    ]
    if len(non_stopwords) > 1:
        aliases.add("".join(word[0] for word in non_stopwords))
    return {alias.upper() for alias in aliases if 1 < len(alias) <= 5}


def _hero_aliases(hero_name: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    common_aliases = COMMON_HERO_ALIASES.get(hero_name, [])
    generated_aliases = (
        []
        if common_aliases
        else _initial_aliases(hero_name) - AMBIGUOUS_GENERATED_ALIASES
    )
    for alias in [*common_aliases, *generated_aliases]:
        if len(alias) < 2 or alias == hero_name:
            continue
        key = _normalize_hero_ref(alias)
        if key in seen:
            continue
        aliases.append(alias)
        seen.add(key)
    return sorted(aliases, key=lambda item: (len(item), item.lower()))


def _split_hero_refs(value: str) -> list[str]:
    refs: list[str] = []
    for chunk in re.split(r"[,;\n]+", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch(r"\d+(\s+\d+)*", chunk):
            refs.extend(chunk.split())
        else:
            refs.append(chunk)
    return refs


def _hero_lookup(heroes: pl.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    by_name: dict[str, int] = {}
    by_id: dict[int, str] = {}
    if heroes.is_empty():
        return by_name, by_id
    for row in heroes.iter_rows(named=True):
        hero_id = int(row["hero_id"])
        hero_name = str(row.get("hero_name") or f"Hero {hero_id}")
        by_id[hero_id] = hero_name
        by_name[_normalize_hero_ref(hero_name)] = hero_id
        for alias in _hero_aliases(hero_name):
            alias_key = _normalize_hero_ref(alias)
            if alias_key not in by_name:
                by_name[alias_key] = hero_id
    return by_name, by_id


def _parse_heroes(value: object, lookup: dict[str, int]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    unknown: list[str] = []
    seen: set[int] = set()
    if isinstance(value, list):
        refs = [str(item) for item in value if item not in {None, ""}]
    else:
        refs = _split_hero_refs(str(value or ""))
    for ref in refs:
        hero_id: int | None = None
        try:
            hero_id = int(ref)
        except ValueError:
            hero_id = lookup.get(_normalize_hero_ref(ref))
        if hero_id is None:
            unknown.append(ref)
            continue
        if hero_id not in seen:
            ids.append(hero_id)
            seen.add(hero_id)
    return ids, unknown


def _format_hero_ids(hero_ids: list[int], names: dict[int, str]) -> str:
    return ", ".join(f"{names.get(hero_id, f'Hero {hero_id}')} ({hero_id})" for hero_id in hero_ids)


def _asset_url(path: object) -> str:
    value = str(path or "")
    if not value:
        return ""
    if value.startswith("http"):
        return value
    if value.startswith("/"):
        return f"{ASSET_BASE_URL}{value}"
    return value


def _item_icon_url(item_key: object) -> str:
    return f"{ASSET_BASE_URL}/apps/dota2/images/dota_react/items/{item_key}.png"


def _html_escape(value: object) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _role_tags_html(roles: object) -> str:
    tags = [part.strip() for part in str(roles or "").split(",") if part.strip()]
    if not tags:
        return ""
    return (
        "<div class='entity-tags'>"
        + "".join(
            f"<span class='entity-tag'>{_role_icon_html(tag)}{_html_escape(tag)}</span>"
            for tag in tags
        )
        + "</div>"
    )


def _format_roles_text(roles: object) -> str:
    return ", ".join(part.strip() for part in str(roles or "").split(",") if part.strip())


def _format_item_time(seconds: object) -> str:
    value = float(seconds or 0)
    if value < 0:
        return "pre-game"
    return f"{round(value / 60, 1)} min"


def _hero_metadata(heroes: pl.DataFrame) -> dict[int, dict[str, str]]:
    metadata = {}
    if heroes.is_empty():
        return metadata
    for row in heroes.iter_rows(named=True):
        hero_id = int(row["hero_id"])
        hero_name = str(row.get("hero_name") or f"Hero {hero_id}")
        metadata[hero_id] = {
            "name": hero_name,
            "roles": _format_roles_text(row.get("roles")),
            "icon": _asset_url(row.get("icon") or row.get("img")),
            "aliases": ", ".join(_hero_aliases(hero_name)),
        }
    return metadata


def _hero_choices(heroes: pl.DataFrame) -> list[tuple[str, int]]:
    metadata = _hero_metadata(heroes)
    choices = []
    for hero_id, row in sorted(metadata.items(), key=lambda item: item[1]["name"]):
        aliases = f" ({row['aliases']})" if row["aliases"] else ""
        role_text = f" · {row['roles']}" if row["roles"] else ""
        choices.append((f"{row['name']}{aliases}{role_text}", hero_id))
    return choices


def _hero_icon_by_label(
    hero_choices: list[tuple[str, int]], metadata: dict[int, dict[str, str]]
) -> dict[str, dict[str, str]]:
    return {
        label: {"src": metadata.get(int(hero_id), {}).get("icon", ""), "kind": "hero"}
        for label, hero_id in hero_choices
        if metadata.get(int(hero_id), {}).get("icon")
    }


def _role_icon_html(role: str) -> str:
    icon = ROLE_ICON_URLS.get(role)
    if not icon:
        return ""
    return f"<img class='role-icon' src='{_html_escape(icon)}' alt='' loading='lazy'>"


def _dropdown_icon_by_label(
    hero_choices: list[tuple[str, int]], metadata: dict[int, dict[str, str]]
) -> dict[str, dict[str, str]]:
    icons = _hero_icon_by_label(hero_choices, metadata)
    icons.update(
        {
            label: {"src": ROLE_DROPDOWN_ICONS[value], "kind": "role"}
            for label, value in ROLE_OPTIONS
        }
    )
    icons.update(
        {
            label: {"src": SCOPE_ICON_URLS[value], "kind": "scope"}
            for label, value in SCOPE_OPTIONS
        }
    )
    return icons


def _sidebar_brand_html() -> str:
    logo = _html_escape(DOTA_LOGO_URL)
    return (
        "<div class='app-sidebar-brand'>"
        f"<img src='{logo}' alt='Dota 2' loading='eager'>"
        "<div class='app-title'><strong>DOTA2Tuned</strong>"
        "<span>Draft, meta, counters, builds, and match prediction</span></div>"
        "</div>"
    )


def _item_choices(items: pl.DataFrame) -> list[tuple[str, str]]:
    if items.is_empty():
        return []
    choices = []
    for row in items.iter_rows(named=True):
        key = str(row.get("item_key") or "")
        name = str(row.get("item_name") or key.replace("_", " ").title())
        if not key or key.startswith("recipe_"):
            continue
        cost = row.get("cost")
        cost_text = f" - {cost}g" if cost else ""
        choices.append((f"{name}{cost_text}", key))
    return sorted(choices, key=lambda item: item[0])


def _selected_hero_html(hero_ids: object, metadata: dict[int, dict[str, str]]) -> str:
    if not isinstance(hero_ids, list) or not hero_ids:
        return "<div class='hero-strip'></div>"
    chips = []
    for raw_id in hero_ids:
        hero_id = int(raw_id)
        row = metadata.get(hero_id, {})
        name = _html_escape(row.get("name") or f"Hero {hero_id}")
        icon = _html_escape(row.get("icon") or "")
        if icon:
            chips.append(
                f"<div class='entity-chip hero-card' title='{name}'>"
                f"<img src='{icon}' alt='{name}' loading='lazy'></div>"
            )
    return "<div class='hero-strip'>" + "".join(chips) + "</div>"


def _selected_item_html(item_key: object, items: pl.DataFrame) -> str:
    if not item_key:
        return "<div class='empty-strip'>No item selected.</div>"
    item_name = str(item_key).replace("_", " ").title()
    cost_text = ""
    if not items.is_empty():
        rows = items.filter(pl.col("item_key") == item_key)
        if not rows.is_empty():
            row = rows.row(0, named=True)
            item_name = str(row.get("item_name") or item_name)
            cost = row.get("cost")
            cost_text = f"{cost} gold" if cost else ""
    icon = _html_escape(_item_icon_url(item_key))
    return (
        "<div class='item-strip'><div class='entity-chip'>"
        f"<img src='{icon}' alt='{_html_escape(item_name)}' loading='lazy'>"
        f"<div><strong>{_html_escape(item_name)}</strong><span>{_html_escape(cost_text)}</span></div>"
        "</div></div>"
    )


def _format_recs(recs: list) -> str:
    if not recs:
        return (
            "No local recommendation data is available yet. Run ingestion and normalization first."
        )
    lines = []
    for idx, rec in enumerate(recs, start=1):
        lines.append(
            f"{idx}. **{rec.hero_name}** | score `{rec.score:.3f}` | "
            f"delta `{rec.win_prob_delta:+.3f}` | sample `{rec.sample_size}` | "
            f"confidence `{rec.confidence}`"
        )
        if rec.caveats:
            lines.append(f"   Caveat: {', '.join(rec.caveats)}")
    return "\n".join(lines)


def _call_tuned_model(settings, question: str, context: str, max_new_tokens: int = 384) -> str:
    if not question.strip():
        return "Enter a question."
    if not settings.modal_enabled:
        return "Tuned model is unavailable because `MODAL_ENABLED` is not set."
    if not settings.modal_token_id or not settings.modal_token_secret:
        return "Tuned model is unavailable because Modal credentials are not configured."
    try:
        import modal
    except ImportError:
        return "Tuned model is unavailable because the Modal client is not installed."

    os.environ["MODAL_TOKEN_ID"] = settings.modal_token_id
    os.environ["MODAL_TOKEN_SECRET"] = settings.modal_token_secret
    try:
        generate_fn = modal.Function.from_name(settings.modal_app_name, "generate_answer")
        result = generate_fn.remote(question, context, max_new_tokens)
    except Exception as exc:
        return f"Tuned model call failed: {str(exc)[:500]}"

    if result.get("status") != "ok":
        return json.dumps(result, indent=2)
    answer = result.get("answer") or ""
    model = result.get("model") or settings.hf_model_repo_id
    tokens = result.get("tokens")
    return f"{answer}\n\n`model: {model}` `tokens: {tokens}`"


def build_app() -> gr.Blocks:
    settings = get_settings()
    recommender = DraftRecommender(settings.parquet_dir)
    retriever = Retriever(settings.rag_dir)
    hero_name_lookup, hero_names = _hero_lookup(recommender.heroes)
    hero_metadata = _hero_metadata(recommender.heroes)
    hero_choices = _hero_choices(recommender.heroes)
    item_table = read_parquet(settings.parquet_dir / "dim_item.parquet")
    item_choices = _item_choices(item_table)

    def hero_preview(hero_ids: list[int] | None) -> str:
        return _selected_hero_html(hero_ids or [], hero_metadata)

    def hero_single_preview(hero_id: int | None) -> str:
        return hero_preview([hero_id] if hero_id else [])

    def item_preview(item_key: str | None) -> str:
        return _selected_item_html(item_key, item_table)

    def draft_coach(
        allies: list[int] | None,
        enemies: list[int] | None,
        bans: list[int] | None,
        role: str,
        scope: str,
    ) -> tuple[str, str]:
        allied_ids, allied_unknown = _parse_heroes(allies, hero_name_lookup)
        enemy_ids, enemy_unknown = _parse_heroes(enemies, hero_name_lookup)
        banned_ids, banned_unknown = _parse_heroes(bans, hero_name_lookup)
        draft = DraftInput(
            allied_heroes=allied_ids,
            enemy_heroes=enemy_ids,
            banned_heroes=banned_ids,
            role=role or None,
            scope=scope,
            patch="current",
        )
        recs = recommender.recommend(draft, limit=8)
        query_terms = [
            role or "",
            " ".join(hero_names.get(hero_id, str(hero_id)) for hero_id in allied_ids),
            " ".join(hero_names.get(hero_id, str(hero_id)) for hero_id in enemy_ids),
            " ".join(hero_names.get(hero_id, str(hero_id)) for hero_id in banned_ids),
        ]
        evidence = retriever.search(
            " ".join(query_terms), patch="current", limit=5
        )
        unknown = allied_unknown + enemy_unknown + banned_unknown
        warning = f"Unrecognized heroes ignored: {', '.join(unknown)}\n\n" if unknown else ""
        return warning + _format_recs(recs), json.dumps(evidence, indent=2)

    def hero_meta(query: str, hero_id: int | None, item_key: str | None) -> str:
        parts = [query or "current meta"]
        if hero_id:
            parts.append(hero_names.get(int(hero_id), str(hero_id)))
        if item_key:
            parts.append(str(item_key).replace("_", " "))
        docs = retriever.search(" ".join(parts), patch="current", limit=8)
        if not docs:
            return "No retrieval index is available yet. Run `dota2tuned build-rag` first."
        return "\n\n".join(f"**{doc['source']}** `{doc['score']}`\n{doc['text']}" for doc in docs)

    def tuned_model(question: str, context: str, max_new_tokens: int) -> tuple[str, str]:
        docs = []
        evidence = context.strip()
        if not evidence:
            docs = retriever.search(question or "current meta", patch="current", limit=5)
            evidence = "\n\n".join(
                f"{doc['source']} score={doc['score']}\n{doc['text']}" for doc in docs
            )
        answer = _call_tuned_model(settings, question, evidence, max_new_tokens)
        return answer, json.dumps(docs, indent=2)

    def data_status() -> str:
        files = []
        for path in sorted(settings.parquet_dir.glob("*.parquet")):
            files.append(f"- `{path.name}` ({path.stat().st_size:,} bytes)")
        if not files:
            return "No normalized Parquet files found."
        return "\n".join(files)

    build_stats = read_parquet(settings.parquet_dir / "fact_hero_build_stats.parquet")

    def match_predictor(radiant: list[int] | None, dire: list[int] | None) -> str:
        radiant_ids, radiant_unknown = _parse_heroes(radiant, hero_name_lookup)
        dire_ids, dire_unknown = _parse_heroes(dire, hero_name_lookup)
        if radiant_unknown or dire_unknown:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Unrecognized heroes.",
                    "unknown": radiant_unknown + dire_unknown,
                },
                indent=2,
            )
        prediction = predict_draft_win(settings.model_dir, radiant_ids, dire_ids)
        if prediction.get("status") != "ok":
            return prediction.get("message", "Prediction unavailable.")
        return json.dumps(prediction, indent=2)

    def hero_builds(hero_id: int | None, item_key: str | None) -> str:
        hero_ids, unknown = _parse_heroes([hero_id] if hero_id else [], hero_name_lookup)
        if unknown:
            return f"Unrecognized hero: {', '.join(unknown)}"
        if not hero_ids:
            return "Select a hero."
        if build_stats.is_empty():
            return "No build table is available yet. Run match enrichment and normalization first."
        filtered = build_stats.filter(pl.col("hero_id") == hero_ids[0])
        if item_key:
            filtered = filtered.filter(pl.col("item_key") == item_key)
        rows = filtered.sort("purchases", descending=True).head(15).iter_rows(named=True)
        lines = []
        for row in rows:
            median_time = _format_item_time(row.get("median_time"))
            lines.append(
                f"- **{row.get('item_key')}** in `{row.get('time_bucket')}`: "
                f"{row.get('purchases')} purchases, median `{median_time}`"
            )
        return "\n".join(lines) if lines else "No observed item timings for that hero."

    def draft_lab(enemies: list[int] | None, role: str, twist: str) -> str:
        enemy_ids, unknown = _parse_heroes(enemies, hero_name_lookup)
        if unknown:
            return f"Unrecognized heroes ignored: {', '.join(unknown)}"
        draft = DraftInput(enemy_heroes=enemy_ids, role=role or "mid", scope="pro", patch="current")
        recs = recommender.recommend(draft, limit=5)
        if not recs:
            return "Draft Lab needs local hero stats. Run ingestion and normalization first."
        top = recs[0]
        enemy_text = (
            _format_hero_ids(enemy_ids, hero_names) if enemy_ids else "an unknown enemy draft"
        )
        caveat = ", ".join(top.caveats) if top.caveats else "patch and draft context still matter"
        if twist == "One-minute coach":
            return (
                f"**Challenge:** Sell **{top.hero_name}** for `{role}` against {enemy_text} "
                "in one minute.\n\n"
                f"- Open with the score: `{top.score:.3f}` and confidence `{top.confidence}`.\n"
                f"- Mention one caveat: {caveat}.\n"
                "- End by asking for lane matchup, bans, and teamfight plan."
            )
        if twist == "Chaos constraint":
            return (
                f"**Constraint:** Draft **{top.hero_name}**, but explain it without saying "
                "`meta`, `broken`, or `free win`.\n\n"
                f"- Evidence hook: sample `{top.sample_size}`, delta `{top.win_prob_delta:+.3f}`.\n"
                "- Required caveat: weak samples should change confidence, "
                "not create fake certainty."
            )
        return (
            f"**Tiny scout card:** {top.hero_name} into {enemy_text}\n\n"
            f"- Pick score: `{top.score:.3f}`\n"
            f"- Counter lift: `{top.counter_lift:+.3f}`\n"
            f"- Synergy lift: `{top.synergy_lift:+.3f}`\n"
            f"- Confidence: `{top.confidence}` from `{top.sample_size}` pro samples\n"
            "- Rule: if the evidence is thin, say so before giving the pick."
        )

    def switch_view(selected: str):
        return tuple(gr.update(visible=label == selected) for label in NAV_OPTIONS)

    with gr.Blocks(title="DOTA2Tuned") as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        with gr.Sidebar(open=True, width=238, position="left", elem_classes=["app-sidebar"]):
            gr.HTML(_sidebar_brand_html())
            with gr.Column(elem_classes=["app-sidebar-nav-section"]):
                nav = gr.Radio(
                    choices=NAV_OPTIONS,
                    value="Draft Coach",
                    show_label=False,
                    container=False,
                    elem_classes=["app-nav"],
                )
        with gr.Column(elem_classes=["app-main"]):
            with gr.Column(visible=True, elem_classes=["app-view"]) as draft_view:
                with gr.Row():
                    allies = gr.Dropdown(
                        choices=hero_choices,
                        label="Allied heroes",
                        multiselect=True,
                        filterable=True,
                        max_choices=5,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    enemies = gr.Dropdown(
                        choices=hero_choices,
                        label="Enemy heroes",
                        value=[44, 30],
                        multiselect=True,
                        filterable=True,
                        max_choices=5,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    bans = gr.Dropdown(
                        choices=hero_choices,
                        label="Banned heroes",
                        multiselect=True,
                        filterable=True,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                with gr.Row():
                    ally_preview = gr.HTML(hero_preview([]))
                    enemy_preview = gr.HTML(hero_preview([44, 30]))
                    ban_preview = gr.HTML(hero_preview([]))
                with gr.Row():
                    role = gr.Dropdown(
                        choices=ROLE_OPTIONS,
                        label="Role",
                        value="mid",
                        elem_classes=["dota-dropdown", "role-dropdown"],
                    )
                    scope = gr.Dropdown(
                        choices=SCOPE_OPTIONS,
                        label="Scope",
                        value="pro",
                        elem_classes=["dota-dropdown", "scope-dropdown"],
                    )
                run = gr.Button("Recommend")
                rec_output = gr.Markdown()
                evidence_output = gr.Code(label="Evidence", language="json")
                allies.change(hero_preview, inputs=[allies], outputs=[ally_preview])
                enemies.change(hero_preview, inputs=[enemies], outputs=[enemy_preview])
                bans.change(hero_preview, inputs=[bans], outputs=[ban_preview])
                run.click(
                    draft_coach,
                    inputs=[allies, enemies, bans, role, scope],
                    outputs=[rec_output, evidence_output],
                )

            with gr.Column(visible=False, elem_classes=["app-view"]) as hero_meta_view:
                with gr.Row():
                    meta_hero = gr.Dropdown(
                        choices=hero_choices,
                        label="Hero",
                        filterable=True,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    meta_item = gr.Dropdown(
                        choices=item_choices,
                        label="Item",
                        filterable=True,
                        elem_classes=["dota-dropdown", "item-dropdown"],
                    )
                with gr.Row():
                    meta_hero_preview = gr.HTML(hero_preview([]))
                    meta_item_preview = gr.HTML(item_preview(None))
                query = gr.Textbox(label="Patch or meta query", value="current pro meta")
                meta_button = gr.Button("Search")
                meta_output = gr.Markdown()
                meta_hero.change(
                    hero_single_preview,
                    inputs=[meta_hero],
                    outputs=[meta_hero_preview],
                )
                meta_item.change(item_preview, inputs=[meta_item], outputs=[meta_item_preview])
                meta_button.click(
                    hero_meta,
                    inputs=[query, meta_hero, meta_item],
                    outputs=[meta_output],
                )

            with gr.Column(visible=False, elem_classes=["app-view"]) as tuned_model_view:
                tuned_question = gr.Textbox(
                    label="Question",
                    value=(
                        "Suggest one mid hero against Phantom Assassin and Witch Doctor, "
                        "and include one caveat."
                    ),
                )
                tuned_context = gr.Textbox(
                    label="Optional evidence",
                    lines=5,
                    placeholder="Leave blank to retrieve local patch/stat evidence automatically.",
                )
                tuned_tokens = gr.Slider(
                    minimum=64,
                    maximum=768,
                    value=256,
                    step=32,
                    label="Max response tokens",
                )
                tuned_button = gr.Button("Ask Tuned Model")
                tuned_output = gr.Markdown()
                tuned_evidence = gr.Code(label="Retrieved Evidence", language="json")
                tuned_button.click(
                    tuned_model,
                    inputs=[tuned_question, tuned_context, tuned_tokens],
                    outputs=[tuned_output, tuned_evidence],
                )

            with gr.Column(visible=False, elem_classes=["app-view"]) as match_predictor_view:
                with gr.Row():
                    radiant = gr.Dropdown(
                        choices=hero_choices,
                        label="Radiant heroes",
                        value=[1, 2, 3, 25, 5],
                        multiselect=True,
                        filterable=True,
                        max_choices=5,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    dire = gr.Dropdown(
                        choices=hero_choices,
                        label="Dire heroes",
                        value=[14, 74, 6, 26, 18],
                        multiselect=True,
                        filterable=True,
                        max_choices=5,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                with gr.Row():
                    radiant_preview = gr.HTML(hero_preview([1, 2, 3, 25, 5]))
                    dire_preview = gr.HTML(hero_preview([14, 74, 6, 26, 18]))
                predict_button = gr.Button("Predict")
                predict_output = gr.Code(label="Prediction", language="json")
                radiant.change(hero_preview, inputs=[radiant], outputs=[radiant_preview])
                dire.change(hero_preview, inputs=[dire], outputs=[dire_preview])
                predict_button.click(
                    match_predictor,
                    inputs=[radiant, dire],
                    outputs=[predict_output],
                )

            with gr.Column(visible=False, elem_classes=["app-view"]) as builds_view:
                with gr.Row():
                    hero = gr.Dropdown(
                        choices=hero_choices,
                        label="Hero",
                        value=1,
                        filterable=True,
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    build_item = gr.Dropdown(
                        choices=item_choices,
                        label="Optional item filter",
                        filterable=True,
                        elem_classes=["dota-dropdown", "item-dropdown"],
                    )
                with gr.Row():
                    build_hero_preview = gr.HTML(hero_preview([1]))
                    build_item_preview = gr.HTML(item_preview(None))
                builds_button = gr.Button("Show Builds")
                builds_output = gr.Markdown()
                hero.change(
                    hero_single_preview,
                    inputs=[hero],
                    outputs=[build_hero_preview],
                )
                build_item.change(item_preview, inputs=[build_item], outputs=[build_item_preview])
                builds_button.click(hero_builds, inputs=[hero, build_item], outputs=[builds_output])

            with gr.Column(visible=False, elem_classes=["app-view"]) as draft_lab_view:
                lab_enemies = gr.Dropdown(
                    choices=hero_choices,
                    label="Enemy heroes",
                    value=[44, 30],
                    multiselect=True,
                    filterable=True,
                    max_choices=5,
                    elem_classes=["dota-dropdown", "hero-dropdown"],
                )
                lab_enemy_preview = gr.HTML(hero_preview([44, 30]))
                lab_role = gr.Dropdown(
                    choices=ROLE_OPTIONS,
                    label="Role",
                    value="mid",
                    elem_classes=["dota-dropdown", "role-dropdown"],
                )
                lab_twist = gr.Dropdown(
                    ["Tiny scout card", "One-minute coach", "Chaos constraint"],
                    label="Mode",
                    value="Tiny scout card",
                    elem_classes=["dota-dropdown"],
                )
                lab_button = gr.Button("Generate Draft Lab Card")
                lab_output = gr.Markdown()
                lab_enemies.change(hero_preview, inputs=[lab_enemies], outputs=[lab_enemy_preview])
                lab_button.click(
                    draft_lab,
                    inputs=[lab_enemies, lab_role, lab_twist],
                    outputs=[lab_output],
                )

            with gr.Column(visible=False, elem_classes=["app-view"]) as data_freshness_view:
                status_button = gr.Button("Refresh")
                status_output = gr.Markdown()
                status_button.click(data_status, outputs=[status_output])
        nav.change(
            switch_view,
            inputs=[nav],
            outputs=[
                draft_view,
                hero_meta_view,
                tuned_model_view,
                match_predictor_view,
                builds_view,
                draft_lab_view,
                data_freshness_view,
            ],
            api_visibility="private",
        )

    demo.dota2tuned_js = _dropdown_js(_dropdown_icon_by_label(hero_choices, hero_metadata))
    return demo
