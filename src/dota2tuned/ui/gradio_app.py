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

PRIMARY_ATTR_INFO = {
    "str": ("STR", "#ee8f7d"),
    "agi": ("AGI", "#8fd19e"),
    "int": ("INT", "#8fc8ff"),
    "all": ("UNIVERSAL", "#c09cff"),
}

TIME_BUCKET_ORDER = ["0-10m", "10-20m", "20-30m", "30-40m", "40m+", "unknown"]

TIME_BUCKET_LABELS = {
    "0-10m": "Laning (0-10 min)",
    "10-20m": "Early Game (10-20 min)",
    "20-30m": "Mid Game (20-30 min)",
    "30-40m": "Late Game (30-40 min)",
    "40m+": "End Game (40+ min)",
    "unknown": "Unknown Timing",
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
  font-family: "Trajan Pro";
  src:
    url("https://cdn.steamstatic.com/apps/dota2/fonts/goudytrajan-regular-pro-webfont.woff")
    format("woff");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "Trajan Pro";
  src:
    url("https://cdn.steamstatic.com/apps/dota2/fonts/goudytrajan-medium-pro-webfont.woff")
    format("woff");
  font-weight: 700;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: "Trajan Pro";
  src:
    url("https://cdn.steamstatic.com/apps/dota2/fonts/goudytrajan-bold-pro-webfont.woff")
    format("woff");
  font-weight: 900;
  font-style: normal;
  font-display: swap;
}
/* Dark palette: single source of truth (app is dark-only, WCAG 2.2 AA). */
:root {
  --d2-bg: #0d0f13;
  --d2-fg: #dcdedf;
  --d2-fg-strong: #f1f3f4;
  --d2-fg-muted: #9ba2aa;
  --d2-gold: #d9b166;
  --d2-gold-soft: #efe5bb;
  --d2-red: #ff6046;
  --d2-red-strong: #c2452f;
  --d2-focus: #ff7a63;
  --d2-border: #7d8893;
  --d2-border-menu: #8a96a3;
  --d2-scroll-thumb: #8a96a3;
}
/* Single dark palette, applied unconditionally (no ?__theme=dark redirect).
   Map Gradio's semantic theme variables to their dark values so every built-in
   component renders dark regardless of system preference or the .dark body
   class. Set on .gradio-container so the subtree inherits these over body.dark
   and :root. Branded surfaces are further styled by the !important rules below. */
.gradio-container,
.gradio-container.dark {
  color-scheme: dark;
  --background-fill-primary: var(--neutral-950);
  --background-fill-secondary: var(--neutral-900);
  --body-background-fill: var(--neutral-950);
  --body-text-color: var(--neutral-100);
  --body-text-color-subdued: var(--neutral-400);
  --border-color-primary: var(--neutral-700);
  --border-color-accent: var(--neutral-600);
  --block-background-fill: var(--neutral-800);
  --block-border-color: var(--neutral-700);
  --block-label-background-fill: var(--primary-600);
  --block-label-text-color: #fff;
  --block-title-text-color: #fff;
  --panel-background-fill: var(--neutral-900);
  --panel-border-color: var(--neutral-700);
  --input-background-fill: var(--neutral-700);
  --input-border-color: var(--neutral-700);
  --input-placeholder-color: var(--neutral-500);
  --code-background-fill: var(--neutral-800);
  --table-even-background-fill: var(--neutral-950);
  --table-odd-background-fill: var(--neutral-900);
  --table-border-color: var(--neutral-700);
  --color-accent-soft: var(--neutral-700);
}
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  color: #dcdedf;
  font-family: "Trajan Pro", "Goudy Trajan", "Noto Sans", serif !important;
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
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container label,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
  font-family: "Trajan Pro", "Goudy Trajan", "Noto Sans", serif !important;
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
  border-color: var(--d2-border) !important;
  background: rgba(22, 22, 24, 0.70) !important;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.035) !important;
}
.gradio-container textarea,
.gradio-container input {
  color: var(--d2-gold-soft) !important;
}
.gradio-container :focus-visible {
  outline: 2px solid var(--d2-focus) !important;
  outline-offset: 2px !important;
  border-radius: 2px;
}
.app-main button {
  border: 1px solid rgba(255, 96, 70, 0.42) !important;
  border-radius: 3px !important;
  background:
    linear-gradient(180deg, rgba(149, 46, 70, 0.92), rgba(83, 31, 28, 0.94)) !important;
  color: #fff !important;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.34) !important;
  transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease !important;
}
.app-main button:hover {
  border-color: rgba(255, 96, 70, 0.74) !important;
  background:
    linear-gradient(180deg, var(--d2-red-strong), rgba(126, 43, 34, 0.98)) !important;
  transform: translateY(-1px);
}
.app-main button:active {
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
  border-bottom: 1px solid rgba(235, 207, 135, 0.60);
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
  color: var(--d2-fg-muted);
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
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  margin: -1px !important;
  padding: 0 !important;
  border: 0 !important;
  clip: rect(0 0 0 0) !important;
  clip-path: inset(50%) !important;
  overflow: hidden !important;
  white-space: nowrap !important;
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
  text-transform: uppercase !important;
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
  background: rgba(255, 96, 70, 0.10) !important;
  color: var(--d2-red) !important;
  box-shadow: inset 3px 0 0 var(--d2-red) !important;
  font-weight: 900 !important;
}
.app-sidebar .app-nav label:has(input:checked)::before {
  opacity: 1;
  filter: drop-shadow(0 0 8px rgba(255, 96, 70, 0.45));
}
.app-sidebar .app-nav label:has(input:focus-visible) {
  outline: 2px solid var(--d2-focus) !important;
  outline-offset: -2px !important;
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
/* All views stay mounted (visible=True); JS toggles .d2-active to switch tabs.
   This keeps view visibility fully under our control instead of Gradio's
   visible= toggle, which left freshly-selected tabs blank until re-selected. */
.app-view:not(.d2-active) {
  display: none !important;
}
.dota-dropdown {
  position: relative;
  z-index: 20;
}
.dota-dropdown:focus-within {
  z-index: 2500;
}
.dota-dropdown .wrap,
.dota-dropdown .wrap-inner,
.dota-dropdown .input-container {
  position: relative !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
}
.dota-dropdown .secondary-wrap,
.dota-dropdown .secondary-wrapper {
  display: flex !important;
  align-items: center !important;
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: 100% !important;
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
.dota-dropdown input[autocomplete="off"],
.dota-dropdown input[role="combobox"] {
  flex: 1 1 auto !important;
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
  padding-right: 32px !important;
  background: transparent !important;
  box-shadow: none !important;
}
.dota-dropdown .icon-wrap {
  position: absolute !important;
  top: 50% !important;
  right: 8px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 24px !important;
  height: 24px !important;
  margin: 0 !important;
  transform: translateY(-50%) !important;
  background: transparent !important;
  z-index: 2 !important;
}
.dota-dropdown .icon-wrap svg {
  width: 16px !important;
  height: 16px !important;
}
.dota-dropdown ul[role="listbox"],
.dota-dropdown .options {
  z-index: 4000 !important;
  pointer-events: auto !important;
  min-width: 0 !important;
  max-width: calc(100vw - 16px) !important;
  overflow-x: hidden !important;
  overscroll-behavior: contain;
  box-sizing: border-box !important;
  border: 1px solid var(--d2-border-menu) !important;
  background: linear-gradient(180deg, #36363e 0%, #23262e 100%) !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.58) !important;
  scrollbar-width: thin;
  scrollbar-color: var(--d2-scroll-thumb) transparent;
}
.dota-dropdown ul[role="listbox"]::-webkit-scrollbar,
.dota-dropdown .options::-webkit-scrollbar {
  width: 12px;
  height: 12px;
}
.dota-dropdown ul[role="listbox"]::-webkit-scrollbar-track,
.dota-dropdown .options::-webkit-scrollbar-track {
  background: transparent;
}
.dota-dropdown ul[role="listbox"]::-webkit-scrollbar-thumb,
.dota-dropdown .options::-webkit-scrollbar-thumb {
  background: var(--d2-scroll-thumb);
  border: 3px solid transparent;
  background-clip: padding-box;
  border-radius: 8px;
}
.dota-dropdown [role="option"] {
  color: var(--d2-fg) !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
.dota-dropdown [role="option"]:hover {
  background: rgba(255, 96, 70, 0.16) !important;
  color: #fff !important;
}
/* Rich option rendering is additive: we inject an icon + label as new child
   nodes and collapse Gradio's own text node/checkmark with font-size:0 instead
   of deleting them, so Svelte's reconciliation never touches removed nodes. */
.dota-dropdown li[role="option"].dota-decorated-rich {
  font-size: 0 !important;
  line-height: 0 !important;
}
.dota-dropdown li[role="option"].dota-decorated-rich > .dota-choice-icon,
.dota-dropdown li[role="option"].dota-decorated-rich > .dota-choice-label {
  font-size: 13px;
  line-height: 1.25;
}
.dota-choice-option {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
.dota-choice-option .dota-choice-icon {
  width: 22px;
  height: 22px;
  object-fit: contain;
  border-radius: 2px;
  flex: 0 0 22px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.22);
}
.dota-choice-option .dota-choice-icon,
.dota-choice-label,
.dota-choice-label * {
  pointer-events: none !important;
}
.dota-choice-option.dota-hero-option {
  align-items: flex-start !important;
}
.dota-choice-option.dota-hero-option .dota-choice-icon {
  object-fit: cover;
}
.dota-choice-label {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1 1 auto;
  min-width: 0;
  max-width: calc(100% - 30px);
}
.dota-choice-name {
  color: #f1f3f4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dota-choice-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  min-width: 0;
}
.dota-choice-tag {
  display: inline-flex;
  align-items: center;
  min-height: 15px;
  padding: 0 5px;
  border: 1px solid rgba(235, 207, 135, 0.34);
  border-radius: 2px;
  background: rgba(235, 207, 135, 0.16);
  color: #efe5bb;
  font-size: 10px;
  line-height: 13px;
  max-width: 100%;
}
.hero-dropdown .token,
.hero-dropdown .token-remove.remove-all {
  display: none !important;
}
.hero-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(178px, 1fr));
  gap: 8px;
  margin-top: 6px;
}
.item-strip {
  display: grid;
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
  position: relative;
  min-height: 62px;
  padding: 8px 34px 8px 8px;
  align-items: center;
  overflow: hidden;
}
.hero-card > img {
  width: 48px;
  height: 36px;
  flex: 0 0 48px;
  object-fit: contain;
  border-radius: 2px;
  background: rgba(5, 6, 10, 0.62);
}
.hero-card-body {
  min-width: 0;
  flex: 1 1 auto;
  padding-right: 2px;
}
.app-main .hero-card-remove {
  position: absolute !important;
  top: 6px;
  right: 6px;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 20px !important;
  height: 20px !important;
  min-width: 20px !important;
  padding: 0 !important;
  border: 1px solid rgba(235, 207, 135, 0.70) !important;
  border-radius: 50% !important;
  background: rgba(5, 6, 10, 0.72) !important;
  color: #efe5bb !important;
  box-shadow: none !important;
  font-size: 12px !important;
  line-height: 18px !important;
  cursor: pointer !important;
}
.app-main .hero-card-remove:hover {
  border-color: rgba(255, 96, 70, 0.82) !important;
  background: rgba(96, 32, 28, 0.92) !important;
  color: #fff !important;
  transform: none;
}
.entity-chip strong {
  display: block;
  font-size: 13px;
  line-height: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.hero-card strong {
  overflow: visible;
  text-overflow: clip;
  white-space: normal;
  overflow-wrap: normal;
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
  background: rgba(235, 207, 135, 0.18);
  border: 1px solid rgba(235, 207, 135, 0.38);
  color: #efe5bb;
  font-size: 11px;
  line-height: 14px;
}
.entity-tag .role-icon {
  width: 12px !important;
  height: 12px !important;
  flex: 0 0 12px;
  margin-right: 3px;
  padding: 0;
  border-radius: 50%;
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
  color: var(--d2-fg-muted);
  font-size: 13px;
  padding: 7px 0;
}
.build-hero-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 18px;
  border: 1px solid rgba(235, 207, 135, 0.4);
  border-radius: 4px;
  background-color: rgba(22, 22, 24, 0.78);
  background-size: cover;
  background-position: center 18%;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
  overflow: hidden;
}
.build-hero-header::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    rgba(10, 11, 14, 0.92) 0%,
    rgba(10, 11, 14, 0.55) 55%,
    rgba(10, 11, 14, 0.85) 100%
  );
  pointer-events: none;
}
.build-hero-header .build-hero-icon {
  position: relative;
  z-index: 1;
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: 6px;
  flex: 0 0 64px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.4), 0 6px 18px rgba(0, 0, 0, 0.4);
}
.build-hero-title {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.build-hero-title strong {
  font-size: 20px;
  line-height: 24px;
  color: #efe5bb;
}
.build-stat-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.build-stat-pill {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 12px;
  border: 1px solid var(--pill-color, rgba(235, 207, 135, 0.4));
  background: rgba(235, 207, 135, 0.10);
  color: #efe5bb;
  font-size: 11px;
  line-height: 16px;
}
.build-columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 4px;
}
.build-column {
  border: 1px solid rgba(217, 177, 102, 0.3);
  border-radius: 4px;
  padding: 12px;
  background: linear-gradient(180deg, rgba(50, 51, 58, 0.78), rgba(20, 21, 25, 0.78));
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03), 0 4px 14px rgba(0, 0, 0, 0.28);
}
.build-column-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  color: #efd9a0;
  padding-bottom: 7px;
  margin-bottom: 9px;
  border-bottom: 1px solid rgba(235, 207, 135, 0.32);
}
.build-item-list {
  display: flex;
  flex-direction: column;
  gap: 9px;
}
.build-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px;
  border-radius: 3px;
  cursor: help;
  transition: background 0.12s ease;
}
.build-item:hover {
  background: rgba(255, 96, 70, 0.10);
}
.build-item img {
  width: 38px;
  height: 38px;
  object-fit: cover;
  border-radius: 3px;
  flex: 0 0 38px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.32);
}
.build-item-order {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
  border-radius: 50%;
  border: 1px solid rgba(235, 207, 135, 0.5);
  background: rgba(5, 6, 10, 0.6);
  color: #efe5bb;
  font-size: 10px;
  line-height: 16px;
  font-weight: 700;
}
.build-item-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1 1 auto;
}
.build-item-meta strong {
  font-size: 13px;
  line-height: 16px;
  color: #f6f3e9;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.build-item-meta span {
  font-size: 11px;
  line-height: 14px;
  color: #a3aab2;
}
.build-item-bar {
  height: 4px;
  border-radius: 2px;
  background: rgba(103, 112, 123, 0.32);
  overflow: hidden;
}
.build-item-bar span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, rgba(255, 96, 70, 0.9), rgba(217, 177, 102, 0.9));
}
.build-column-highlight {
  border-color: rgba(235, 207, 135, 0.55);
  background: linear-gradient(180deg, rgba(94, 68, 32, 0.42), rgba(22, 22, 24, 0.7));
  margin-top: 4px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.12), 0 4px 14px rgba(0, 0, 0, 0.28);
}
.build-column-highlight .build-column-title {
  font-size: 13px;
  color: #ffd98c;
}
.build-item-list-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}
.build-item-highlight img {
  width: 48px;
  height: 48px;
  flex: 0 0 48px;
  box-shadow: 0 0 0 1px rgba(235, 207, 135, 0.5), 0 4px 10px rgba(0, 0, 0, 0.3);
}
@media (max-width: 720px) {
  .dota-dropdown .options,
  .dota-dropdown ul[role="listbox"] {
    left: 8px !important;
    right: 8px !important;
    width: calc(100vw - 16px) !important;
    max-width: calc(100vw - 16px) !important;
  }
}
@media (prefers-reduced-motion: reduce) {
  .gradio-container *,
  .gradio-container *::before,
  .gradio-container *::after {
    transition-duration: 0.001ms !important;
    animation-duration: 0.001ms !important;
    transition-delay: 0ms !important;
    scroll-behavior: auto !important;
  }
  .app-main button:hover {
    transform: none !important;
  }
}
""".replace("__NAV_ICON_CSS__", NAV_ICON_CSS)


# Injected into <head>. Preconnect to the asset CDNs. Dark mode is a single CSS
# palette (see the .gradio-container theme-variable overrides in APP_CSS), so no
# ?__theme=dark redirect is needed.
APP_HEAD = """
<link rel="preconnect" href="https://cdn.steamstatic.com" crossorigin>
<link rel="preconnect" href="https://cdn.cloudflare.steamstatic.com" crossorigin>
"""


def _dropdown_js(choice_icon_by_label: dict[str, dict[str, str]]) -> str:
    return f"""
() => {{
  const choiceIcons = {json.dumps(choice_icon_by_label)};
  const cleanLabel = (value) => (value || "").trim().replace(/\\s+/g, " ");
  const closestElement = (target, selector) => {{
    if (!(target instanceof Element)) return null;
    return target.closest(selector);
  }};
  const navViews = {{
    "Draft Coach": "draft-coach-view",
    "Hero Meta": "hero-meta-view",
    "Tuned Model": "tuned-model-view",
    "Match Predictor": "match-predictor-view",
    "Builds": "builds-view",
    "Draft Lab": "draft-lab-view",
    "Data Freshness": "data-freshness-view"
  }};
  const comparableHeroName = (value) =>
    cleanLabel(value)
      .split(" · ", 1)[0]
      .replace(/\\([^)]*\\)/g, " ")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "");
  const isDropdownMenuScroll = (event) => {{
    const target = event?.target;
    if (!(target instanceof Element)) return false;
    return Boolean(
      closestElement(target, '.options, [role="listbox"]')
    );
  }};
  const syncDropdownMenus = () => {{
    const margin = 8;
    if (!document.querySelector('.dota-dropdown .options, .dota-dropdown ul[role="listbox"]')) {{
      return;
    }}
    document.querySelectorAll(".dota-dropdown").forEach((dropdown) => {{
      const input = dropdown.querySelector(
        'input[autocomplete="off"], input[role="combobox"], input'
      );
      const listbox = dropdown.querySelector('.options, ul[role="listbox"]');
      if (!input || !listbox) return;
      const rect = input.closest(".wrap, .wrap-inner, .input-container")?.getBoundingClientRect()
        || input.getBoundingClientRect();
      const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
      const maxWidth = Math.max(240, viewportWidth - margin * 2);
      const width = Math.min(Math.max(rect.width, 240), maxWidth);
      const left = Math.min(
        Math.max(rect.left, margin),
        Math.max(margin, viewportWidth - width - margin)
      );
      listbox.style.width = `${{width}}px`;
      listbox.style.minWidth = "0px";
      listbox.style.maxWidth = `${{maxWidth}}px`;
      listbox.style.left = `${{left}}px`;
      listbox.style.boxSizing = "border-box";
      listbox.style.overflowX = "hidden";
    }});
  }};
  const syncSidebarView = () => {{
    // Views are always rendered; we own visibility via the .d2-active class
    // (CSS hides the rest). We never set inline display/hidden, which would
    // fight Svelte and leave a freshly-selected tab blank until re-selected.
    const nav = document.querySelector(".app-nav");
    if (!nav) return;
    const checked = nav.querySelector('input[type="radio"]:checked');
    const label = checked ? closestElement(checked, "label") : null;
    const selected = cleanLabel(label?.textContent || checked?.value || "Draft Coach");
    Object.entries(navViews).forEach(([name, id]) => {{
      const view = document.getElementById(id);
      if (!view) return;
      const active = name === selected;
      view.classList.toggle("d2-active", active);
      view.setAttribute("aria-hidden", active ? "false" : "true");
    }});
  }};
  const decorateOption = (option) => {{
    if (option.dataset && option.dataset.dotaChoiceIconDecorated === "1") return;
    // Gradio 6.18 prefixes option text with a "✓" checkmark glyph; the clean
    // choice label lives in aria-label, which is what choiceIcons is keyed on.
    const label = cleanLabel(option.getAttribute("aria-label") || option.textContent);
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
    if (icon.kind === "hero" && label.includes(" · ")) {{
      const [name, tagsText] = label.split(" · ", 2);
      const labelWrap = document.createElement("span");
      labelWrap.className = "dota-choice-label";
      const nameEl = document.createElement("span");
      nameEl.className = "dota-choice-name";
      nameEl.textContent = name;
      const tagsEl = document.createElement("span");
      tagsEl.className = "dota-choice-tags";
      tagsText.split(",").map((tag) => tag.trim()).filter(Boolean).forEach((tag) => {{
        const tagEl = document.createElement("span");
        tagEl.className = "dota-choice-tag";
        tagEl.textContent = tag;
        tagsEl.append(tagEl);
      }});
      labelWrap.append(nameEl, tagsEl);
      // Additive only: never delete Gradio/Svelte-owned nodes (textContent="").
      // CSS (.dota-decorated-rich) collapses the original text node + checkmark,
      // so Svelte's per-option reactive effect never reconciles a removed node.
      option.prepend(img);
      option.append(labelWrap);
      option.classList.add("dota-decorated-rich");
    }} else {{
      option.prepend(img);
    }}
    if (option.dataset) option.dataset.dotaChoiceIconDecorated = "1";
  }};
  const decorate = () => {{
    const options = document.querySelectorAll('li[role="option"], [role="option"]');
    if (options.length) {{
      observer.disconnect();
      options.forEach(decorateOption);
      observer.observe(document.body, {{ childList: true, subtree: true }});
    }}
    syncDropdownMenus();
  }};
  let decorateScheduled = false;
  const scheduleDecorate = () => {{
    if (decorateScheduled) return;
    decorateScheduled = true;
    requestAnimationFrame(() => {{
      decorateScheduled = false;
      decorate();
    }});
  }};
  const openDropdown = (dropdown) => {{
    const input = dropdown?.querySelector(
      'input[autocomplete="off"], input[role="combobox"], input'
    );
    if (!input) return false;
    input.focus();
    input.dispatchEvent(new FocusEvent("focus", {{ bubbles: true }}));
    input.dispatchEvent(new MouseEvent(
      "mousedown",
      {{ bubbles: true, cancelable: true, view: window }}
    ));
    input.click();
    return true;
  }};
  const openFromChevron = (event) => {{
    const icon = closestElement(event.target, ".dota-dropdown .icon-wrap");
    if (!icon) return;
    const dropdown = icon.closest(".dota-dropdown");
    if (!dropdown) return;
    event.preventDefault();
    event.stopPropagation();
    openDropdown(dropdown);
    setTimeout(() => {{
      scheduleDecorate();
    }}, 0);
  }};
  const removeSelectedHero = (event) => {{
    const button = closestElement(event.target, ".hero-card-remove");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const dropdown = document.getElementById(button.dataset.dotaTarget || "");
    if (!dropdown) return;
    const wantedName = comparableHeroName(button.dataset.dotaHeroName || "");
    const token = Array.from(dropdown.querySelectorAll(".token")).find((candidate) => {{
      const tokenName = comparableHeroName(candidate.textContent || "");
      return tokenName === wantedName || tokenName.startsWith(wantedName);
    }});
    const removeButton = token?.querySelector(".token-remove:not(.remove-all)");
    if (removeButton) {{
      removeButton.click();
      setTimeout(decorate, 0);
      return;
    }}
    if (!openDropdown(dropdown)) return;
    setTimeout(() => {{
      const option = Array.from(document.querySelectorAll(
        '[role="option"], .option, .dropdown-option, li[data-index]'
      )).find((candidate) => {{
        const optionName = comparableHeroName(candidate.textContent || "");
        return optionName === wantedName || optionName.startsWith(wantedName);
      }});
      option?.dispatchEvent(new MouseEvent(
        "mousedown",
        {{ bubbles: true, cancelable: true, view: window }}
      ));
      setTimeout(decorate, 0);
    }}, 0);
  }};
  const activateSidebarNav = (event) => {{
    const label = closestElement(event.target, ".app-nav label");
    if (!label) return;
    const input = label.querySelector('input[type="radio"]');
    if (input && !input.checked) {{
      input.checked = true;
      input.dispatchEvent(new Event("input", {{ bubbles: true }}));
      input.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }}
    setTimeout(syncSidebarView, 0);
  }};
  const observer = new MutationObserver(scheduleDecorate);
  observer.observe(document.body, {{ childList: true, subtree: true }});
  let syncScheduled = false;
  const scheduleSync = (event) => {{
    if (isDropdownMenuScroll(event)) return;
    if (syncScheduled) return;
    syncScheduled = true;
    requestAnimationFrame(() => {{
      syncScheduled = false;
      syncDropdownMenus();
    }});
  }};
  document.addEventListener("mousedown", openFromChevron, true);
  document.addEventListener("click", activateSidebarNav, true);
  document.addEventListener("change", syncSidebarView, true);
  document.addEventListener("click", removeSelectedHero, true);
  window.addEventListener("resize", scheduleSync, {{ passive: true }});
  window.addEventListener("scroll", scheduleSync, {{ passive: true, capture: true }});
  decorate();
  syncSidebarView();
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
        .replace("'", "&#39;")
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
        pro_win_rate = row.get("pro_win_rate")
        metadata[hero_id] = {
            "name": hero_name,
            "roles": _format_roles_text(row.get("roles")),
            "icon": _asset_url(row.get("icon") or row.get("img")),
            "image": _asset_url(row.get("img") or row.get("icon")),
            "primary_attr": str(row.get("primary_attr") or "").lower(),
            "pro_pick": int(row.get("pro_pick") or 0),
            "pro_win_rate": float(pro_win_rate) if pro_win_rate is not None else None,
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
        "<div class='app-title'><strong>DOTA2Tuned</strong></div>"
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


def _item_tooltip(item_name: str, item_row: dict[str, object]) -> str:
    lines = [item_name]
    cost = item_row.get("cost")
    if cost:
        lines.append(f"Cost: {int(cost):,}g")
    attrib = item_row.get("attrib")
    if attrib:
        lines.append(str(attrib))
    notes = item_row.get("notes")
    if notes:
        lines.append(str(notes))
    return _html_escape("\n".join(lines))


def _item_metadata(items: pl.DataFrame) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    if items.is_empty():
        return metadata
    for row in items.iter_rows(named=True):
        key = str(row.get("item_key") or "")
        if not key:
            continue
        metadata[key] = {
            "name": str(row.get("item_name") or key.replace("_", " ").title()),
            "cost": row.get("cost"),
            "attrib": row.get("attrib"),
            "notes": row.get("notes"),
        }
    return metadata


def _build_hero_header_html(hero_id: int, metadata: dict[int, dict[str, object]]) -> str:
    row = metadata.get(hero_id, {})
    name = _html_escape(row.get("name") or f"Hero {hero_id}")
    icon = _html_escape(row.get("icon") or "")
    image = _html_escape(row.get("image") or "")
    attr_key = str(row.get("primary_attr") or "")
    attr_label, attr_color = PRIMARY_ATTR_INFO.get(attr_key, ("", ""))
    pills = []
    if attr_label:
        style = f" style='--pill-color: {attr_color}'" if attr_color else ""
        pills.append(f"<span class='build-stat-pill'{style}>{attr_label}</span>")
    pro_win_rate = row.get("pro_win_rate")
    if pro_win_rate is not None:
        pills.append(f"<span class='build-stat-pill'>Pro WR {pro_win_rate * 100:.1f}%</span>")
    pro_pick = row.get("pro_pick") or 0
    if pro_pick:
        pills.append(f"<span class='build-stat-pill'>{pro_pick:,} pro picks</span>")
    icon_html = (
        f"<img class='build-hero-icon' src='{icon}' alt='{name}' loading='lazy'>"
        if icon
        else ""
    )
    style = f" style=\"background-image: url('{image}')\"" if image else ""
    return (
        f"<div class='build-hero-header'{style}>"
        f"{icon_html}"
        "<div class='build-hero-title'>"
        f"<strong>{name}</strong>"
        f"{_role_tags_html(row.get('roles'))}"
        f"<div class='build-stat-row'>{''.join(pills)}</div>"
        "</div></div>"
    )


def _build_top_items_html(
    hero_id: int,
    build_stats: pl.DataFrame,
    item_metadata: dict[str, dict[str, object]],
    top_n: int = 6,
) -> str:
    if build_stats.is_empty():
        return ""
    filtered = build_stats.filter(
        (pl.col("hero_id") == hero_id) & ~pl.col("item_key").str.starts_with("recipe_")
    )
    if filtered.is_empty():
        return ""
    totals = filtered.group_by("item_key").agg(pl.col("purchases").sum().alias("purchases"))
    total = totals["purchases"].sum() or 0
    items_html = []
    for row in totals.sort("purchases", descending=True).head(top_n).iter_rows(named=True):
        item_key = str(row["item_key"])
        item_row = item_metadata.get(item_key, {})
        item_name = _html_escape(item_row.get("name") or item_key.replace("_", " ").title())
        icon = _html_escape(_item_icon_url(item_key))
        tooltip = _item_tooltip(item_row.get("name") or item_name, item_row)
        purchases = int(row["purchases"])
        share = (purchases / total * 100) if total else 0
        items_html.append(
            f"<div class='build-item build-item-highlight' title='{tooltip}'>"
            f"<img src='{icon}' alt='{item_name}' loading='lazy'>"
            "<div class='build-item-meta'>"
            f"<strong>{item_name}</strong>"
            f"<span>{share:.0f}% of buys</span>"
            f"<div class='build-item-bar'><span style='width: {share:.0f}%'></span></div>"
            "</div></div>"
        )
    if not items_html:
        return ""
    return (
        "<div class='build-column build-column-highlight'>"
        "<div class='build-column-title'>Most Popular Items</div>"
        f"<div class='build-item-list build-item-list-row'>{''.join(items_html)}</div>"
        "</div>"
    )


def _build_columns_html(
    hero_id: int,
    build_stats: pl.DataFrame,
    item_metadata: dict[str, dict[str, object]],
    top_n: int = 8,
) -> str:
    if build_stats.is_empty():
        return (
            "<div class='empty-strip'>No build table is available yet. "
            "Run match enrichment and normalization first.</div>"
        )
    filtered = build_stats.filter(
        (pl.col("hero_id") == hero_id) & ~pl.col("item_key").str.starts_with("recipe_")
    )
    if filtered.is_empty():
        return "<div class='empty-strip'>No observed item timings for that hero.</div>"
    columns = []
    for bucket in TIME_BUCKET_ORDER:
        bucket_rows = filtered.filter(pl.col("time_bucket") == bucket)
        if bucket_rows.is_empty():
            continue
        total = bucket_rows["purchases"].sum() or 0
        top_rows = bucket_rows.sort("purchases", descending=True).head(top_n)
        items_html = []
        for order, row in enumerate(
            top_rows.sort("median_time").iter_rows(named=True), start=1
        ):
            item_key = str(row["item_key"])
            item_row = item_metadata.get(item_key, {})
            item_name = _html_escape(item_row.get("name") or item_key.replace("_", " ").title())
            icon = _html_escape(_item_icon_url(item_key))
            tooltip = _item_tooltip(item_row.get("name") or item_name, item_row)
            purchases = int(row["purchases"])
            share = (purchases / total * 100) if total else 0
            median_time = _format_item_time(row.get("median_time"))
            items_html.append(
                f"<div class='build-item' title='{tooltip}'>"
                f"<span class='build-item-order'>{order}</span>"
                f"<img src='{icon}' alt='{item_name}' loading='lazy'>"
                "<div class='build-item-meta'>"
                f"<strong>{item_name}</strong>"
                f"<span>{share:.0f}% of buys &middot; median {median_time}</span>"
                f"<div class='build-item-bar'><span style='width: {share:.0f}%'></span></div>"
                "</div></div>"
            )
        columns.append(
            "<div class='build-column'>"
            "<div class='build-column-title'>"
            f"{_html_escape(TIME_BUCKET_LABELS.get(bucket, bucket))}</div>"
            f"<div class='build-item-list'>{''.join(items_html)}</div>"
            "</div>"
        )
    if not columns:
        return "<div class='empty-strip'>No observed item timings for that hero.</div>"
    return "<div class='build-columns'>" + "".join(columns) + "</div>"


def _selected_hero_html(
    hero_ids: object, metadata: dict[int, dict[str, str]], target: str | None = None
) -> str:
    if not isinstance(hero_ids, list) or not hero_ids:
        return "<div class='hero-strip'></div>"
    chips = []
    for raw_id in hero_ids:
        if raw_id in {None, ""}:
            continue
        try:
            hero_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        row = metadata.get(hero_id, {})
        name = _html_escape(row.get("name") or f"Hero {hero_id}")
        icon = _html_escape(row.get("icon") or "")
        img = f"<img src='{icon}' alt='{name}' loading='lazy'>" if icon else ""
        remove = ""
        if target:
            remove = (
                "<button type='button' class='hero-card-remove' "
                f"data-dota-target='{_html_escape(target)}' "
                f"data-dota-hero-id='{hero_id}' data-dota-hero-name='{name}' "
                f"aria-label='Remove {name}' title='Remove {name}'>x</button>"
            )
        chips.append(
            f"<div class='entity-chip hero-card' title='{name}' data-dota-hero-id='{hero_id}'>"
            f"{img}<div class='hero-card-body'><strong>{name}</strong>"
            f"{_role_tags_html(row.get('roles'))}</div>{remove}</div>"
        )
    if not chips:
        return "<div class='hero-strip'></div>"
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
    item_metadata = _item_metadata(item_table)

    def clean_hero_values(value: object, max_count: int | None = None) -> list[int]:
        hero_ids, _ = _parse_heroes(value, hero_name_lookup)
        return hero_ids[:max_count] if max_count is not None else hero_ids

    def has_transient_null_selection(value: object) -> bool:
        return isinstance(value, list) and any(item is None for item in value)

    def hero_preview(hero_ids: list[int] | None, target: str | None = None) -> str | dict:
        if has_transient_null_selection(hero_ids):
            return gr.skip()
        return _selected_hero_html(clean_hero_values(hero_ids), hero_metadata, target)

    def hero_preview_for(target: str):
        return lambda hero_ids: hero_preview(hero_ids, target)

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

    def hero_builds(hero_id: int | None) -> str:
        hero_ids, unknown = _parse_heroes([hero_id] if hero_id else [], hero_name_lookup)
        if unknown:
            unknown_text = _html_escape(", ".join(unknown))
            return f"<div class='empty-strip'>Unrecognized hero: {unknown_text}</div>"
        if not hero_ids:
            return "<div class='empty-strip'>Select a hero.</div>"
        header = _build_hero_header_html(hero_ids[0], hero_metadata)
        top_build = _build_top_items_html(hero_ids[0], build_stats, item_metadata)
        columns = _build_columns_html(hero_ids[0], build_stats, item_metadata)
        return header + top_build + columns

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

    with gr.Blocks(title="DOTA2Tuned") as demo:
        with gr.Sidebar(open=True, width=238, position="left", elem_classes=["app-sidebar"]):
            gr.HTML(_sidebar_brand_html())
            with gr.Column(elem_classes=["app-sidebar-nav-section"]):
                gr.Radio(
                    choices=NAV_OPTIONS,
                    value="Draft Coach",
                    show_label=False,
                    container=False,
                    # Force interactive: the nav drives view switching purely in JS
                    # (no server event), so without this Gradio renders it disabled.
                    interactive=True,
                    elem_classes=["app-nav"],
                )
        with gr.Column(elem_classes=["app-main"]):
            with gr.Column(
                visible=True, elem_id="draft-coach-view", elem_classes=["app-view", "d2-active"]
            ):
                with gr.Row():
                    allies = gr.Dropdown(
                        choices=hero_choices,
                        label="Allied heroes",
                        multiselect=True,
                        allow_custom_value=True,
                        filterable=True,
                        max_choices=5,
                        elem_id="draft-allies-dropdown",
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    enemies = gr.Dropdown(
                        choices=hero_choices,
                        label="Enemy heroes",
                        value=[44, 30],
                        multiselect=True,
                        allow_custom_value=True,
                        filterable=True,
                        max_choices=5,
                        elem_id="draft-enemies-dropdown",
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    bans = gr.Dropdown(
                        choices=hero_choices,
                        label="Banned heroes",
                        multiselect=True,
                        allow_custom_value=True,
                        filterable=True,
                        elem_id="draft-bans-dropdown",
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )

                    def update_choices(allies_val, enemies_val, bans_val):
                        if any(
                            has_transient_null_selection(value)
                            for value in (allies_val, enemies_val, bans_val)
                        ):
                            return gr.skip(), gr.skip(), gr.skip()
                        allied_ids = clean_hero_values(allies_val, max_count=5)
                        enemy_ids = clean_hero_values(enemies_val, max_count=5)
                        banned_ids = clean_hero_values(bans_val)
                        selected = set(allied_ids) | set(enemy_ids) | set(banned_ids)

                        def filtered(exclude_self):
                            excl = selected - set(exclude_self)
                            return [c for c in hero_choices if c[1] not in excl]

                        return (
                            gr.update(choices=filtered(allied_ids)),
                            gr.update(choices=filtered(enemy_ids)),
                            gr.update(choices=filtered(banned_ids)),
                        )

                    for dd in (allies, enemies, bans):
                        dd.change(
                            update_choices,
                            inputs=[allies, enemies, bans],
                            outputs=[allies, enemies, bans],
                            api_visibility="private",
                            queue=False,
                        )
                with gr.Row():
                    ally_preview = gr.HTML(hero_preview([], "draft-allies-dropdown"))
                    enemy_preview = gr.HTML(hero_preview([44, 30], "draft-enemies-dropdown"))
                    ban_preview = gr.HTML(hero_preview([], "draft-bans-dropdown"))
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
                allies.change(
                    hero_preview_for("draft-allies-dropdown"),
                    inputs=[allies],
                    outputs=[ally_preview],
                    api_visibility="private",
                    queue=False,
                )
                enemies.change(
                    hero_preview_for("draft-enemies-dropdown"),
                    inputs=[enemies],
                    outputs=[enemy_preview],
                    api_visibility="private",
                    queue=False,
                )
                bans.change(
                    hero_preview_for("draft-bans-dropdown"),
                    inputs=[bans],
                    outputs=[ban_preview],
                    api_visibility="private",
                    queue=False,
                )
                run.click(
                    draft_coach,
                    inputs=[allies, enemies, bans, role, scope],
                    outputs=[rec_output, evidence_output],
                )

            with gr.Column(
                visible=True, elem_id="hero-meta-view", elem_classes=["app-view"]
            ):
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
                    api_visibility="private",
                    queue=False,
                )
                meta_item.change(
                    item_preview,
                    inputs=[meta_item],
                    outputs=[meta_item_preview],
                    api_visibility="private",
                    queue=False,
                )
                meta_button.click(
                    hero_meta,
                    inputs=[query, meta_hero, meta_item],
                    outputs=[meta_output],
                )

            with gr.Column(
                visible=True, elem_id="tuned-model-view", elem_classes=["app-view"]
            ):
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

            with gr.Column(
                visible=True, elem_id="match-predictor-view", elem_classes=["app-view"]
            ):
                with gr.Row():
                    radiant = gr.Dropdown(
                        choices=hero_choices,
                        label="Radiant heroes",
                        value=[1, 2, 3, 25, 5],
                        multiselect=True,
                        allow_custom_value=True,
                        filterable=True,
                        max_choices=5,
                        elem_id="predict-radiant-dropdown",
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                    dire = gr.Dropdown(
                        choices=hero_choices,
                        label="Dire heroes",
                        value=[14, 74, 6, 26, 18],
                        multiselect=True,
                        allow_custom_value=True,
                        filterable=True,
                        max_choices=5,
                        elem_id="predict-dire-dropdown",
                        elem_classes=["dota-dropdown", "hero-dropdown"],
                    )
                with gr.Row():
                    radiant_preview = gr.HTML(
                        hero_preview([1, 2, 3, 25, 5], "predict-radiant-dropdown")
                    )
                    dire_preview = gr.HTML(
                        hero_preview([14, 74, 6, 26, 18], "predict-dire-dropdown")
                    )
                predict_button = gr.Button("Predict")
                predict_output = gr.Code(label="Prediction", language="json")
                radiant.change(
                    hero_preview_for("predict-radiant-dropdown"),
                    inputs=[radiant],
                    outputs=[radiant_preview],
                    api_visibility="private",
                    queue=False,
                )
                dire.change(
                    hero_preview_for("predict-dire-dropdown"),
                    inputs=[dire],
                    outputs=[dire_preview],
                    api_visibility="private",
                    queue=False,
                )
                predict_button.click(
                    match_predictor,
                    inputs=[radiant, dire],
                    outputs=[predict_output],
                )

            with gr.Column(
                visible=True, elem_id="builds-view", elem_classes=["app-view"]
            ):
                hero = gr.Dropdown(
                    choices=hero_choices,
                    label="Hero",
                    value=1,
                    filterable=True,
                    elem_classes=["dota-dropdown", "hero-dropdown"],
                )
                build_hero_preview = gr.HTML(hero_preview([1]))
                builds_output = gr.HTML(hero_builds(1))
                hero.change(
                    hero_single_preview,
                    inputs=[hero],
                    outputs=[build_hero_preview],
                    api_visibility="private",
                    queue=False,
                )
                hero.change(hero_builds, inputs=[hero], outputs=[builds_output])

            with gr.Column(
                visible=True, elem_id="draft-lab-view", elem_classes=["app-view"]
            ):
                lab_enemies = gr.Dropdown(
                    choices=hero_choices,
                    label="Enemy heroes",
                    value=[44, 30],
                    multiselect=True,
                    allow_custom_value=True,
                    filterable=True,
                    max_choices=5,
                    elem_id="lab-enemies-dropdown",
                    elem_classes=["dota-dropdown", "hero-dropdown"],
                )
                lab_enemy_preview = gr.HTML(hero_preview([44, 30], "lab-enemies-dropdown"))
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
                lab_enemies.change(
                    hero_preview_for("lab-enemies-dropdown"),
                    inputs=[lab_enemies],
                    outputs=[lab_enemy_preview],
                    api_visibility="private",
                    queue=False,
                )
                lab_button.click(
                    draft_lab,
                    inputs=[lab_enemies, lab_role, lab_twist],
                    outputs=[lab_output],
                )

            with gr.Column(
                visible=True, elem_id="data-freshness-view", elem_classes=["app-view"]
            ):
                status_button = gr.Button("Refresh")
                status_output = gr.Markdown()
                status_button.click(data_status, outputs=[status_output])

        # Gradio 6.18 runs client JS only via the load EVENT — launch(js=) and
        # Blocks(js=) do not fire on page load. Attach the script here so dropdown
        # icons, sidebar nav switching, and hero-card removal actually work.
        demo.load(
            None,
            None,
            None,
            js=_dropdown_js(_dropdown_icon_by_label(hero_choices, hero_metadata)),
        )
    return demo
