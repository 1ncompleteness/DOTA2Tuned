from __future__ import annotations

import json
import os
import re

import gradio as gr
import polars as pl

from dota2tuned.config import get_settings
from dota2tuned.rag import Retriever
from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput
from dota2tuned.storage import read_parquet
from dota2tuned.train_predictor import predict_draft_win

ASSET_BASE_URL = "https://cdn.cloudflare.steamstatic.com"

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
    "Queen of Pain": ["QOP"],
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
    "Zeus": ["Z"],
}

APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(128, 32, 32, 0.10), transparent 34rem),
    linear-gradient(180deg, #101414 0%, #151716 100%);
}
.hero-strip, .item-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
  margin-top: 6px;
}
.entity-chip {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 42px;
  padding: 7px 8px;
  border: 1px solid rgba(220, 190, 120, 0.24);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.045);
}
.entity-chip img {
  width: 32px;
  height: 32px;
  object-fit: contain;
  border-radius: 5px;
}
.entity-chip strong {
  display: block;
  font-size: 13px;
  line-height: 16px;
}
.entity-chip span {
  display: block;
  color: rgba(245, 245, 235, 0.68);
  font-size: 11px;
  line-height: 14px;
}
.empty-strip {
  color: rgba(245, 245, 235, 0.60);
  font-size: 13px;
  padding: 7px 0;
}
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
    if not words:
        return set()
    aliases = {"".join(word[0] for word in words)}
    non_stopwords = [word for word in words if word.lower() not in {"of", "the", "and"}]
    if non_stopwords:
        aliases.add("".join(word[0] for word in non_stopwords))
    return {alias.upper() for alias in aliases if len(alias) <= 5}


def _hero_aliases(hero_name: str) -> list[str]:
    aliases = set(COMMON_HERO_ALIASES.get(hero_name, []))
    aliases.update(_initial_aliases(hero_name))
    aliases.discard(hero_name)
    return sorted(aliases, key=lambda item: (len(item), item))


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
            "roles": str(row.get("roles") or ""),
            "icon": _asset_url(row.get("icon") or row.get("img")),
            "aliases": ", ".join(_hero_aliases(hero_name)),
        }
    return metadata


def _hero_choices(heroes: pl.DataFrame) -> list[tuple[str, int]]:
    metadata = _hero_metadata(heroes)
    choices = []
    for hero_id, row in sorted(metadata.items(), key=lambda item: item[1]["name"]):
        aliases = f" ({row['aliases']})" if row["aliases"] else ""
        role_text = f" - {row['roles']}" if row["roles"] else ""
        choices.append((f"{row['name']}{aliases}{role_text}", hero_id))
    return choices


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
        return "<div class='empty-strip'>No heroes selected.</div>"
    chips = []
    for raw_id in hero_ids:
        hero_id = int(raw_id)
        row = metadata.get(hero_id, {})
        name = _html_escape(row.get("name") or f"Hero {hero_id}")
        roles = _html_escape(row.get("roles") or "")
        icon = _html_escape(row.get("icon") or "")
        img = f"<img src='{icon}' alt='{name}' loading='lazy'>" if icon else ""
        chips.append(
            f"<div class='entity-chip'>{img}<div><strong>{name}</strong>"
            f"<span>{roles}</span></div></div>"
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

    with gr.Blocks(title="DOTA2Tuned") as demo, gr.Tabs():
        with gr.Tab("Draft Coach"):
            with gr.Row():
                allies = gr.Dropdown(
                    choices=hero_choices,
                    label="Allied heroes",
                    multiselect=True,
                    filterable=True,
                    max_choices=5,
                )
                enemies = gr.Dropdown(
                    choices=hero_choices,
                    label="Enemy heroes",
                    value=[44, 30],
                    multiselect=True,
                    filterable=True,
                    max_choices=5,
                )
                bans = gr.Dropdown(
                    choices=hero_choices,
                    label="Banned heroes",
                    multiselect=True,
                    filterable=True,
                )
            with gr.Row():
                ally_preview = gr.HTML(hero_preview([]))
                enemy_preview = gr.HTML(hero_preview([44, 30]))
                ban_preview = gr.HTML(hero_preview([]))
            with gr.Row():
                role = gr.Dropdown(
                    ["carry", "mid", "offlane", "soft support", "hard support"],
                    label="Role",
                    value="mid",
                )
                scope = gr.Dropdown(["pro", "high-rank", "public"], label="Scope", value="pro")
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

        with gr.Tab("Hero Meta"):
            with gr.Row():
                meta_hero = gr.Dropdown(
                    choices=hero_choices,
                    label="Hero",
                    filterable=True,
                )
                meta_item = gr.Dropdown(
                    choices=item_choices,
                    label="Item",
                    filterable=True,
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

        with gr.Tab("Tuned Model"):
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

        with gr.Tab("Match Predictor"):
            with gr.Row():
                radiant = gr.Dropdown(
                    choices=hero_choices,
                    label="Radiant heroes",
                    value=[1, 2, 3, 25, 5],
                    multiselect=True,
                    filterable=True,
                    max_choices=5,
                )
                dire = gr.Dropdown(
                    choices=hero_choices,
                    label="Dire heroes",
                    value=[14, 74, 6, 26, 18],
                    multiselect=True,
                    filterable=True,
                    max_choices=5,
                )
            with gr.Row():
                radiant_preview = gr.HTML(hero_preview([1, 2, 3, 25, 5]))
                dire_preview = gr.HTML(hero_preview([14, 74, 6, 26, 18]))
            predict_button = gr.Button("Predict")
            predict_output = gr.Code(label="Prediction", language="json")
            radiant.change(hero_preview, inputs=[radiant], outputs=[radiant_preview])
            dire.change(hero_preview, inputs=[dire], outputs=[dire_preview])
            predict_button.click(match_predictor, inputs=[radiant, dire], outputs=[predict_output])

        with gr.Tab("Builds"):
            with gr.Row():
                hero = gr.Dropdown(
                    choices=hero_choices,
                    label="Hero",
                    value=1,
                    filterable=True,
                )
                build_item = gr.Dropdown(
                    choices=item_choices,
                    label="Optional item filter",
                    filterable=True,
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

        with gr.Tab("Draft Lab"):
            lab_enemies = gr.Dropdown(
                choices=hero_choices,
                label="Enemy heroes",
                value=[44, 30],
                multiselect=True,
                filterable=True,
                max_choices=5,
            )
            lab_enemy_preview = gr.HTML(hero_preview([44, 30]))
            lab_role = gr.Dropdown(
                ["carry", "mid", "offlane", "soft support", "hard support"],
                label="Role",
                value="mid",
            )
            lab_twist = gr.Dropdown(
                ["Tiny scout card", "One-minute coach", "Chaos constraint"],
                label="Mode",
                value="Tiny scout card",
            )
            lab_button = gr.Button("Generate Draft Lab Card")
            lab_output = gr.Markdown()
            lab_enemies.change(hero_preview, inputs=[lab_enemies], outputs=[lab_enemy_preview])
            lab_button.click(
                draft_lab,
                inputs=[lab_enemies, lab_role, lab_twist],
                outputs=[lab_output],
            )

        with gr.Tab("Data Freshness"):
            status_button = gr.Button("Refresh")
            status_output = gr.Markdown()
            status_button.click(data_status, outputs=[status_output])

    return demo
