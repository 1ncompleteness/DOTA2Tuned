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
    return by_name, by_id


def _parse_heroes(value: str, lookup: dict[str, int]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    unknown: list[str] = []
    seen: set[int] = set()
    for ref in _split_hero_refs(value):
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


def _format_item_time(seconds: object) -> str:
    value = float(seconds or 0)
    if value < 0:
        return "pre-game"
    return f"{round(value / 60, 1)} min"


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

    def draft_coach(allies: str, enemies: str, bans: str, role: str, scope: str) -> tuple[str, str]:
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
        evidence = retriever.search(
            " ".join([role or "", allies, enemies, bans]), patch="current", limit=5
        )
        unknown = allied_unknown + enemy_unknown + banned_unknown
        warning = f"Unrecognized heroes ignored: {', '.join(unknown)}\n\n" if unknown else ""
        return warning + _format_recs(recs), json.dumps(evidence, indent=2)

    def hero_meta(query: str) -> str:
        docs = retriever.search(query or "current meta", patch="current", limit=8)
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

    def match_predictor(radiant: str, dire: str) -> str:
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

    def hero_builds(hero: str) -> str:
        hero_ids, unknown = _parse_heroes(hero, hero_name_lookup)
        if unknown:
            return f"Unrecognized hero: {', '.join(unknown)}"
        if not hero_ids:
            return "Enter a hero name or numeric hero ID."
        if build_stats.is_empty():
            return "No build table is available yet. Run match enrichment and normalization first."
        rows = (
            build_stats.filter(pl.col("hero_id") == hero_ids[0])
            .sort("purchases", descending=True)
            .head(15)
            .iter_rows(named=True)
        )
        lines = []
        for row in rows:
            median_time = _format_item_time(row.get("median_time"))
            lines.append(
                f"- **{row.get('item_key')}** in `{row.get('time_bucket')}`: "
                f"{row.get('purchases')} purchases, median `{median_time}`"
            )
        return "\n".join(lines) if lines else "No observed item timings for that hero."

    def draft_lab(enemies: str, role: str, twist: str) -> str:
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
                allies = gr.Textbox(label="Allied heroes", placeholder="Anti-Mage, Axe or 1, 2")
                enemies = gr.Textbox(
                    label="Enemy heroes", placeholder="Phantom Assassin, Witch Doctor"
                )
                bans = gr.Textbox(label="Banned heroes", placeholder="Pudge, Invoker or 7, 8")
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
            run.click(
                draft_coach,
                inputs=[allies, enemies, bans, role, scope],
                outputs=[rec_output, evidence_output],
            )

        with gr.Tab("Hero Meta"):
            query = gr.Textbox(
                label="Hero, item, patch, or meta question", value="current pro meta"
            )
            meta_button = gr.Button("Search")
            meta_output = gr.Markdown()
            meta_button.click(hero_meta, inputs=[query], outputs=[meta_output])

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
                radiant = gr.Textbox(
                    label="Radiant heroes", placeholder="Anti-Mage, Axe, Bane, Lina, Crystal Maiden"
                )
                dire = gr.Textbox(
                    label="Dire heroes", placeholder="Pudge, Invoker, Drow Ranger, Lion, Sven"
                )
            predict_button = gr.Button("Predict")
            predict_output = gr.Code(label="Prediction", language="json")
            predict_button.click(match_predictor, inputs=[radiant, dire], outputs=[predict_output])

        with gr.Tab("Builds"):
            hero = gr.Textbox(label="Hero", placeholder="Anti-Mage or 1")
            builds_button = gr.Button("Show Builds")
            builds_output = gr.Markdown()
            builds_button.click(hero_builds, inputs=[hero], outputs=[builds_output])

        with gr.Tab("Draft Lab"):
            lab_enemies = gr.Textbox(
                label="Enemy heroes", value="Phantom Assassin, Witch Doctor"
            )
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
