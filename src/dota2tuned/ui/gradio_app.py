from __future__ import annotations

import json

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


def build_app() -> gr.Blocks:
    settings = get_settings()
    recommender = DraftRecommender(settings.parquet_dir)
    retriever = Retriever(settings.rag_dir)

    def draft_coach(allies: str, enemies: str, bans: str, role: str, scope: str) -> tuple[str, str]:
        draft = DraftInput(
            allied_heroes=_parse_ids(allies),
            enemy_heroes=_parse_ids(enemies),
            banned_heroes=_parse_ids(bans),
            role=role or None,
            scope=scope,
            patch="current",
        )
        recs = recommender.recommend(draft, limit=8)
        evidence = retriever.search(
            " ".join([role or "", allies, enemies, bans]), patch="current", limit=5
        )
        return _format_recs(recs), json.dumps(evidence, indent=2)

    def hero_meta(query: str) -> str:
        docs = retriever.search(query or "current meta", patch="current", limit=8)
        if not docs:
            return "No retrieval index is available yet. Run `dota2tuned build-rag` first."
        return "\n\n".join(f"**{doc['source']}** `{doc['score']}`\n{doc['text']}" for doc in docs)

    def data_status() -> str:
        files = []
        for path in sorted(settings.parquet_dir.glob("*.parquet")):
            files.append(f"- `{path.name}` ({path.stat().st_size:,} bytes)")
        if not files:
            return "No normalized Parquet files found."
        return "\n".join(files)

    build_stats = read_parquet(settings.parquet_dir / "fact_hero_build_stats.parquet")

    def match_predictor(radiant: str, dire: str) -> str:
        prediction = predict_draft_win(settings.model_dir, _parse_ids(radiant), _parse_ids(dire))
        if prediction.get("status") != "ok":
            return prediction.get("message", "Prediction unavailable.")
        return json.dumps(prediction, indent=2)

    def hero_builds(hero: str) -> str:
        hero_ids = _parse_ids(hero)
        if not hero_ids:
            return "Enter a numeric hero ID."
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
            minutes = round(float(row.get("median_time") or 0) / 60, 1)
            lines.append(
                f"- **{row.get('item_key')}** in `{row.get('time_bucket')}`: "
                f"{row.get('purchases')} purchases, median `{minutes}` min"
            )
        return "\n".join(lines) if lines else "No observed item timings for that hero."

    with gr.Blocks(title="DOTA2Tuned") as demo, gr.Tabs():
        with gr.Tab("Draft Coach"):
            with gr.Row():
                allies = gr.Textbox(label="Allied hero IDs", placeholder="1, 2, 3")
                enemies = gr.Textbox(label="Enemy hero IDs", placeholder="4, 5, 6")
                bans = gr.Textbox(label="Banned hero IDs", placeholder="7, 8")
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

        with gr.Tab("Match Predictor"):
            with gr.Row():
                radiant = gr.Textbox(label="Radiant hero IDs", placeholder="1, 2, 3, 4, 5")
                dire = gr.Textbox(label="Dire hero IDs", placeholder="6, 7, 8, 9, 10")
            predict_button = gr.Button("Predict")
            predict_output = gr.Code(label="Prediction", language="json")
            predict_button.click(match_predictor, inputs=[radiant, dire], outputs=[predict_output])

        with gr.Tab("Builds"):
            hero = gr.Textbox(label="Hero ID", placeholder="1")
            builds_button = gr.Button("Show Builds")
            builds_output = gr.Markdown()
            builds_button.click(hero_builds, inputs=[hero], outputs=[builds_output])

        with gr.Tab("Draft Lab"):
            gr.Markdown(
                "Use Draft Coach with playful constraints for the Thousand Token Wood demo mode."
            )

        with gr.Tab("Data Freshness"):
            status_button = gr.Button("Refresh")
            status_output = gr.Markdown()
            status_button.click(data_status, outputs=[status_output])

    return demo
