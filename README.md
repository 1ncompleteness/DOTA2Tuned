---
title: DOTA2Tuned
sdk: gradio
app_file: app.py
python_version: "3.12"
---

# DOTA2Tuned

DOTA2Tuned is a Hugging Face Build Small Hackathon project for Dota 2 drafting, meta analysis, build suggestions, and match prediction.

The implementation is designed around a simple rule: stats and predictors choose the recommendations, while a small fine-tuned model explains them with patch-aware evidence.

## Quick Start

```bash
uv sync
cp .env.example .env
uv run dota2tuned health
uv run dota2tuned smoke --live
uv run dota2tuned ingest --pro-matches 100 --public-matches 100 --enrich-limit 20
uv run dota2tuned normalize
uv run dota2tuned features
uv run dota2tuned build-rag
uv run dota2tuned serve
```

Update `.env` with your Hugging Face, STRATZ, OpenDota, and Steam tokens before running large ingestion or Hub operations.

## Main Commands

- `dota2tuned ingest` fetches raw reference data, pro matches, public matches, patch notes, and optional match enrichment.
- `dota2tuned smoke --live` validates configured tokens with tiny live API checks.
- `dota2tuned normalize` converts raw JSONL into Parquet tables and refreshes DuckDB views.
- `dota2tuned features` refreshes DuckDB views and reports feature table row counts.
- `dota2tuned train-predictor` trains the draft win predictor from normalized matches.
- `dota2tuned build-rag` creates patch/stat cards and a local retrieval index.
- `dota2tuned make-sft` creates JSONL examples for SFT.
- `dota2tuned finetune --launch-job` launches a Hugging Face Jobs QLoRA run.
- `dota2tuned serve` launches the Gradio app.

See [PLAN.md](PLAN.md) for the full architecture and delivery plan.
See [MODEL_SELECTION.md](MODEL_SELECTION.md) for the current LLM decision and eval protocol.
