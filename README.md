---
title: DOTA2Tuned
sdk: gradio
app_file: app.py
python_version: "3.12"
---

# DOTA2Tuned

DOTA2Tuned is a Hugging Face Build Small Hackathon project for Dota 2 drafting, meta analysis, build suggestions, and match prediction.

The implementation is designed around a simple rule: stats and predictors choose the recommendations, while a small fine-tuned model explains them with patch-aware evidence.

- Hugging Face Space: https://build-small-hackathon-dota2tuned.hf.space
- Modal alternate UI: https://dracufeuer--dota2tuned-ui.modal.run
- Fine-tuned adapter: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
- Dataset artifacts: https://huggingface.co/datasets/build-small-hackathon/dota2tuned-data

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
Fine-tuning now runs on Modal by default for this project. Set `MODAL_ENABLED=1`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET`, then use `uv sync --extra modal` before deploying Modal functions. `HF_TOKEN` must still include `repo.write` so the training run can push the adapter to the configured Hub model repo.

The submitted Space includes compact serving artifacts under `data/parquet`, `data/rag`, and `data/models`. Raw API responses remain local-only and ignored by git.

## Main Commands

- `dota2tuned ingest` fetches raw reference data, pro matches, public matches, patch notes, and optional match enrichment.
- `dota2tuned smoke --live` validates configured tokens with tiny live API checks.
- `dota2tuned normalize` converts raw JSONL into Parquet tables and refreshes DuckDB views.
- `dota2tuned features` refreshes DuckDB views and reports feature table row counts.
- `dota2tuned train-predictor` trains the draft win predictor from normalized matches.
- `dota2tuned build-rag` creates patch/stat cards and a local retrieval index.
- `dota2tuned make-sft` creates JSONL examples for SFT.
- `dota2tuned modal-deploy` deploys the Modal Gradio app and GPU training function.
- `dota2tuned modal-smoke` validates the deployed Modal app can load artifacts.
- `dota2tuned modal-train` submits the Modal GPU QLoRA run.
- `dota2tuned modal-ask` calls the fine-tuned adapter through Modal GPU inference.
- `dota2tuned finetune --launch-job` remains available for Hugging Face Jobs if a token has `job.write`.
- `dota2tuned serve` launches the Gradio app.

The Gradio app includes a **Tuned Model** tab. It calls the Modal `generate_answer` function when Modal credentials are configured in the runtime environment; otherwise the tab degrades with a clear unavailable message.

See [PLAN.md](PLAN.md) for the full architecture and delivery plan.
See [MODEL_SELECTION.md](MODEL_SELECTION.md) for the current LLM decision and eval protocol.
See [DATASET_CARD.md](DATASET_CARD.md) for the dataset card mirrored to the Hub dataset.
See [SUBMISSION.md](SUBMISSION.md) for the final hackathon handoff, demo script, social post draft, and verification checklist.
