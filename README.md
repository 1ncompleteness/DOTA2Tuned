---
title: DOTA2Tuned
sdk: gradio
app_file: app.py
python_version: "3.12"
tags:
  - track:backyard
  - track:wood
  - sponsor:openbmb
  - sponsor:openai
  - sponsor:modal
  - achievement:welltuned
  - achievement:offbrand
  - achievement:sharing
models:
- build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
- build-small-hackathon/dota2tuned-minicpm4-1-8b-lora
- build-small-hackathon/dota2tuned-qwen3-30b-a3b-2507-lora
datasets:
- build-small-hackathon/dota2tuned-data
---

# DOTA2Tuned

DOTA2Tuned is a Hugging Face Build Small Hackathon project for Dota 2 drafting, meta analysis, build suggestions, and match prediction.

The implementation is designed around a simple rule: stats and predictors choose the recommendations, while a small fine-tuned model explains them with patch-aware evidence.

- Hugging Face Space: https://build-small-hackathon-dota2tuned.hf.space
- Modal alternate UI: https://dracufeuer--dota2tuned-ui.modal.run
- Tiny fine-tuned adapter: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
- Balanced fine-tuned adapter: https://huggingface.co/build-small-hackathon/dota2tuned-minicpm4-1-8b-lora
- Quality fine-tuned adapter: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-30b-a3b-2507-lora
- Dataset artifacts: https://huggingface.co/datasets/build-small-hackathon/dota2tuned-data
- Demo video: https://drive.google.com/file/d/12l0sKN-rJJ3RwKDEE-TGbXfiRMepN_zz/view?usp=drive_link
- Social post: TODO add final social post URL before validation.

## Hackathon Validation

The Build Small validator checks this Space README. Entry requirements are: sub-32B models, Gradio Space in the Build Small org, demo video, social-media post linked from this README, and GPU-limit compliance.

Validator: https://build-small-hackathon-field-guide.hf.space/submit

Selected tracks, prizes, and badges:

- Tracks: Backyard AI, Thousand Token Wood.
- Sponsor prizes: OpenBMB Best MiniCPM Build, OpenAI Best Use of Codex, Modal Best Use of Modal.
- Bonus badges: Well-Tuned, Off-Brand, Sharing is Caring.
- Not claimed: Nemotron Hardware Prize, Off the Grid, Llama Champion.
- GPU note: this Space does not use Zero GPU allocation; training and tuned-model serving use Modal.

Social post draft:

> Built DOTA2Tuned for the Hugging Face Build Small Hackathon: a Gradio Dota 2 draft coach that combines STRATZ/OpenDota match evidence, deterministic draft stats, and fine-tuned sub-32B adapters for grounded explanations.
>
> It suggests heroes, counters, synergies, builds, match predictions, and caveats weak data instead of inventing unsupported meta claims.
>
> Space: https://build-small-hackathon-dota2tuned.hf.space
> Tiny: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
> Balanced: https://huggingface.co/build-small-hackathon/dota2tuned-minicpm4-1-8b-lora
> Quality: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-30b-a3b-2507-lora
> Repo: https://github.com/1ncompleteness/DOTA2Tuned

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
Fine-tuning now runs on Modal by default for this project. Set `MODAL_ENABLED=1`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET`, then use `uv sync --extra modal` before deploying Modal functions. `HF_TOKEN` must still include `repo.write` so the training run can push adapters to the configured Hub model repos. Use `MODEL_PROFILE` or `--profile` to select Tiny, Balanced, or Quality.

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
- `dota2tuned modal-train --profile qwen3_4b_2507` submits the Tiny Modal GPU QLoRA run. Other profiles include `minicpm4_1_8b` and `qwen3_30b_a3b_2507`; Quality routes to the H200-backed `train_sft_quality` function.
- `uv run python scripts/watch_modal_training.py --call tiny=fc-... --interval 300` polls detached Modal training calls without blocking local development.
- `dota2tuned modal-ask` calls the fine-tuned adapter through Modal GPU inference.
- `dota2tuned finetune --launch-job` remains available for Hugging Face Jobs if a token has `job.write`.
- `dota2tuned serve` launches the Gradio app.

The Gradio app includes searchable hero, item, role, scope, and ability dropdowns, alias-aware hero lookup (`PA`, `CM`, `AM`, `QOP`, `KOTL`, and curated initials), selected icon previews, and six sidebar modules: Ask, Draft, Meta, Builds, Predictor, and Data. Ask calls the Modal `generate_answer` function when Modal credentials are configured in the runtime environment; otherwise it degrades with a clear unavailable message.

See [PLAN.md](PLAN.md) for the full architecture and delivery plan.
See [MODEL_SELECTION.md](MODEL_SELECTION.md) for the current LLM decision and eval protocol.
See [MODEL_CARD.md](MODEL_CARD.md) for the model card mirrored to the Hub adapter repo.
See [DATASET_CARD.md](DATASET_CARD.md) for the dataset card mirrored to the Hub dataset.
See [SUBMISSION.md](SUBMISSION.md) for the final hackathon handoff, demo script, social post draft, and verification checklist.
