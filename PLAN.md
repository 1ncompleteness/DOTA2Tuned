# DOTA2Tuned Hackathon Plan

## Summary

- Build a full Dota 2 data platform plus a GPU-backed Hugging Face Gradio Space for the Build Small Hackathon.
- Optimize for both tracks: a serious draft/meta coach for Backyard AI and a playful Draft Lab mode for Thousand Token Wood.
- Use deterministic stats and predictors for recommendations; use a fine-tuned 3-4B QLoRA model for grounded explanations.
- Keep Hugging Face Space as the canonical submitted app. Use Modal only as an optional backend accelerator, per the Gradio Modal guide's sticky-session caveat.
- First implementation step: create `PLAN.md`, `.env.example`, `.env`, and `.gitignore` secret rules, then build the code scaffold.

## Source Constraints

- Hackathon: `<=32B` total model parameters, Gradio app hosted as a Hugging Face Space, Space link + short demo video + social post by June 15, 2026. Source: https://huggingface.co/build-small-hackathon
- Hugging Face: use HF Jobs + TRL SFT/QLoRA for training, push model/dataset artifacts to Hub, serve through GPU Space.
- Modal: do not host the primary UI there unless needed; Gradio-on-Modal requires `mount_gradio_app`, `@modal.asgi_app()`, and `max_containers=1` to satisfy sticky sessions. Preferred use is separate GPU functions called from the Space.
- API limits: OpenDota free metadata reports `3000/day`, `60/min`, premium `3000/min`; STRATZ default token is `20/sec`, `250/min`, `2000/hour`, `10000/day`.

## Architecture

- Stack: Python 3.11, `uv`, `src/` layout, `ruff`, `pytest`, `pydantic`, `httpx`, `tenacity`, `polars`, `pyarrow`, `duckdb`, `scikit-learn`, `lightgbm`, `datasets`, `huggingface_hub`, `transformers`, `trl`, `peft`, `bitsandbytes`, `gradio`, optional `modal`.
- Storage: raw API JSONL in `data/raw`, normalized Parquet in `data/parquet`, DuckDB at `data/dota2tuned.duckdb`, generated RAG docs in `data/rag`.
- Tables: `dim_patch`, `dim_hero`, `dim_item`, `dim_league`, `fact_match`, `fact_player_match`, `fact_draft_pickban`, `fact_item_purchase`, `fact_hero_pair_stats`, `fact_hero_build_stats`, `doc_patch_change`, `doc_stat_card`, `ingest_run`, `api_call_log`.
- Data clients: Steam discovery/raw facts, OpenDota normalized stats, STRATZ rich enrichment, Valve patch JSON feed, dotaconstants constants.
- Modal backend only: optional functions for batch ETL, expensive evals, and GPU inference fallback; the HF Space calls Modal only when configured.

## Interfaces

- CLI: `dota2tuned ingest`, `normalize`, `features`, `train-predictor`, `build-rag`, `make-sft`, `finetune`, `eval`, `serve`, `modal-deploy`.
- `.env` keys: `HF_TOKEN`, `HF_ORG`, `HF_SPACE_ID`, `HF_MODEL_REPO_ID`, `HF_DATASET_REPO_ID`, `STRATZ_TOKEN`, `OPENDOTA_API_KEY`, `STEAM_API_KEY`, `BASE_MODEL_ID`, `TRAINING_FLAVOR`, `SPACE_HARDWARE`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_APP_NAME`, `MODAL_ENABLED`, `DUCKDB_PATH`, `RAW_DATA_DIR`, `PARQUET_DIR`.
- Recommendation schema: `hero_id`, `hero_name`, `role`, `score`, `win_prob_delta`, `counter_lift`, `synergy_lift`, `sample_size`, `patch`, `scope`, `sources`, `confidence`, `caveats`.
- Gradio tabs: Draft Coach, Match Predictor, Hero Meta, Builds, Data Freshness, Draft Lab.

## Pipeline

- Ingest pro tournaments first: OpenDota `/proMatches`, professional `/leagues`, `/leagues/{league_id}/matches`, then `/matches/{match_id}` and STRATZ `match(id)` enrichment.
- Initial dataset mix: `65%` pro/tournament, `25%` high-rank current-patch public, `10%` broad current-patch public for rare hero coverage.
- Budget defaults: OpenDota free day uses about `1900` pro/current league calls, `750` high-rank/public enrichment calls, `250` constants/retries; STRATZ default day uses `7000` pro enrich, `2000` high-rank/public enrich, `1000` retry/schema reserve.
- Prediction: train calibrated logistic baseline and LightGBM draft predictor; choose by temporal patch validation.
- Recommendations: rank heroes by predicted win-probability delta plus empirical counter/synergy lift; builds come from observed item/skill/talent timing stats, not LLM invention.
- RAG: index patch-change docs and stat cards with patch/scope filters before retrieval.
- Fine-tuning: default `Qwen/Qwen3-4B-Instruct`; fallback `HuggingFaceTB/SmolLM3-3B`. Use QLoRA with TRL SFT on HF Jobs, 1-2 epochs, structured answer examples, then push to Hub.
- Serving: GPU-backed HF Gradio Space loads predictor, DuckDB/RAG artifacts, and fine-tuned model; Modal is an optional backend for expensive jobs only.

## Test Plan

- Unit tests: API clients, auth headers, rate limiters, retry/cache keys, patch parser, normalizers, schema validation.
- Data checks: no duplicate match IDs, valid hero/item IDs, correct patch join by `start_time`, no abandoned/remake training matches, no future-patch leakage.
- Model evals: ROC-AUC, log loss, Brier/calibration for match prediction; Recall@K/NDCG@K for recommendations; citation support rate for generated answers.
- App checks: Gradio import/launch smoke, GPU availability logging, no secret leakage, deterministic fallback when model/backend unavailable.
- Modal checks: disabled-by-default local behavior, backend function smoke when `MODAL_ENABLED=1`, no Gradio UI hosted on Modal unless explicitly selected later.

## Assumptions

- The team can create the required HF Space before the June 15, 2026 deadline.
- Hugging Face GPU credits are available for QLoRA training and GPU Space inference.
- Modal is backend-only, not the primary submission UI.
- STRATZ/OpenDota/Steam tokens will be supplied in `.env`; no tokens will be committed.
