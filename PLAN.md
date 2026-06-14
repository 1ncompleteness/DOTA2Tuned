# DOTA2Tuned Hackathon Plan

## Summary

- Build a full Dota 2 data platform plus a Hugging Face Gradio Space for the Build Small Hackathon, backed by Modal GPU functions for training and optional serving.
- Optimize for both tracks: a serious draft/meta coach for Backyard AI and a playful Draft Lab mode for Thousand Token Wood.
- Use deterministic stats and predictors for recommendations; use a fine-tuned 3-4B QLoRA model for grounded explanations.
- Keep Hugging Face Space as the canonical submitted app, with Modal as the GPU execution path because Modal credits are available.
- First implementation step: create `PLAN.md`, `.env.example`, `.env`, and `.gitignore` secret rules, then build the code scaffold.

## Source Constraints

- Hackathon: `<=32B` total model parameters, Gradio app hosted as a Hugging Face Space, Space link + short demo video + social post by June 15, 2026. Source: https://huggingface.co/build-small-hackathon
- Hugging Face: host the submitted Space and model/dataset artifacts on the Hub.
- Modal: use TRL SFT/QLoRA for GPU training, and optionally host the Gradio app with `mount_gradio_app`, `@modal.asgi_app()`, and `max_containers=1` to satisfy Gradio sticky-session requirements.
- API limits: OpenDota free metadata reports `3000/day`, `60/min`, premium `3000/min`; STRATZ default token is `20/sec`, `250/min`, `2000/hour`, `10000/day`.

## Architecture

- Stack: Python 3.11, `uv`, `src/` layout, `ruff`, `pytest`, `pydantic`, `httpx`, `tenacity`, `polars`, `pyarrow`, `duckdb`, `scikit-learn`, `lightgbm`, `datasets`, `huggingface_hub`, `transformers`, `trl`, `peft`, `bitsandbytes`, `gradio`, optional `modal`.
- Storage: raw API JSONL in `data/raw`, normalized Parquet in `data/parquet`, DuckDB at `data/dota2tuned.duckdb`, generated RAG docs in `data/rag`.
- Tables: `dim_patch`, `dim_hero`, `dim_item`, `dim_league`, `fact_match`, `fact_player_match`, `fact_draft_pickban`, `fact_item_purchase`, `fact_hero_pair_stats`, `fact_hero_build_stats`, `doc_patch_change`, `doc_stat_card`, `ingest_run`, `api_call_log`.
- Data clients: Steam discovery/raw facts, OpenDota normalized stats, STRATZ rich enrichment, Valve patch JSON feed, dotaconstants constants.
- Modal backend: deployed `ui`, `remote_smoke`, `train_sft`, and `generate_answer` functions. The HF Space remains available; Modal provides GPU training, adapter inference, and a verified alternate Gradio endpoint.

## Interfaces

- CLI: `dota2tuned ingest`, `normalize`, `features`, `train-predictor`, `build-rag`, `make-sft`, `finetune`, `eval`, `serve`, `modal-deploy`, `modal-smoke`, `modal-train`, `modal-ask`.
- `.env` keys: `HF_TOKEN`, `HF_ORG`, `HF_SPACE_ID`, `HF_MODEL_REPO_ID`, `HF_DATASET_REPO_ID`, `STRATZ_TOKEN`, `OPENDOTA_API_KEY`, `STEAM_API_KEY`, `BASE_MODEL_ID`, `TRAINING_FLAVOR`, `SPACE_HARDWARE`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_APP_NAME`, `MODAL_ENABLED`, `MODAL_TRAIN_GPU`, `MODAL_TRAIN_TIMEOUT`, `MODAL_INFER_GPU`, `MODAL_INFER_TIMEOUT`, `MODAL_CACHE_VOLUME`, `MODAL_OUTPUT_VOLUME`, `DUCKDB_PATH`, `RAW_DATA_DIR`, `PARQUET_DIR`.
- Recommendation schema: `hero_id`, `hero_name`, `role`, `score`, `win_prob_delta`, `counter_lift`, `synergy_lift`, `sample_size`, `patch`, `scope`, `sources`, `confidence`, `caveats`.
- Gradio tabs: Draft Coach, Hero Meta, Tuned Model, Match Predictor, Builds, Draft Lab, Data Freshness.

## Pipeline

- Ingest pro tournaments first: OpenDota `/proMatches`, professional `/leagues`, `/leagues/{league_id}/matches`, then `/matches/{match_id}` and STRATZ `match(id)` enrichment.
- Initial dataset mix: `65%` pro/tournament, `25%` high-rank current-patch public, `10%` broad current-patch public for rare hero coverage.
- Budget defaults: OpenDota free day uses about `1900` pro/current league calls, `750` high-rank/public enrichment calls, `250` constants/retries; STRATZ default day uses `7000` pro enrich, `2000` high-rank/public enrich, `1000` retry/schema reserve.
- Prediction: train calibrated logistic baseline and LightGBM draft predictor; choose by temporal patch validation.
- Recommendations: rank heroes by predicted win-probability delta plus empirical counter/synergy lift; builds come from observed item/skill/talent timing stats, not LLM invention.
- RAG: index patch-change docs and stat cards with patch/scope filters before retrieval.
- Fine-tuning: default `Qwen/Qwen3-4B-Instruct-2507`; fallback `HuggingFaceTB/SmolLM3-3B`. Use explicit 4-bit QLoRA with TRL SFT on Modal `A100-80GB`, 1 epoch, structured answer examples, then push to Hub.
- Serving: HF Gradio Space loads compact predictor/RAG artifacts; Modal `ui` is a verified alternate Gradio endpoint, Modal `train_sft` handles GPU fine-tuning, and Modal `generate_answer` serves the fine-tuned adapter for explicit LLM responses.

## Current Execution Process

1. Configure local secrets in `.env`; never commit `.env`.
   - Required for live data/API smoke: `HF_TOKEN`, `STRATZ_TOKEN`, `OPENDOTA_API_KEY`, `STEAM_API_KEY`.
   - Required for Modal: `MODAL_ENABLED=1`, `MODAL_APP_NAME=dota2tuned`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
   - Modal GPU defaults: `MODAL_TRAIN_GPU=A100-80GB`, `MODAL_TRAIN_TIMEOUT=21600`, `MODAL_INFER_GPU=A10G`, `MODAL_INFER_TIMEOUT=900`, `MODAL_CACHE_VOLUME=dota2tuned-hf-cache`, `MODAL_OUTPUT_VOLUME=dota2tuned-outputs`.
2. Install local dependencies for Modal work.
   - `uv sync --extra modal --extra dev`
3. Validate live API configuration.
   - `uv run dota2tuned smoke --live`
4. Validate code before external deploys.
   - `uv run ruff check app.py src tests`
   - `uv run pytest -q`
   - `uv run python -c "from app import demo; print(type(demo).__name__, len(demo.blocks), len(demo.fns))"`
5. Deploy Modal functions.
   - `uv run dota2tuned modal-deploy`
   - Expected functions: `ui`, `remote_smoke`, `train_sft`, `generate_answer`.
   - Current Modal URL: `https://dracufeuer--dota2tuned-ui.modal.run`
6. Verify Modal runtime artifacts.
   - `uv run dota2tuned modal-smoke`
   - Expected: Gradio `Blocks`, 11 Parquet files, RAG index present, SFT examples present.
7. Train the QLoRA adapter on Modal.
   - `uv run dota2tuned modal-train`
   - Completed training call: `fc-01KV2GTE604BCRND3M6AH3GGT1`.
   - Output repo: `build-small-hackathon/dota2tuned-qwen3-4b-2507-lora`.
   - Expected files: `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, `training_args.bin`.
8. Verify fine-tuned adapter inference.
   - `uv run dota2tuned modal-ask "Suggest one mid hero against Phantom Assassin and Witch Doctor. Include one caveat."`
   - Target behavior: concise grounded answer; caveat weak evidence; no invented heroes/items.
9. Configure HF Space runtime variables/secrets for Modal-backed inference.
   - Variables: `MODAL_ENABLED=1`, `MODAL_APP_NAME=dota2tuned`.
   - Secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
   - Do not expose these values in logs or docs.
10. Upload the Space code/artifacts through the Hub API with explicit ignore patterns.
    - Include compact serving artifacts: `data/parquet/*.parquet`, `data/rag/tfidf.joblib`, `data/models/draft_predictor.joblib`, `data/models/sft_examples.jsonl`.
    - Exclude `.env`, caches, raw data, probe data, run folders, and DuckDB files.
    - Current HF Space URL: `https://build-small-hackathon-dota2tuned.hf.space`.
11. Wait for Space runtime status.
    - Use `HfApi.space_info(...).runtime.stage`.
    - Required terminal state: `RUNNING`.
12. Final endpoint checks.
    - HF Space HTTP 200 and Modal UI HTTP 200.
    - `uv run dota2tuned modal-smoke`.
    - `uv run dota2tuned modal-ask ...`.
    - `git status --short --branch` clean after commit/push.

## Timeline

All times are `America/Los_Angeles` / PDT unless noted.

- 2026-06-13 19:54:19: created the hackathon scaffold with `PLAN.md`, secret-safe env templates, package layout, API clients, pipeline commands, Gradio entrypoint, tests, and initial `.gitignore` rules.
- 2026-06-13 20:29:39: completed model-selection pass and set `Qwen/Qwen3-4B-Instruct-2507` as the default sub-32B base model, with `HuggingFaceTB/SmolLM3-3B` as fallback. Gemma-class options were considered but not selected because Qwen 4B gave the better balance for structured reasoning, tool-like answers, and compact QLoRA serving.
- 2026-06-13 20:51:29: added live smoke checks for Hugging Face, STRATZ, OpenDota, and Steam configuration without printing secrets.
- 2026-06-13 20:58:09: fixed Hugging Face Space package loading so `src/` imports resolve correctly in the hosted runtime.
- 2026-06-13 22:37:42: shipped compact data/model artifacts and A100 preflight wiring for GPU training readiness.
- 2026-06-13 23:08:39: separated Hugging Face job token handling from general Hub token handling. This path was later superseded by Modal after the available credit source was corrected.
- 2026-06-14 00:24:00: created and deployed the Modal app `dota2tuned` as the GPU execution backend.
- 2026-06-14 00:26-00:30: first Modal remote smoke surfaced missing runtime env inside Modal. Fixed by baking `MODAL_ENABLED=1` and `MODAL_APP_NAME=dota2tuned` into the Modal image environment while keeping credentials secret.
- 2026-06-14 00:30-00:31: redeployed Modal; `remote_smoke` passed with Gradio `Blocks`, 11 Parquet artifacts, RAG index, and SFT examples available remotely. Modal UI returned HTTP 200 at `https://dracufeuer--dota2tuned-ui.modal.run`.
- 2026-06-14 00:31: submitted an initial Modal training call, then cancelled it as stale after redeploy so training would run against the current image.
- 2026-06-14 00:32-00:34: submitted fresh Modal training call `fc-01KV2GTE604BCRND3M6AH3GGT1`; QLoRA training completed and pushed the adapter to `build-small-hackathon/dota2tuned-qwen3-4b-2507-lora`.
- 2026-06-14 00:34-00:35: verified the model repo contents, including `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, chat template, and `training_args.bin`.
- 2026-06-14 00:35:53: committed and pushed `c4dd25b` (`feat: add modal gpu path`) with the Modal deployment, training, and smoke-test path.
- 2026-06-14 00:36-00:45: added Modal adapter inference via `generate_answer`, local CLI access through `dota2tuned modal-ask`, and grounded generation rules for evidence-aware Dota 2 answers.
- 2026-06-14 00:46:02: committed and pushed `2cc4796` (`feat: serve adapter on modal`) with Modal inference serving.
- 2026-06-14 00:47-01:00: configured HF Space variables and secrets for Modal-backed inference without exposing values, uploaded the refreshed Space bundle, and verified the canonical Space at `https://build-small-hackathon-dota2tuned.hf.space`.
- 2026-06-14 01:00-01:12: added the Gradio `Tuned Model` tab, added the Space-side `modal` dependency, ran a 10-prompt Modal adapter eval, fixed one grounding issue around unsupported generic build advice, and retested Shadow Fiend and fake-hero prompts successfully.
- 2026-06-14 01:13: updated this plan with the execution timeline. Current required finish checks are lint, tests, Gradio import smoke, commit/push, Space upload, Space HTTP 200, Modal UI HTTP 200, `modal-smoke`, and `modal-ask`.
- 2026-06-14 01:21: checked the latest hackathon requirements again and added `SUBMISSION.md` with required links, final checklist, demo video script, social post draft, and verification commands.
- 2026-06-14 01:27: addressed the submission-readiness review by adding hero-name parsing, making Draft Lab demoable, adding tests, preparing a dataset card, declaring Apache-2.0 licensing, and aligning the Modal dependency version.
- 2026-06-14 02:36: continued demo-readiness hardening by adding role-aware recommendation filtering, Bayesian win-rate shrinkage for low samples, pre-game item timing labels, and a repeatable public Space API check script.

## Adapter Eval Notes

- Ran 10 Modal adapter prompts covering draft, countering Phantom Assassin, Crystal Maiden synergy, Shadow Fiend build timing, patch/meta caveats, prediction limits, fake hero refusal, support pick, anti-push response, and low-sample caveats.
- Mechanical pass: 10/10 calls returned `status=ok` with non-empty answers.
- Found and fixed one grounding weakness: generic Shadow Fiend build prompt produced unsupported item advice. The Modal inference system prompt now forbids inventing hero names, item names, scores, timings, patches, or match data and requires an uncertainty response when evidence is missing or generic.
- Targeted retest passed:
  - Generic Shadow Fiend build prompt now says no specific item timing is available without concrete evidence.
  - Fake hero `Banana King` is rejected as not a real Dota 2 hero.

## Test Plan

- Unit tests: API clients, auth headers, rate limiters, retry/cache keys, patch parser, normalizers, schema validation.
- Data checks: no duplicate match IDs, valid hero/item IDs, correct patch join by `start_time`, no abandoned/remake training matches, no future-patch leakage.
- Model evals: ROC-AUC, log loss, Brier/calibration for match prediction; Recall@K/NDCG@K for recommendations; citation support rate for generated answers.
- App checks: Gradio import/launch smoke, GPU availability logging, no secret leakage, deterministic fallback when model/backend unavailable.
- Modal checks: disabled-by-default local behavior, backend function smoke when `MODAL_ENABLED=1`, `ui` HTTP 200, bundled artifacts visible remotely, and GPU training call tracked by Modal function call id.

## Assumptions

- The team can create the required HF Space before the June 15, 2026 deadline.
- Modal credits are available for QLoRA training.
- Modal can host an alternate Gradio endpoint, but the Hugging Face Space remains the canonical hackathon submission link.
- STRATZ/OpenDota/Steam tokens will be supplied in `.env`; no tokens will be committed.
