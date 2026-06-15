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
- Gradio left-sidebar views: Draft Coach, Hero Meta, Tuned Model, Match Predictor, Builds, Draft Lab, Data Freshness.

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
13. Confidence expansion gate.
    - The runtime confidence ceiling is `high`, defined in code as `sample_size >= 500`.
    - Audit current artifacts with `uv run python scripts/audit_confidence.py --threshold 500`.
    - Continue background expansion with `scripts/run_confidence_expansion.sh` after the active tmux ingest completes.
    - Default continuation policy: targets `12,000`, `17,000`, `22,000`, `27,000` match details and a `30,000` cap unless overridden through `DOTA2TUNED_CONFIDENCE_TARGET`, `DOTA2TUNED_CONFIDENCE_STEP`, or `DOTA2TUNED_CONFIDENCE_MAX_TARGET`.
    - Stop early when every hero base sample reaches `500`, when the source stops yielding new match details, or when the configured cap is reached. Rare heroes may remain below `500` even after large pro-match expansion because pick distribution is not uniform; those remain explicitly caveated.
14. Targeted rare-hero backfill.
    - Use `uv run dota2tuned targeted-ingest --threshold 500 --hero-limit 64 --matches-per-hero 700 --max-new-details 5000` to discover recent public match IDs for under-sampled heroes through OpenDota Explorer and append missing `/matches/{match_id}` details.
    - Use `scripts/run_targeted_confidence_backfill.sh` for the durable loop; each round runs targeted ingest, normalization, feature rebuild, predictor training, RAG rebuild, SFT generation, and confidence audit.
    - Retrain the Modal adapter only after the targeted fetch loop reaches high confidence or exhausts useful new match details, so the adapter is trained on the strongest available SFT snapshot.

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
- 2026-06-14 02:41: replaced the generic generated adapter README with a DOTA2Tuned-specific model card covering intended behavior, evidence discipline, usage, evaluation notes, limitations, and public links.
- 2026-06-14 02:57: added a final submission-readiness checker that verifies public URLs, Space runtime, Hub model/dataset cards, git cleanliness, and all public Gradio API endpoints while listing the human-only video/social/submission steps separately.
- 2026-06-14 03:29: started a durable tmux background data expansion toward 1,600 OpenDota match details, added checkpointed detail ingestion, moved recommendation confidence to normalized player-match samples, and upgraded the Gradio UI with searchable hero/item dropdowns, alias search, and icon previews.
- 2026-06-14 03:49: completed the background data expansion with 1,604 enriched OpenDota match details, 16,040 player-match rows, 38,347 draft pick/ban rows, 728,416 item-purchase rows, 23,080 hero-pair rows, 32,517 build-stat rows, 1,604 draft predictor samples, and 279 SFT examples. Medium-confidence recommendations now appear where normalized sample sizes exceed 100.
- 2026-06-14 03:50: started the next durable tmux expansion `dota2tuned_data_expand` toward 7,000 OpenDota match details with `--pro-matches 8000 --public-matches 500 --enrich-limit 7000 --patch-count 4 --league-limit 50`; the run continues in the background and checkpoints existing details every 25 new match fetches.
- 2026-06-14 04:06: audited the current 1,604-match parquet artifacts before the 7,000-match run finished: `0` high-confidence heroes, `67` medium, `60` low, median sample `104`, max sample `496`. Added repeatable confidence audit and continuation scripts so expansion can proceed after the active tmux job until the `500`-sample high-confidence gate is reached or the configured cap/source exhaustion stops it.
- 2026-06-14 05:37: completed the 7,000-target expansion with 7,004 enriched OpenDota match details, 70,040 player-match rows, 166,082 draft pick/ban rows, 2,661,012 item-purchase rows, 30,734 hero-pair rows, 49,327 build-stat rows, 489 SFT examples, and draft predictor metrics `roc_auc=0.5522`, `log_loss=0.7080`, `brier=0.2560`.
- 2026-06-14 06:19: confidence supervisor finished after attempting continuation through the 27,000 target. The available current source pool only grew to 8,562 enriched details, producing 85,619 player-match rows and confidence coverage of `74` high, `52` medium, `1` low, median sample `533`, max sample `2479`. Full max confidence was not reached because rare heroes, especially Elder Titan at sample `42`, remain under the `500` high-confidence gate; further improvement requires broadening match discovery beyond the current OpenDota pro/league source pool.
- 2026-06-14 15:13: added targeted rare-hero backfill based on OpenDota Explorer `public_matches` hero arrays. A capped smoke run targeted Elder Titan, discovered `10` candidate rows, fetched `5` new details, and raised raw OpenDota match details from `8,562` to `8,567` before launching the durable targeted loop.
- 2026-06-14 16:18: polished Gradio dropdown UX by replacing packed role text symbols with image-backed role badges, adding role/scope dropdown icon decoration, keeping hero dropdown icon decoration, and preserving the Steam CDN Dota logo in the topbar. The targeted tmux loop was still running separately at `11,946` raw OpenDota match details and `3,380` targeted candidate rows.
- 2026-06-14 16:36: removed all one-letter generated hero aliases from dropdown/search labels, kept curated conventional aliases such as `AM` and `QoP`, and replaced the top tab bar with a native left `gr.Sidebar` navigation so all former tab views share one consistent content layout.
- 2026-06-14 16:54: refined the Gradio UI to match current Dota 2 web styling from the official Dota React bundle: Radiance font, black/blue-gray panels, ember red actions, muted gold accents, and plain icon+text sidebar navigation. Moved the DOTA2Tuned logo/text into the sidebar brand section, removed topbar reinjection, made view content full-width, kept hero dropdown rows icon-decorated, and changed selected hero previews to image-only cards.
- 2026-06-14 17:37: incorporated the GitHub merge from `terrible-additions`, which filters Draft Coach hero dropdown choices across allies, enemies, and bans. Rechecked the current Dota home/heroes bundles and confirmed the homepage typography inherits `font-family: "Radiance","Noto Sans",sans-serif`; kept Radiance globally, narrowed red action-button styling to app content so Gradio utility controls stay neutral, shrank selected hero preview cards, and rendered hero dropdown roles as visible tag pills.
- 2026-06-14 17:40: committed and pushed the Dota typography/UI fixes to GitHub as `32f691c` with Codex coauthor, synced the same source/test/plan files to the Hugging Face Space as `3b4275d7`, restarted the Space, and verified the public HTML now serves Radiance globally with `.app-main button` scoped action styling plus visible dropdown role tags.
- 2026-06-14 17:46: reran the targeted confidence backfill supervisor. It exited without fetching more data because the current normalized artifacts already meet the `500` sample high-confidence gate for every hero: `127` high, `0` medium, `0` low, minimum sample `556`, median `1016`, max `3572`.
- 2026-06-14 18:04: switched all app text to the Dota Trajan Pro font assets, made sidebar nav labels uppercase, removed the style-only `gr.HTML` wrapper by passing CSS/JS through Gradio launch/mount paths, restored text-and-tag hero preview cards with remove buttons, hid selected hero tokens inside multiselect dropdowns so the control remains search-only, fixed chevron clicks by focusing the underlying dropdown input, and fixed the cross-browser dropdown scroll error by making injected option icons/tags non-interactive so Gradio receives events on the original option element.
- 2026-06-14 18:07: removed the sidebar brand subtitle and added client-side sidebar navigation synchronization with stable view IDs so page content switches on the first click while the Gradio backend visibility callback catches up.
- 2026-06-14 18:08: tightened role icon sizing inside hero-card tag pills so tag icons fit the allocated pill height instead of inheriting larger card image dimensions.
- 2026-06-14 18:30: constrained Gradio dropdown overlays to the trigger/viewport width, scoped listbox CSS to `.dota-dropdown`, hid horizontal overflow, and added a runtime clamp that repositions fixed dropdown menus on open, scroll, and resize so long hero labels no longer push menus outside the field.
- 2026-06-14 21:20: resolved the GitHub merge conflict in `src/dota2tuned/ui/gradio_app.py` by keeping the new Builds page while preserving the JS-only sidebar switching and dropdown clamp fixes. Refreshed direct dependency minimums and `uv.lock` to the latest resolved set, including Gradio `6.18.0`, FastAPI `0.137.0`, Transformers `5.12.0`, Torch `2.12.0`, Datasets `5.0.0`, and matching Modal/HF training dependency lists. Validation passed with `uv lock --check`, `uv run ruff check app.py src tests`, `uv run pytest -q`, and a Gradio app-construction smoke check.
- 2026-06-14 21:36: fixed the Hugging Face Space build after the dependency refresh by capping Pydantic to `>=2.12.5,<2.13.0`, the latest range compatible with the Space builder's injected `gradio[oauth,mcp]==6.18.0`. Revalidated with `uv lock --check`, `uv run ruff check app.py src tests`, `uv run pytest -q`, `uv pip compile --python-version 3.12 requirements.txt -` plus Space-injected requirements, and a Gradio app-construction smoke check. Next remote sync should upload code and tracked runtime artifacts, including `data/rag/tfidf.joblib`, so judges see the latest RAG artifacts in the Space.
- 2026-06-14 21:48: fixed the dropdown scrollbar exception by ignoring inner dropdown-menu scroll events in the client-side overlay clamp and allowing hero multiselects to accept Gradio's transient `[None]` payload before sanitizing it back to valid hero IDs. Added regression tests for transient null hero selections and the menu-scroll guard. Validation passed with `uv run ruff check app.py src tests`, `uv run pytest -q`, a Gradio app-construction smoke check, and an assertion that all hero multiselect dropdowns have `allow_custom_value=True`.
- 2026-06-14 21:53: corrected the dropdown scroll fix after the first patch caused Draft Coach dropdowns to re-render during scroll. Transient `[None]` events now return `gr.skip()` for choice filtering and hero previews, and the scroll guard ignores any `.options` or `[role="listbox"]` scroll target, including portaled Gradio dropdown menus. Revalidated with `uv run ruff check app.py src tests`, `uv run pytest -q`, and Gradio app construction.
- 2026-06-14 22:11: fixed the Draft Coach hero dropdown field layout after Gradio 6 placed the chevron wrapper outside the visible input area. Replaced the forced `100%` hero search width with a flex/min-width-safe dropdown layout, reserved right-side input padding for `.icon-wrap`, and made the secondary search wrappers transparent so Allied heroes, Enemy heroes, and Banned heroes no longer show a distinct nested text background.
- 2026-06-14 22:18: fixed selected hero-card readability by changing the direct hero portrait image to `object-fit: contain` in the existing 48x36 slot, narrowing the portrait selector so role icons are not affected, and allowing hero names to wrap instead of being ellipsized.
- 2026-06-14 22:25: fixed stale dropdown search rendering after confirming Gradio can reuse option DOM nodes while changing the underlying `aria-label`/text. The client decorator now stores the label it rendered, removes only DOTA2Tuned-injected icon/label children when a reused row changes, and observes option label/text mutations so visible search results match the actual clickable choice.
- 2026-06-14 22:59: added an early startup loader in `APP_HEAD` so the app shows the DOTA2Tuned brand and Dota symbol while Gradio mounts instead of briefly exposing bare elements. The Dota symbol pulses from white to the palette red `#ff6046` and back over a 4.2 second cycle, and the sidebar brand icon uses the same animation after the loader exits.
- 2026-06-14 23:15: removed the remaining Gradio blue primary backgrounds by routing primary/theme variables, block labels, selected radio/checkbox labels, and range accents through the same red action fill used by the Recommend button. This covers Allied heroes, Enemy heroes, Banned heroes, Role, Scope, and equivalent labeled inputs across all mounted app views.
- 2026-06-14 23:21: made selected hero-card portrait backgrounds transparent and smoothed the Dota symbol pulse for both the startup loader and sidebar brand by using matching filter functions across all keyframes, with intermediate 25%/75% stops inside the existing 4.2 second white-red-white cycle.
- 2026-06-14 23:39: hardened the startup loader against first-paint flashes by hiding `.gradio-container` by default until `html.d2-ready`, adding a CSS-only `body::before/after` branded fallback before the JS loader div can mount, and revealing the app only after the Gradio load hook waits for fonts, currently-mounted images, and paint frames, with a timeout fallback.
- 2026-06-14 23:44: removed the redundant Builds page selected-hero preview `gr.HTML` component and its change handler, eliminating the extra generated `html-container` wrapper while keeping the Builds dropdown and actual builds output intact.
- 2026-06-15 01:37: aligned the Builds page `.build-hero-header` portrait with the conventional hero-card icon treatment: 48x36 landscape slot, `object-fit: contain`, transparent background, and no square crop. Started tmux job `dota2tuned_pro_patch_fetch` to continue pro/tournament-focused OpenDota ingestion from 15,144 existing match details using `--pro-matches 50000 --public-matches 0 --enrich-limit 25000 --league-limit 250`, then normalize, rebuild features, retrain the draft predictor, rebuild RAG, regenerate SFT, and run a confidence audit. The live official patch context is `7.41d` from June 4, 2026; OpenDota metadata still reports the major patch as `7.41`.
- 2026-06-15 01:48: fetched the complete official Valve patch-note family for the recent 7.41 line: `7.41`, `7.41a`, `7.41b`, `7.41c`, and `7.41d`; `7.41e` returned no rows. Updated patch ingestion to expand major versions through lettered suffixes `a` through `e`, refreshed `doc_patch_change.parquet`, rebuilt RAG, regenerated SFT examples, and validated 2,333 total 7.41-family patch-change rows. Also preserved the user change that forces `GRADIO_SSR_MODE=False` before Gradio import in `app.py`.
- 2026-06-15 02:10: doubled the Builds page `.build-hero-header` hero icon from 48x36 to 96x72 while preserving the conventional landscape shape, transparent background, and `object-fit: contain`. Confirmed `dota2tuned_pro_patch_fetch` finished successfully with 15,989 OpenDota match details, 9,202 pro match summaries, 9,202 league/tournament match rows, 0 public rows, 150 STRATZ details, 159,888 normalized player-match rows, refreshed predictor/RAG/SFT artifacts, and confidence audit `high=127`, `medium=0`, `low=0`, `max_confidence=True`.
- 2026-06-15 02:40: investigated the HF Space startup log noise against Gradio 6.18.0 and upstream Gradio issue `#12699` (`ValueError: Invalid file descriptor: -1` on Spaces). Moved the targeted unraisable-hook filter into a stdlib-only startup path, added root `sitecustomize.py` so it installs before Gradio imports, moved CLI UI imports behind `serve`, added conservative cleanup of any idle main-thread event loop before Gradio/Uvicorn launch, and added a targeted `asyncio.BaseEventLoop.__del__` guard that suppresses only the invalid-fd destructor error while preserving unrelated exceptions.
- 2026-06-15 02:58: promoted the tuned model to the first/default sidebar view and rebuilt it as a GPT/Claude-style assistant landing surface: centered prompt composer, sample prompt chips, automatic local RAG evidence retrieval, Modal fine-tuned model call, rendered answer text, detected hero/item icon chips, linked source cards for Valve patch notes/OpenDota hero stat cards, and raw evidence in a collapsed panel.
- 2026-06-15 03:07: removed the duplicate landing-page DOTA2Tuned heading block from the assistant view while keeping the sidebar brand and tuned-model composer/result surface intact.
- 2026-06-15 03:12: aligned the assistant view with the rest of the app pages by dropping the centered hero-style layout and making the tuned-model surface full-width, top-aligned, compactly spaced, and flush with the existing app view grid.
- 2026-06-15 03:20: centered the startup-loader radial glow on the combined Dota icon and DOTA2Tuned text mark in both the critical first-paint fallback and hydrated loader, then changed the glow from inner Dota red `#ff6046` to outer Dota gold `#d9b166`.
- 2026-06-15 03:42: increased the centered startup-loader radial glow by 30%, from `22rem` to `28.6rem`, while keeping the red-to-gold gradient centered on the brand mark in both loader paths.
- 2026-06-15 04:02: diagnosed the Safari dropdown layering bug as a Gradio/Svelte stacking-chain issue: the fixed dropdown menu was nested under a `.dota-dropdown` block with `overflow: hidden` and a parent `.form` with `overflow: auto hidden`, so Safari could clip or paint the leftmost menu under later rows. Promoted the open dropdown plus its `.form`/`.row`/`.app-view` ancestors with temporary classes, forced the menu layer to fixed `z-index: 10000`, and kept ancestor overflow visible only while a menu is open.
- 2026-06-15 04:18: added the official Dota 2 homepage montage video as the startup-loader background, using the Steam WebM/MP4 sources with muted inline autoplay and keeping the red-to-gold radial glow plus dark vignette as an overlay/fallback.
- 2026-06-15 04:24: fixed the loader video visibility by changing the hydrated loader overlay from an opaque dark linear gradient to translucent RGBA layers and increasing the video layer opacity/brightness, so the montage remains visible behind the brand mark.
- 2026-06-15 04:34: enabled Gradio `share=True` in both Space and CLI launch paths, added a narrow startup warning filter for Gradio's deprecated Starlette 422 constant, and added root `agents.md` with Space/API instructions for coding agents.
- 2026-06-15 04:50: fixed the HF Space startup log regression from `share=True`: Gradio sharing is now enabled only for local runs, while HF Space containers use `share=False` and `quiet=True` because Spaces are already public and Gradio does not support share tunnels there.
- 2026-06-15 04:59: raised loader asset priority by preloading Trajan font weights, the Dota logo, and the MP4 montage in `APP_HEAD`; declared critical Trajan font faces before the loader CSS; added eager/fetch-priority hints to the logo/video elements; and explicitly warmed loader fonts plus the video in startup JS.
- 2026-06-15 05:07: fixed the mobile responsiveness breakpoint where Gradio expanded the open sidebar to the full viewport width. Below `820px`, the sidebar now becomes a compact sticky top navigation bar with horizontal scrolling tabs, while an unscoped head override removes Gradio's outer mobile padding so the active page content keeps the full mobile viewport width below it.
- 2026-06-15 05:12: standardized scrollbars across the app to use slim dark-track styling with Dota palette thumbs, and hardened the startup loader video for Safari by using the MP4 as the direct source plus muted inline autoplay attributes/properties (`defaultMuted`, `playsinline`, and `webkit-playsinline`) before playback.
- 2026-06-15 05:16: moved the real startup-loader video into middleware-injected initial body HTML so Safari sees a visible muted inline `<video>` during document parse instead of waiting for Gradio's delayed `head=` injection. The app-head script now primes an existing critical loader and retries playback on `loadeddata`, `canplay`, and visibility changes while recording any play-block reason on the video dataset.
- 2026-06-15 05:26: added an explicit Dota-styled startup-loader play button for Safari's autoplay safeguard path. The button is centered below the DOTA2Tuned icon/text, sits above all video and gradient layers, and invokes the same video prime/play logic in both the critical initial-body loader and the hydrated loader fallback.
- 2026-06-15 05:34: researched Safari's native video start overlay and confirmed it is browser-owned, with older WebKit pseudo-elements only a best-effort positioning path because newer Safari/iOS controls can live in closed UA shadow DOM. Removed the custom loader play button completely, restored pointer events on the loader video so Safari's native play affordance is active, and moved the WebKit `::-webkit-media-controls-start-playback-button` down with a scoped transform to keep it below the DOTA2Tuned icon/text.
- 2026-06-15 05:51: removed the red/gold circular radial glow from the startup loader now that the Steam montage video is the visual background. Kept the dark linear video overlays/vignette so the DOTA2Tuned mark remains legible.

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
