# Repository Guidelines

## Project Structure & Module Organization

DOTA2Tuned is a Python 3.12 Gradio app and data pipeline for Dota 2 draft assistance. Core package code lives in `src/dota2tuned/`: API clients, ingestion, normalization, recommendations, RAG, SFT generation, Modal backend, and UI. The Hugging Face Space entrypoint is `app.py`. Tests live in `tests/`. Operational scripts live in `scripts/`. Runtime artifacts are under `data/parquet`, `data/rag`, and `data/models`; raw API outputs under `data/raw` are local-only. Planning and submission docs are in `PLAN.md`, `MODEL_SELECTION.md`, `MODEL_CARD.md`, `DATASET_CARD.md`, and `SUBMISSION.md`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install project and dev dependencies.
- `uv run dota2tuned health`: validate local configuration loading.
- `uv run dota2tuned smoke --live`: run small live checks against configured APIs.
- `uv run dota2tuned ingest --pro-matches 100 --public-matches 100 --enrich-limit 20`: fetch a small sample.
- `uv run dota2tuned normalize && uv run dota2tuned features`: rebuild local Parquet and feature views.
- `uv run dota2tuned build-rag`: rebuild retrieval artifacts.
- `uv run dota2tuned serve`: launch the Gradio app locally.
- `uv run ruff check app.py src tests`: lint.
- `uv run pytest -q`: run tests.

## Coding Style & Naming Conventions

Use 4-space indentation and Python type hints where practical. Keep modules focused and prefer existing helpers over new abstractions. Ruff is the source of style truth, with line length `100` and lint families `E`, `F`, `I`, `B`, `UP`, and `SIM`. Use `snake_case` for functions, variables, and CLI command internals; use `PascalCase` for classes.

## Testing Guidelines

Tests use `pytest` and should be named `test_*.py`. Add focused tests for parser, normalization, recommendation, UI helper, and API-client behavior when changing those surfaces. For UI changes, include a lightweight `build_app()` or helper assertion rather than relying only on manual browser checks.

## Commit & Pull Request Guidelines

Recent history uses concise conventional commits, for example `fix(ui): stabilize draft controls` and `docs: record UI sync and confidence`. Keep commits scoped and avoid committing secrets or unrelated generated artifacts. Pull requests should include a short description, validation commands run, screenshots for UI changes, and links to related issues or hackathon deliverables.

## Security & Configuration Tips

Copy `.env.example` to `.env` and keep tokens local. Never print or commit `HF_TOKEN`, `STRATZ_TOKEN`, `OPENDOTA_API_KEY`, `STEAM_API_KEY`, or Modal credentials. Prefer small smoke ingests before large API-consuming runs.
