---
license: apache-2.0
task_categories:
- text-generation
- tabular-classification
language:
- en
tags:
- dota-2
- recommendation
- gradio
- qlora
- open-data
---

# DOTA2Tuned Data

This dataset supports the DOTA2Tuned Hugging Face Build Small Hackathon app. It contains compact derived artifacts for Dota 2 draft recommendations, hero meta lookup, build timing summaries, match prediction, retrieval, and supervised fine-tuning examples.

## Contents

- `sft_examples.jsonl`: instruction examples generated from normalized Dota 2 recommendations, patch/stat cards, and app behaviors.
- Compact Parquet artifacts used by the Space:
  - `dim_hero`
  - `dim_item`
  - `dim_league`
  - `dim_patch`
  - `fact_match`
  - `fact_player_match`
  - `fact_draft_pickban`
  - `fact_item_purchase`
  - `fact_hero_pair_stats`
  - `fact_hero_build_stats`
  - `doc_patch_change`

The submitted Space also includes a local TF-IDF retrieval index and a draft predictor artifact. Raw API payloads are intentionally not published in the Space bundle.

## Sources

- OpenDota API for public Dota 2 metadata, hero statistics, pro match references, league data, and match details.
- STRATZ API for optional match enrichment where configured.
- Steam Web API for configured Steam-backed checks.
- Valve / Dota 2 patch and constants feeds where available through the ingestion pipeline.

## Processing

The pipeline stores raw JSONL locally, normalizes it into Parquet tables, builds hero pair and build timing features, trains a draft predictor, builds retrieval documents, and emits SFT examples for the published LoRA adapter.

The app treats deterministic stats and predictors as the source of truth. The fine-tuned model explains retrieved evidence and should caveat weak or missing data.

## Intended Use

- Dota 2 draft recommendation demos.
- Patch-aware hero meta lookup.
- Counter, synergy, and build timing summaries.
- Grounded small-model answer generation examples.
- Hackathon reproducibility for the DOTA2Tuned Space.

## Limitations

- Public and pro match samples may be incomplete, sparse, or patch-skewed.
- Some hero/item statistics can have low sample sizes.
- Match prediction is a draft-only approximation and does not account for player skill, lane execution, itemization, or in-game events.
- The model must not invent heroes, items, scores, patch changes, or match data when evidence is absent.
- This is not affiliated with Valve, OpenDota, STRATZ, or Steam.

## Privacy and Tokens

No API tokens are included. Local `.env` files, raw API payloads, probe data, run directories, and DuckDB files are ignored by git and excluded from Space uploads.
