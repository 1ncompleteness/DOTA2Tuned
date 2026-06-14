from __future__ import annotations

from pathlib import Path

from dota2tuned.storage import read_parquet, refresh_views

FEATURE_TABLES = [
    "fact_match",
    "fact_player_match",
    "fact_draft_pickban",
    "fact_item_purchase",
    "fact_hero_pair_stats",
    "fact_hero_build_stats",
]


def build_feature_summary(parquet_dir: Path, duckdb_path: Path) -> dict[str, int | str]:
    refresh_views(duckdb_path, parquet_dir)
    summary: dict[str, int | str] = {"duckdb_path": str(duckdb_path)}
    for table in FEATURE_TABLES:
        frame = read_parquet(parquet_dir / f"{table}.parquet")
        summary[table] = frame.height
    return summary
