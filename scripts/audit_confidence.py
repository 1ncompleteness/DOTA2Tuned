from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput

ROLES = [None, "carry", "mid", "offlane", "soft support", "hard support"]


def _confidence(sample_size: int, threshold: int) -> str:
    if sample_size >= threshold:
        return "high"
    if sample_size >= 100:
        return "medium"
    return "low"


def _read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing parquet file: {path}")
    return pl.read_parquet(path)


def _hero_samples(parquet_dir: Path, threshold: int) -> tuple[dict[str, Any], pl.DataFrame]:
    heroes = _read_parquet(parquet_dir / "dim_hero.parquet")
    players = _read_parquet(parquet_dir / "fact_player_match.parquet")
    if "pro_pick" not in heroes.columns:
        heroes = heroes.with_columns(pl.lit(0).alias("pro_pick"))
    counts = players.group_by("hero_id").agg(pl.len().cast(pl.Int64).alias("player_games"))
    base = (
        heroes.select(["hero_id", "hero_name", "pro_pick"])
        .join(counts, on="hero_id", how="left")
        .with_columns(pl.col("player_games").fill_null(0))
        .with_columns(pl.max_horizontal("player_games", "pro_pick").alias("sample_size"))
        .with_columns(
            pl.col("sample_size")
            .map_elements(lambda value: _confidence(int(value), threshold), return_dtype=pl.String)
            .alias("confidence")
        )
    )
    summary = base.select(
        [
            pl.len().alias("heroes"),
            (pl.col("sample_size") >= threshold).sum().alias("high"),
            ((pl.col("sample_size") >= 100) & (pl.col("sample_size") < threshold))
            .sum()
            .alias("medium"),
            (pl.col("sample_size") < 100).sum().alias("low"),
            pl.col("sample_size").min().alias("min"),
            pl.col("sample_size").quantile(0.25).alias("p25"),
            pl.col("sample_size").median().alias("median"),
            pl.col("sample_size").quantile(0.75).alias("p75"),
            pl.col("sample_size").max().alias("max"),
        ]
    ).to_dicts()[0]
    summary["threshold"] = threshold
    summary["max_confidence"] = summary["low"] == 0 and summary["medium"] == 0
    return summary, base


def _recommendation_surfaces(parquet_dir: Path, threshold: int) -> list[dict[str, Any]]:
    recommender = DraftRecommender(parquet_dir)
    surfaces: list[tuple[str, DraftInput]] = [
        (role or "any", DraftInput(role=role)) for role in ROLES
    ]
    surfaces.append(
        (
            "demo_pa_wd_mid",
            DraftInput(enemy_heroes=[44, 30], role="mid", scope="pro", patch="current"),
        )
    )
    rows = []
    for label, draft in surfaces:
        recs = recommender.recommend(draft, limit=8)
        low = [rec for rec in recs if rec.sample_size < threshold]
        rows.append(
            {
                "surface": label,
                "all_high": len(low) == 0 and bool(recs),
                "min_sample": min((rec.sample_size for rec in recs), default=0),
                "recommendations": [
                    {
                        "hero": rec.hero_name,
                        "sample_size": rec.sample_size,
                        "confidence": rec.confidence,
                    }
                    for rec in recs
                ],
            }
        )
    return rows


def build_payload(parquet_dir: Path, threshold: int, lowest: int) -> dict[str, Any]:
    summary, base = _hero_samples(parquet_dir, threshold)
    low_heroes = (
        base.filter(pl.col("sample_size") < threshold)
        .sort("sample_size")
        .select(["hero_id", "hero_name", "player_games", "pro_pick", "sample_size", "confidence"])
        .head(lowest)
        .to_dicts()
    )
    surfaces = _recommendation_surfaces(parquet_dir, threshold)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "parquet_dir": str(parquet_dir),
        "hero_samples": summary,
        "lowest_sample_heroes": low_heroes,
        "recommendation_surfaces": surfaces,
    }


def print_report(payload: dict[str, Any]) -> None:
    hero = payload["hero_samples"]
    print(
        "hero confidence: "
        f"high={hero['high']} medium={hero['medium']} low={hero['low']} "
        f"min={hero['min']} median={hero['median']} max={hero['max']} "
        f"threshold={hero['threshold']}"
    )
    print(f"max_confidence={hero['max_confidence']}")
    print("lowest sample heroes:")
    for row in payload["lowest_sample_heroes"]:
        print(
            f"- {row['hero_name']}: sample={row['sample_size']} "
            f"players={row['player_games']} pro_pick={row['pro_pick']} "
            f"confidence={row['confidence']}"
        )
    print("recommendation surfaces:")
    for surface in payload["recommendation_surfaces"]:
        print(
            f"- {surface['surface']}: all_high={surface['all_high']} "
            f"min_sample={surface['min_sample']} "
            f"top={[item['hero'] for item in surface['recommendations'][:3]]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DOTA2Tuned recommendation confidence.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--threshold", type=int, default=500)
    parser.add_argument("--lowest", type=int, default=25)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-under-max",
        action="store_true",
        help="Exit 1 unless every hero sample is at or above the threshold.",
    )
    args = parser.parse_args()

    payload = build_payload(args.parquet_dir, args.threshold, args.lowest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_report(payload)
    if args.fail_under_max and not payload["hero_samples"]["max_confidence"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
