from __future__ import annotations

from pathlib import Path

import polars as pl

from dota2tuned.schemas import DraftInput, Recommendation
from dota2tuned.storage import read_parquet


def _confidence(sample_size: int) -> str:
    if sample_size >= 500:
        return "high"
    if sample_size >= 100:
        return "medium"
    return "low"


class DraftRecommender:
    def __init__(self, parquet_dir: Path) -> None:
        self.parquet_dir = parquet_dir
        self.heroes = read_parquet(parquet_dir / "dim_hero.parquet")
        self.pairs = read_parquet(parquet_dir / "fact_hero_pair_stats.parquet")

    def ready(self) -> bool:
        return not self.heroes.is_empty()

    def recommend(self, draft: DraftInput, *, limit: int = 8) -> list[Recommendation]:
        if self.heroes.is_empty():
            return []

        candidates = self.heroes
        excluded = set(draft.allied_heroes + draft.enemy_heroes + draft.banned_heroes)
        if excluded:
            candidates = candidates.filter(~pl.col("hero_id").is_in(list(excluded)))

        rows = []
        for row in candidates.iter_rows(named=True):
            hero_id = int(row["hero_id"])
            pro_pick = int(row.get("pro_pick") or 0)
            pro_win_rate = row.get("pro_win_rate")
            base_score = float(pro_win_rate or 0.5)
            synergy_lift = self._pair_lift(hero_id, draft.allied_heroes, "ally")
            counter_lift = self._pair_lift(hero_id, draft.enemy_heroes, "enemy")
            score = base_score + synergy_lift + counter_lift
            caveats = []
            if pro_pick < 100:
                caveats.append("low sample size")
            rows.append(
                Recommendation(
                    hero_id=hero_id,
                    hero_name=row.get("hero_name") or f"Hero {hero_id}",
                    role=draft.role,
                    score=round(score, 4),
                    win_prob_delta=round(score - 0.5, 4),
                    counter_lift=round(counter_lift, 4),
                    synergy_lift=round(synergy_lift, 4),
                    sample_size=pro_pick,
                    patch=draft.patch,
                    scope=draft.scope,
                    sources=["OpenDota heroStats", "normalized pair stats"],
                    confidence=_confidence(pro_pick),
                    caveats=caveats,
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:limit]

    def _pair_lift(self, hero_id: int, others: list[int], relation: str) -> float:
        if self.pairs.is_empty() or not others:
            return 0.0
        subset = self.pairs.filter(
            (pl.col("hero_id") == hero_id)
            & (pl.col("other_hero_id").is_in(others))
            & (pl.col("relation") == relation)
        )
        if subset.is_empty():
            return 0.0
        avg = subset.select(pl.col("win_rate").mean()).item()
        return float(avg - 0.5) * 0.25 if avg is not None else 0.0
