from pathlib import Path

from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput, Recommendation
from dota2tuned.storage import write_parquet


def test_recommendation_schema_and_exclusions(tmp_path: Path):
    write_parquet(
        tmp_path / "dim_hero.parquet",
        [
            {
                "hero_id": 1,
                "hero_name": "Anti-Mage",
                "pro_pick": 1000,
                "pro_win": 520,
                "pro_win_rate": 0.52,
            },
            {
                "hero_id": 2,
                "hero_name": "Axe",
                "pro_pick": 200,
                "pro_win": 90,
                "pro_win_rate": 0.45,
            },
        ],
    )
    write_parquet(tmp_path / "fact_hero_pair_stats.parquet", [])
    recommender = DraftRecommender(tmp_path)
    recs = recommender.recommend(DraftInput(banned_heroes=[1]), limit=5)
    assert len(recs) == 1
    assert isinstance(recs[0], Recommendation)
    assert recs[0].hero_id == 2
