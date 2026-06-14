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
                "roles": "Carry,Escape,Nuker",
                "pro_pick": 1000,
                "pro_win": 520,
                "pro_win_rate": 0.52,
            },
            {
                "hero_id": 2,
                "hero_name": "Axe",
                "roles": "Initiator,Durable,Disabler",
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


def test_mid_recommendations_filter_support_first_heroes(tmp_path: Path):
    write_parquet(
        tmp_path / "dim_hero.parquet",
        [
            {
                "hero_id": 13,
                "hero_name": "Puck",
                "roles": "Initiator,Disabler,Escape,Nuker",
                "pro_pick": 12,
                "pro_win": 4,
                "pro_win_rate": 0.3333,
            },
            {
                "hero_id": 91,
                "hero_name": "Io",
                "roles": "Support,Escape,Nuker",
                "pro_pick": 4,
                "pro_win": 4,
                "pro_win_rate": 1.0,
            },
        ],
    )
    write_parquet(tmp_path / "fact_hero_pair_stats.parquet", [])

    recs = DraftRecommender(tmp_path).recommend(DraftInput(role="mid"), limit=5)

    assert [rec.hero_name for rec in recs] == ["Puck"]


def test_low_sample_win_rates_are_shrunk(tmp_path: Path):
    write_parquet(
        tmp_path / "dim_hero.parquet",
        [
            {
                "hero_id": 1,
                "hero_name": "Tiny Sample",
                "roles": "Carry",
                "pro_pick": 1,
                "pro_win": 1,
                "pro_win_rate": 1.0,
            },
            {
                "hero_id": 2,
                "hero_name": "Large Sample",
                "roles": "Carry",
                "pro_pick": 200,
                "pro_win": 120,
                "pro_win_rate": 0.6,
            },
        ],
    )
    write_parquet(tmp_path / "fact_hero_pair_stats.parquet", [])

    recs = DraftRecommender(tmp_path).recommend(DraftInput(role="carry"), limit=2)

    assert recs[0].hero_name == "Large Sample"


def test_recommendations_use_player_match_samples_for_confidence(tmp_path: Path):
    write_parquet(
        tmp_path / "dim_hero.parquet",
        [
            {
                "hero_id": 1,
                "hero_name": "Bigger Sample Hero",
                "roles": "Carry",
                "pro_pick": 3,
                "pro_win": 3,
                "pro_win_rate": 1.0,
            }
        ],
    )
    write_parquet(tmp_path / "fact_hero_pair_stats.parquet", [])
    write_parquet(
        tmp_path / "fact_player_match.parquet",
        [
            {
                "match_id": match_id,
                "hero_id": 1,
                "is_radiant": True,
                "win": 1 if match_id <= 120 else 0,
            }
            for match_id in range(1, 151)
        ],
    )

    recs = DraftRecommender(tmp_path).recommend(DraftInput(role="carry"), limit=1)

    assert recs[0].sample_size == 150
    assert recs[0].confidence == "medium"
    assert "normalized player matches" in recs[0].sources
