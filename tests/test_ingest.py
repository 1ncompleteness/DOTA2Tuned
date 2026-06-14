from pathlib import Path

import polars as pl

from dota2tuned.ingest import _targeted_match_query, _under_sampled_heroes


def test_under_sampled_heroes_uses_player_games_and_pro_pick(tmp_path: Path):
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    pl.DataFrame(
        [
            {"hero_id": 1, "hero_name": "Anti-Mage", "pro_pick": 12},
            {"hero_id": 2, "hero_name": "Axe", "pro_pick": 0},
            {"hero_id": 3, "hero_name": "Bane", "pro_pick": 0},
        ]
    ).write_parquet(parquet / "dim_hero.parquet")
    pl.DataFrame([{"hero_id": 2}, {"hero_id": 2}]).write_parquet(
        parquet / "fact_player_match.parquet"
    )

    gaps = _under_sampled_heroes(parquet, threshold=3, limit=10)

    assert gaps == [
        {"hero_id": 3, "hero_name": "Bane", "sample_size": 0, "needed": 3},
        {"hero_id": 2, "hero_name": "Axe", "sample_size": 2, "needed": 1},
    ]


def test_targeted_match_query_is_select_only_and_filters_quality():
    query = _targeted_match_query(103, limit=100, offset=200)

    assert query.lower().startswith("select")
    assert "from public_matches" in query
    assert "103 = any(m.radiant_team)" in query
    assert "103 = any(m.dire_team)" in query
    assert "m.duration >= 600" in query
    assert "m.radiant_win is not null" in query
    assert "limit 100" in query
    assert "offset 200" in query
