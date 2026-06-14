import polars as pl

from dota2tuned.train_predictor import _build_matrix


def test_build_matrix_uses_radiant_rows_for_label():
    players = pl.DataFrame(
        [
            {"match_id": 1, "hero_id": 1, "is_radiant": True, "win": 1},
            {"match_id": 1, "hero_id": 2, "is_radiant": False, "win": 0},
            {"match_id": 2, "hero_id": 1, "is_radiant": True, "win": 0},
            {"match_id": 2, "hero_id": 2, "is_radiant": False, "win": 1},
        ]
    )

    x, y, hero_ids = _build_matrix(players)

    assert x.shape == (2, 4)
    assert hero_ids == [1, 2]
    assert y.tolist() == [1, 0]
