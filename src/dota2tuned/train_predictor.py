from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from dota2tuned.storage import read_parquet


def _build_matrix(players: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if players.is_empty():
        return np.empty((0, 0)), np.empty((0,)), []
    hero_ids = sorted(int(x) for x in players["hero_id"].drop_nulls().unique().to_list())
    hero_index = {hero_id: i for i, hero_id in enumerate(hero_ids)}
    match_ids = sorted(int(x) for x in players["match_id"].unique().to_list())
    by_match = {mid: players.filter(pl.col("match_id") == mid) for mid in match_ids}
    rows_x = []
    rows_y = []
    for match_id in match_ids:
        rows = by_match[match_id]
        radiant = rows.filter(pl.col("is_radiant"))
        if radiant.is_empty():
            continue
        label = radiant.select(pl.col("win").drop_nulls().max()).item()
        if label is None:
            continue
        features = np.zeros(len(hero_ids) * 2, dtype=np.float32)
        for row in rows.iter_rows(named=True):
            hero_id = int(row["hero_id"])
            offset = 0 if row["is_radiant"] else len(hero_ids)
            features[offset + hero_index[hero_id]] = 1.0
        rows_x.append(features)
        rows_y.append(int(label))
    if not rows_x:
        return np.empty((0, len(hero_ids) * 2)), np.empty((0,)), hero_ids
    x = np.vstack(rows_x)
    y = np.asarray(rows_y, dtype=np.int64)
    return x, y, hero_ids


def train_predictor(parquet_dir: Path, model_dir: Path) -> dict[str, float | int | str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    players = read_parquet(parquet_dir / "fact_player_match.parquet")
    x, y, hero_ids = _build_matrix(players)
    if len(y) < 20 or len(set(y.tolist())) < 2:
        raise RuntimeError(
            "Need at least 20 labeled matches with both outcomes to train predictor."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    metrics = {
        "samples": int(len(y)),
        "heroes": int(len(hero_ids)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "log_loss": float(log_loss(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "model_path": str(model_dir / "draft_predictor.joblib"),
    }
    joblib.dump(
        {"model": model, "hero_ids": hero_ids, "metrics": metrics},
        model_dir / "draft_predictor.joblib",
    )
    return metrics


def predict_draft_win(
    model_dir: Path, radiant_heroes: list[int], dire_heroes: list[int]
) -> dict[str, object]:
    payload_path = model_dir / "draft_predictor.joblib"
    if not payload_path.exists():
        return {"status": "missing", "message": "No draft predictor has been trained yet."}
    payload = joblib.load(payload_path)
    hero_ids = [int(hero_id) for hero_id in payload["hero_ids"]]
    hero_index = {hero_id: i for i, hero_id in enumerate(hero_ids)}
    features = np.zeros(len(hero_ids) * 2, dtype=np.float32)
    ignored = []
    for hero_id in radiant_heroes:
        if hero_id in hero_index:
            features[hero_index[hero_id]] = 1.0
        else:
            ignored.append(hero_id)
    for hero_id in dire_heroes:
        if hero_id in hero_index:
            features[len(hero_ids) + hero_index[hero_id]] = 1.0
        else:
            ignored.append(hero_id)
    probability = float(payload["model"].predict_proba([features])[0, 1])
    return {
        "status": "ok",
        "radiant_win_probability": round(probability, 4),
        "dire_win_probability": round(1.0 - probability, 4),
        "ignored_hero_ids": ignored,
        "samples": payload.get("metrics", {}).get("samples"),
    }
