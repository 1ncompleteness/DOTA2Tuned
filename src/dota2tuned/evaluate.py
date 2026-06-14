from __future__ import annotations

from pathlib import Path

import joblib


def load_predictor_metrics(model_dir: Path) -> dict[str, object]:
    path = model_dir / "draft_predictor.joblib"
    if not path.exists():
        return {"status": "missing", "message": "No draft predictor has been trained yet."}
    payload = joblib.load(path)
    return {"status": "ok", **payload.get("metrics", {})}
