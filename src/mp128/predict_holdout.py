"""Stage 4: predict pm25_next_day for every row of holdout_features.csv using
the current @champion model, as required for final submission."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from mp128.train import load_params
except ImportError:
    from src.mp128.train import load_params

ROOT = Path(__file__).resolve().parents[2]


def predict_holdout() -> Path:
    config = load_params()
    holdout_path = ROOT / config["data"]["holdout_path"]
    holdout = pd.read_csv(holdout_path)

    model = joblib.load(ROOT / "models" / "model.joblib")
    feature_columns = [c for c in holdout.columns if c not in ("record_id", "date")]
    predictions = model.predict(holdout[feature_columns])

    out = pd.DataFrame({"record_id": holdout["record_id"], "prediction": predictions})
    out_path = ROOT / "predictions" / "holdout_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = predict_holdout()
    print(f"Wrote {path}")
