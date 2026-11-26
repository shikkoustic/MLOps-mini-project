"""Stage 4 monitoring: PSI drift check and per-group MAE on later.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "pm25_today",
    "wind_kmph",
    "humidity_pct",
    "temp_c",
    "traffic_index",
    "stubble_fires",
]


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a reference (train) and a new
    (later) distribution, using `bins` quantile buckets cut on the reference
    distribution (as in lab P13)."""
    expected = expected.dropna()
    actual = actual.dropna()
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected.quantile(quantiles).to_numpy())
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(expected, breakpoints).value_counts(sort=False)
    actual_counts = pd.cut(actual, breakpoints).value_counts(sort=False)

    expected_pct = (expected_counts / len(expected)).reindex(expected_counts.index, fill_value=0)
    actual_pct = (actual_counts / len(actual)).reindex(expected_counts.index, fill_value=0)

    eps = 1e-4
    expected_pct = expected_pct.clip(lower=eps)
    actual_pct = actual_pct.clip(lower=eps)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def psi_report(train_clean: pd.DataFrame, later: pd.DataFrame, bins: int = 10) -> dict:
    psi = {col: compute_psi(train_clean[col], later[col], bins=bins) for col in NUMERIC_COLS}
    most_drifted = max(psi, key=psi.get)
    return {"psi": psi, "most_drifted_feature": most_drifted}


def mae_by_group(model, later: pd.DataFrame, target_column: str = "pm25_next_day",
                  group_column: str = "station_zone") -> dict:
    from sklearn.metrics import mean_absolute_error

    feature_columns = [c for c in later.columns if c not in (target_column, "record_id", "date")]
    results = {}
    for group_value, rows in later.groupby(group_column):
        preds = model.predict(rows[feature_columns])
        results[group_value] = float(mean_absolute_error(rows[target_column], preds))
    worst = max(results, key=results.get)
    return {"mae_by_group": results, "worst_served_group": worst}
