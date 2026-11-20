"""Checks for the seven kinds of data problem described in STUDENT_HANDBOOK.md.

Each function inspects data/raw/train.csv (and sometimes later.csv /
holdout_features.csv for comparison) and returns a plain dict that maps
directly onto the corresponding block of reports/findings.yaml.
"""
from __future__ import annotations

import pandas as pd

NUMERIC_COLS = [
    "pm25_today",
    "wind_kmph",
    "humidity_pct",
    "temp_c",
    "traffic_index",
    "stubble_fires",
]
TEXT_COLS = ["day_type", "station_zone"]


def load_raw(path: str = "data/raw/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """A column only filled in after the event: missing from holdout and
    almost perfectly correlated with the target."""
    holdout_cols = set(holdout.columns)
    candidates = [c for c in train.columns if c not in holdout_cols and c != "pm25_next_day"]
    best_col, best_corr = None, 0.0
    for col in candidates:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = train[col].corr(train["pm25_next_day"])
            if pd.notna(corr) and abs(corr) > abs(best_corr):
                best_col, best_corr = col, corr
    present = best_col is not None and abs(best_corr) > 0.95
    return {
        "present": bool(present),
        "column": best_col if present else None,
        "_debug_corr": best_corr,
        "_debug_candidates": candidates,
    }


def check_unit_mixup(train: pd.DataFrame) -> dict:
    """One group (station_zone) recorded one numeric column in another unit.

    Uses the median (robust to outliers) of each group vs. the median of
    every OTHER group, so one contaminated group can't drag down its own
    comparison baseline.
    """
    common_factors = [10, 12, 60, 100, 1000]
    best = None
    for col in NUMERIC_COLS:
        for zone in train["station_zone"].unique():
            in_group = train.loc[train["station_zone"] == zone, col]
            rest = train.loc[train["station_zone"] != zone, col]
            group_med, rest_med = in_group.median(), rest.median()
            if rest_med == 0 or pd.isna(group_med) or pd.isna(rest_med):
                continue
            ratio = group_med / rest_med
            if ratio > 5 or ratio < 0.2:
                factor = min(common_factors, key=lambda f: abs(ratio - f) if ratio > 1 else abs((1 / ratio) - f))
                deviation = max(ratio, 1 / ratio)
                if best is None or deviation > best["_debug_deviation"]:
                    best = {
                        "present": True,
                        "column": col,
                        "group_value": zone,
                        "factor": factor,
                        "_debug_ratio": ratio,
                        "_debug_deviation": deviation,
                        "_debug_group_median": group_med,
                        "_debug_rest_median": rest_med,
                    }
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    return best


def check_sentinel_value(train: pd.DataFrame) -> dict:
    """A special number such as -1, -999, 999, 9999 typed instead of blank."""
    sentinels = [-999, -1, 999, 9999, -9999]
    best = None
    for col in NUMERIC_COLS:
        counts = train[col].value_counts()
        for s in sentinels:
            if s in counts.index and counts[s] >= 3:
                if best is None or counts[s] > best["count"]:
                    best = {"present": True, "column": col, "value": float(s), "count": int(counts[s])}
    if best is None:
        return {"present": False, "column": None, "value": None, "count": None}
    return best


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    """Same record appearing twice. Count EXTRA copies (rows to delete)."""
    dup_mask = train.duplicated(subset=[c for c in train.columns if c != "record_id"], keep="first")
    extra = int(dup_mask.sum())
    return {"present": extra > 0, "count": extra if extra > 0 else None}


def check_stuck_sensor(train: pd.DataFrame, min_run: int = 10) -> dict:
    """A column holding the same value for a long run of consecutive days."""
    df = train.sort_values("date").reset_index(drop=True)
    best = None
    for col in NUMERIC_COLS:
        vals = df[col].values
        dates = df["date"].values
        run_start = 0
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] != vals[run_start]:
                run_len = i - run_start
                if run_len >= min_run:
                    if best is None or run_len > best["_debug_run_len"]:
                        best = {
                            "present": True,
                            "column": col,
                            "start_date": str(dates[run_start]),
                            "end_date": str(dates[i - 1]),
                            "_debug_run_len": run_len,
                        }
                run_start = i
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    return best


def check_target_typos(train: pd.DataFrame) -> dict:
    """A few targets typed with an extra zero (10x too big): an IQR-based
    upper fence isolates the small group that sits far above the rest with a
    big empty gap below it, and dividing each by 10 puts it back in the
    normal range (confirming it really is an extra-zero typo, not a real
    extreme reading)."""
    target = train["pm25_next_day"]
    q1, q3 = target.quantile(0.25), target.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return {"present": False, "count": None}
    upper_fence = q3 + 3 * iqr
    outliers = target[target > upper_fence]
    if outliers.empty:
        return {"present": False, "count": None}
    normal_max = target[target <= upper_fence].max()
    confirmed = outliers[(outliers / 10) <= normal_max * 1.1]
    present = len(confirmed) > 0
    return {
        "present": bool(present),
        "count": int(len(confirmed)) if present else None,
        "_debug_upper_fence": upper_fence,
        "_debug_outlier_values": sorted(outliers.tolist()),
    }


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame | None = None) -> dict:
    """One text column written with inconsistent capitalisation/spacing/
    punctuation for the same category."""
    best = None
    for col in TEXT_COLS:
        raw_levels = train[col].dropna().unique()
        normalised = {str(v).strip().lower().rstrip(".") for v in raw_levels}
        if len(raw_levels) > len(normalised):
            true_levels = len(normalised)
            if later is not None and col in later.columns:
                true_levels = later[col].dropna().nunique()
            if best is None or (len(raw_levels) - true_levels) > best["_debug_extra"]:
                best = {
                    "present": True,
                    "column": col,
                    "true_levels": int(true_levels),
                    "_debug_extra": len(raw_levels) - true_levels,
                    "_debug_raw_levels": sorted(str(v) for v in raw_levels),
                }
    if best is None:
        return {"present": False, "column": None, "true_levels": None}
    return best


def run_all_checks(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> dict:
    train = load_raw(train_path)
    later = load_raw(later_path)
    holdout = load_raw(holdout_path)
    return {
        "leaking_column": check_leaking_column(train, holdout),
        "unit_mixup": check_unit_mixup(train),
        "sentinel_value": check_sentinel_value(train),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train),
        "target_typos": check_target_typos(train),
        "spelling_variants": check_spelling_variants(train, later),
    }


if __name__ == "__main__":
    import json

    results = run_all_checks()
    print(json.dumps(results, indent=2, default=str))
