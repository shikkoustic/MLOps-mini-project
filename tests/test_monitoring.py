"""Tests for Stage 4 monitoring helpers. Uses tiny fake data, not the CSVs."""
import pandas as pd

from mp128.monitoring import compute_psi, mae_by_group


def test_compute_psi_is_zero_for_identical_distributions():
    series = pd.Series(range(100))
    psi = compute_psi(series, series, bins=10)
    assert psi < 1e-6


def test_compute_psi_is_large_for_shifted_distribution():
    expected = pd.Series(range(100))
    actual = pd.Series(range(1000, 1100))
    psi = compute_psi(expected, actual, bins=10)
    assert psi > 1.0


class _ConstantModel:
    def predict(self, X):
        return [100.0] * len(X)


def test_mae_by_group_finds_worst_group():
    later = pd.DataFrame({
        "station_zone": ["A", "A", "B", "B"],
        "pm25_next_day": [100.0, 100.0, 150.0, 170.0],
    })
    result = mae_by_group(_ConstantModel(), later)
    assert result["mae_by_group"]["A"] == 0.0
    assert result["worst_served_group"] == "B"
