import numpy as np
import pandas as pd
import pytest

from forecast_scarce import figures

HISTORIES = (60, 730)
CELL = dict(dataset="cta", n_series=20, missing_pattern="none", seed=0)


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    rows = []
    for history in HISTORIES:
        # LightGBM worse than the baseline at 60 days, better at 730.
        shift = 0.2 if history == 60 else -0.2
        for i in range(20):
            base = 1.0 + rng.normal(0, 0.05)
            for model, value in (("lightgbm", base + shift), ("ets", base), ("croston", base + 1)):
                rows.append(
                    {
                        "series_id": f"s{i}", "model": model, "mase": value, "error": None,
                        "missing_rate": 0.0, "quadrant": "smooth",
                        "history_days": history, **CELL,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def daily():
    rng = np.random.default_rng(1)
    ds = pd.date_range("2024-01-01", periods=56, freq="D")
    frames = []
    for history in HISTORIES:
        for model, level in (("lightgbm", 1.0), ("ets", 1.1), ("croston", 2.0)):
            frames.append(
                pd.DataFrame(
                    {"ds": ds, "model": model, "scaled_error": level + rng.normal(0, 0.05, 56),
                     "history_days": history, "missing_rate": 0.0, **CELL}
                )
            )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def shap():
    features = ["lag_1", "lag_7", "mean_7", "mean_28", "horizon", "dayofweek", "lag_364"]
    rows = []
    for history in HISTORIES:
        for rate, pattern in ((0.0, "none"), (0.15, "mcar")):
            for seed in (0, 1):
                for rank, feature in enumerate(features):
                    rows.append(
                        {
                            "feature": feature, "mean_abs_shap": float(len(features) - rank),
                            "dataset": "cta", "history_days": history, "n_series": 20,
                            "missing_rate": rate, "missing_pattern": pattern,
                            "model": "lightgbm", "seed": seed, "fold": 0,
                        }
                    )
    return pd.DataFrame(rows)


def written(out, name):
    return all((out / f"{name}.{ext}").exists() for ext in ("pdf", "png", "csv"))


def test_crossover_writes_every_format_and_its_table(results, tmp_path):
    figures.crossover(results, tmp_path)
    assert written(tmp_path, "crossover")

    table = pd.read_csv(tmp_path / "crossover.csv")
    ets = table.loc[table["baseline"] == "ets"].set_index("history_days")
    assert ets.loc[60, "difference"] > 0 > ets.loc[730, "difference"]
    assert ets.loc[60, "lower"] > 0  # the crossover is resolved, not noise


def test_crossover_ignores_failed_rows(results, tmp_path):
    broken = results.copy()
    broken.loc[broken.index[:5], "error"] = "TypeError: boom"
    figures.crossover(broken, tmp_path)
    assert written(tmp_path, "crossover")


def test_drift_figure(tmp_path):
    manifests = [
        {"spec": {"dataset": "cta", "history_days": h, "missing_rate": 0.0},
         "quadrant_drift_rate": rate}
        for h, rate in ((60, 0.1), (60, 0.05), (730, 0.0))
    ]
    figures.drift(manifests, tmp_path)
    assert written(tmp_path, "drift")
    table = pd.read_csv(tmp_path / "drift.csv").set_index("history_days")
    assert table.loc[60, "drift"] == pytest.approx(0.075)


def test_significance_matrix_is_antisymmetric(daily, tmp_path):
    figures.significance(daily, tmp_path, horizon=1)
    assert written(tmp_path, "significance")

    table = pd.read_csv(tmp_path / "significance.csv")
    one = table.set_index(["history_days", "row", "column"])["difference"]
    for (history, row, column), value in one.items():
        assert one[(history, column, row)] == pytest.approx(-value)


def test_attribution_figure(shap, tmp_path):
    figures.attribution(shap, tmp_path)
    assert written(tmp_path, "attribution")
    table = pd.read_csv(tmp_path / "attribution.csv")
    assert {"60 vs 730 days", "15% MCAR vs clean"} <= set(table["comparison"])


def test_every_model_has_a_fixed_distinct_colour_and_marker():
    """Colour follows the model, including models not yet implemented, so adding
    them later repaints nothing that already exists."""
    assert figures.MODEL_COLORS["lightgbm"] == "#2a78d6"
    assert len(set(figures.MODEL_COLORS.values())) == len(figures.MODEL_ORDER)
    assert len(set(figures.MODEL_MARKERS.values())) == len(figures.MODEL_ORDER)
    assert set(figures.MODEL_NAMES) == set(figures.MODEL_ORDER)
