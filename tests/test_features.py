import numpy as np
import pandas as pd
import pytest

from forecast_scarce.features import LAGS, build, feature_columns

ORIGIN = pd.Timestamp("2024-01-01")


def history(values, series="a", end_before=ORIGIN):
    ds = pd.date_range(end=end_before - pd.Timedelta(days=1), periods=len(values), freq="D")
    return pd.DataFrame({"series_id": series, "ds": ds, "y": np.asarray(values, "float64")})


def targets(n=28, series="a", start=ORIGIN):
    ds = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"series_id": series, "ds": ds})


# Leakage is the bug that would invalidate every downstream result.


def test_history_reaching_the_origin_is_rejected():
    bad = history(list(range(10)))
    bad.loc[bad.index[-1], "ds"] = ORIGIN
    with pytest.raises(ValueError, match="would leak future observations"):
        build(bad, targets(), ORIGIN)


def test_history_past_the_origin_is_rejected():
    bad = history(list(range(10)))
    bad.loc[bad.index[-1], "ds"] = ORIGIN + pd.Timedelta(days=5)
    with pytest.raises(ValueError, match="leak"):
        build(bad, targets(), ORIGIN)


def test_every_feature_row_is_identical_across_horizons():
    """Origin-anchored means the history features cannot vary by target date."""
    out = build(history(list(range(1, 401))), targets(28), ORIGIN)
    history_cols = [c for c in out.columns if c.startswith(("lag_", "mean_", "std_"))]
    assert (out[history_cols].nunique() == 1).all()


def test_horizon_counts_from_one():
    out = build(history(list(range(1, 401))), targets(28), ORIGIN)
    assert out["horizon"].tolist() == list(range(1, 29))


# Lags.


def test_lag_one_is_the_day_before_the_origin():
    out = build(history([10, 20, 30]), targets(1), ORIGIN)
    assert out["lag_1"].iloc[0] == 30
    assert out["lag_2"].iloc[0] == 20
    assert out["lag_3"].iloc[0] == 10


def test_lags_beyond_the_history_are_nan():
    out = build(history([10, 20, 30]), targets(1), ORIGIN).iloc[0]
    for lag in LAGS:
        if lag > 3:
            assert np.isnan(out[f"lag_{lag}"]), f"lag_{lag} should be unavailable"


# Rolling statistics.


def test_window_not_reported_when_history_is_too_short():
    """A 60-day window must not report a '364-day mean' computed from 60 days."""
    out = build(history([5.0] * 60), targets(1), ORIGIN).iloc[0]
    assert out["mean_28"] == 5.0
    assert np.isnan(out["mean_91"])
    assert np.isnan(out["mean_364"])


def test_rolling_mean_uses_only_the_window():
    values = [0.0] * 50 + [10.0] * 7
    out = build(history(values), targets(1), ORIGIN).iloc[0]
    assert out["mean_7"] == 10.0
    assert out["mean_14"] == pytest.approx(70 / 14)


def test_unobserved_days_are_excluded_from_rolling_stats():
    values = [10.0, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0]
    out = build(history(values), targets(1), ORIGIN).iloc[0]
    assert out["mean_7"] == 10.0  # not dragged toward zero by the gap


def test_zero_rate_counts_observed_zeros_only():
    values = [0.0, 0.0, np.nan, 5.0, 5.0, 5.0, 5.0]
    out = build(history(values), targets(1), ORIGIN).iloc[0]
    assert out["zero_rate_7"] == pytest.approx(2 / 6)


# Intermittency features.


def test_days_since_demand():
    out = build(history([5.0, 0.0, 0.0]), targets(1), ORIGIN).iloc[0]
    assert out["days_since_demand"] == 3.0

    recent = build(history([0.0, 0.0, 5.0]), targets(1), ORIGIN).iloc[0]
    assert recent["days_since_demand"] == 1.0


def test_days_since_demand_is_nan_when_nothing_ever_sold():
    out = build(history([0.0] * 30), targets(1), ORIGIN).iloc[0]
    assert np.isnan(out["days_since_demand"])


# The property that makes the scarcity comparison valid.


def test_feature_columns_are_identical_at_every_history_length():
    short = build(history([5.0] * 60), targets(28), ORIGIN)
    long = build(history([5.0] * 730), targets(28), ORIGIN)
    assert feature_columns(short) == feature_columns(long)


def test_short_history_yields_the_same_columns_but_missing_values():
    short = build(history([5.0] * 60), targets(1), ORIGIN)
    long = build(history([5.0] * 730), targets(1), ORIGIN)
    assert np.isnan(short["lag_364"].iloc[0])
    assert long["lag_364"].iloc[0] == 5.0


def test_series_absent_from_history_gets_null_features():
    out = build(history([5.0] * 30, series="a"), targets(1, series="b"), ORIGIN).iloc[0]
    assert np.isnan(out["lag_1"])
    assert out["n_observed"] == 0.0


# Calendar and static attributes.


def test_calendar_features_describe_the_target_date():
    out = build(history([5.0] * 30), targets(3), ORIGIN)
    # 2024-01-01 was a Monday.
    assert out["dayofweek"].tolist() == [0, 1, 2]
    assert out["month"].tolist() == [1, 1, 1]
    assert out["is_weekend"].tolist() == [0, 0, 0]


def test_weekend_flag():
    out = build(history([5.0] * 30), targets(7), ORIGIN)
    assert out["is_weekend"].tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_static_attributes_are_joined():
    static = pd.DataFrame({"dept": ["FOODS"]}, index=pd.Index(["a"], name="series_id"))
    out = build(history([5.0] * 30), targets(1), ORIGIN, static=static)
    assert out["dept"].iloc[0] == "FOODS"


def test_multiple_series_are_built_independently():
    hist = pd.concat([history([1.0, 2.0, 3.0], "a"), history([9.0, 9.0, 9.0], "b")])
    tgt = pd.concat([targets(1, "a"), targets(1, "b")], ignore_index=True)
    out = build(hist, tgt, ORIGIN).set_index("series_id")
    assert out.loc["a", "lag_1"] == 3.0
    assert out.loc["b", "lag_1"] == 9.0
