import numpy as np
import pandas as pd
import pytest

from forecast_scarce.metrics import (
    evaluate,
    mase,
    pinball,
    rmsse,
    seasonal_diffs,
    wape,
)

# Seasonal differences are all exactly 1, so the MASE/RMSSE scale is 1 and the
# metrics reduce to plain error statistics that can be checked by hand.
TRAIN = [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8]


def test_scale_is_one_for_this_training_series():
    assert seasonal_diffs(TRAIN, 7).tolist() == [1.0] * 7


def test_mase_reduces_to_mean_absolute_error_at_unit_scale():
    assert mase([10, 20], [12, 18], TRAIN) == 2.0


def test_rmsse_reduces_to_root_mean_squared_error_at_unit_scale():
    assert rmsse([10, 20], [12, 18], TRAIN) == 2.0


def test_wape_is_total_error_over_total_demand():
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_perfect_forecast_scores_zero():
    assert mase([10, 20], [10, 20], TRAIN) == 0.0
    assert rmsse([10, 20], [10, 20], TRAIN) == 0.0
    assert wape([10, 20], [10, 20]) == 0.0


# The distinction the whole benchmark rests on.


def test_unobserved_actuals_are_excluded_not_scored_as_zero():
    """A NaN target must not contribute, however wild the forecast against it."""
    scored = mase([10, np.nan], [12, 1000], TRAIN)
    assert scored == 2.0  # identical to scoring the first point alone


def test_observed_zeros_are_scored_normally():
    """Zero demand is a real observation and must not be skipped like a gap."""
    assert mase([0, 0], [1, 1], TRAIN) == 1.0
    assert not np.isnan(rmsse([0, 0], [1, 1], TRAIN))


def test_all_unobserved_actuals_give_nan():
    assert np.isnan(mase([np.nan, np.nan], [1, 2], TRAIN))
    assert np.isnan(wape([np.nan, np.nan], [1, 2]))


def test_gaps_in_training_drop_only_the_straddling_pairs():
    train = list(TRAIN)
    train[0] = np.nan  # kills exactly one seasonal pair
    assert seasonal_diffs(train, 7).tolist() == [1.0] * 6


def test_missing_forecast_is_excluded_too():
    assert mase([10, 20], [12, np.nan], TRAIN) == 2.0


# Degenerate denominators.


def test_flat_training_series_gives_nan_rather_than_dividing_by_zero():
    flat = [5] * 14  # every seasonal difference is zero
    assert np.isnan(mase([10, 20], [12, 18], flat))
    assert np.isnan(rmsse([10, 20], [12, 18], flat))


def test_training_shorter_than_the_season_gives_nan():
    assert np.isnan(mase([10], [12], [1, 2, 3], seasonality=7))


def test_wape_with_no_demand_gives_nan():
    assert np.isnan(wape([0, 0], [1, 1]))


# Pinball loss.


def test_pinball_penalises_under_forecasting_more_at_high_quantiles():
    assert pinball([10], [8], 0.9) == pytest.approx(1.8)
    assert pinball([10], [12], 0.9) == pytest.approx(0.2)


def test_pinball_is_symmetric_at_the_median():
    assert pinball([10], [8], 0.5) == pytest.approx(1.0)
    assert pinball([10], [12], 0.5) == pytest.approx(1.0)


def test_pinball_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="quantile must be in"):
        pinball([10], [8], 1.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        mase([10, 20], [12], TRAIN)


def test_seasonality_must_be_positive():
    with pytest.raises(ValueError, match="seasonality must be at least 1"):
        seasonal_diffs(TRAIN, 0)


# Panel-level evaluation.


def frame(series, values, start):
    ds = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"series_id": series, "ds": ds, "y": np.asarray(values, "float64")})


def test_evaluate_produces_one_row_per_series():
    train = pd.concat([frame("a", TRAIN, "2020-01-01"), frame("b", TRAIN, "2020-01-01")])
    actual = pd.concat([frame("a", [10, 20], "2020-01-15"), frame("b", [10, 20], "2020-01-15")])
    forecast = pd.concat([frame("a", [12, 18], "2020-01-15"), frame("b", [10, 20], "2020-01-15")])

    out = evaluate(actual, forecast, train).set_index("series_id")
    assert out.loc["a", "mase"] == 2.0
    assert out.loc["b", "mase"] == 0.0
    assert out.loc["a", "n_scored"] == 2


def test_evaluate_counts_only_scored_points():
    train = frame("a", TRAIN, "2020-01-01")
    actual = frame("a", [10, np.nan], "2020-01-15")
    forecast = frame("a", [12, 18], "2020-01-15")

    out = evaluate(actual, forecast, train).set_index("series_id")
    assert out.loc["a", "n_scored"] == 1
    assert out.loc["a", "mase"] == 2.0
