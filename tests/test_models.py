import numpy as np
import pandas as pd
import pytest

from forecast_scarce.models import available, build
from forecast_scarce.models.prepare import imputation_rate, impute

HORIZON = 28


def panel(values, series="a", start="2023-01-01"):
    ds = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"series_id": series, "ds": ds, "y": np.asarray(values, "float64")})


# Shared imputation.


def test_interior_gap_is_interpolated():
    out = impute(panel([10.0, np.nan, 30.0]))
    assert out["y"].tolist() == [10.0, 20.0, 30.0]


def test_leading_gap_is_dropped_not_invented():
    out = impute(panel([np.nan, np.nan, 5.0, 6.0]))
    assert len(out) == 2
    assert out["y"].tolist() == [5.0, 6.0]


def test_trailing_gap_carries_the_last_observation_forward():
    out = impute(panel([5.0, 6.0, np.nan]))
    assert out["y"].tolist() == [5.0, 6.0, 6.0]


def test_series_with_nothing_observed_is_dropped():
    assert impute(panel([np.nan] * 5)).empty


def test_imputation_leaves_clean_series_untouched():
    clean = panel([1.0, 2.0, 3.0])
    assert impute(clean)["y"].tolist() == [1.0, 2.0, 3.0]


def test_imputation_rate_reports_the_share_invented():
    assert imputation_rate(panel([1.0, np.nan, np.nan, 4.0])) == 0.5
    assert imputation_rate(panel([1.0, 2.0])) == 0.0


def test_series_are_imputed_independently():
    both = pd.concat([panel([10.0, np.nan, 30.0], "a"), panel([1.0, 1.0, 1.0], "b")])
    out = impute(both).set_index(["series_id", "ds"])["y"]
    assert out.loc["a"].tolist() == [10.0, 20.0, 30.0]
    assert out.loc["b"].tolist() == [1.0, 1.0, 1.0]


# Every model honours the same contract.


@pytest.fixture
def training():
    rng = np.random.default_rng(0)
    frames = [
        panel(10 + rng.normal(0, 1, 200), "smooth"),
        panel(np.where(np.arange(200) % 4 == 0, 5.0, 0.0), "intermittent"),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def targets(training):
    start = training["ds"].max() + pd.Timedelta(days=1)
    ds = pd.date_range(start, periods=HORIZON, freq="D")
    return pd.concat(
        [
            pd.DataFrame({"series_id": sid, "ds": ds})
            for sid in ("smooth", "intermittent")
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize("name", available())
def test_model_predicts_every_target(name, training, targets):
    model = build(name)
    model.fit(training)
    out = model.predict(targets)

    assert list(out.columns) == ["series_id", "ds", "y"]
    assert len(out) == len(targets)
    assert out["y"].notna().all(), f"{name} left gaps in its forecast"


@pytest.mark.parametrize("name", available())
def test_predict_before_fit_is_an_error(name, targets):
    with pytest.raises((RuntimeError, ValueError)):
        build(name).predict(targets)


@pytest.mark.parametrize("name", available())
def test_model_tolerates_gaps_in_training(name, training, targets):
    gappy = training.copy()
    rng = np.random.default_rng(1)
    holes = rng.choice(len(gappy), size=len(gappy) // 5, replace=False)
    gappy.loc[gappy.index[holes], "y"] = np.nan

    model = build(name)
    model.fit(gappy)
    assert model.predict(targets)["y"].notna().all()


def test_lightgbm_never_forecasts_negative_demand(training, targets):
    model = build("lightgbm")
    model.fit(training)
    assert (model.predict(targets)["y"] >= 0).all()


def test_seasonal_naive_repeats_the_last_week(targets):
    week = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    train = panel(week * 20, "smooth")
    model = build("seasonal_naive")
    model.fit(train)

    start = train["ds"].max() + pd.Timedelta(days=1)
    horizon = pd.DataFrame(
        {"series_id": "smooth", "ds": pd.date_range(start, periods=7, freq="D")}
    )
    assert model.predict(horizon)["y"].tolist() == week


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build("prophet")
