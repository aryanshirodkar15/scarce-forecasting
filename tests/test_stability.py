import numpy as np
import pandas as pd
import pytest

from forecast_scarce.analysis import (
    available_at,
    feature_lookback,
    regime_stability,
    seed_stability,
)

SHORT = {
    "lag_1": 6.0,
    "lag_7": 5.0,
    "mean_7": 4.0,
    "mean_28": 3.0,
    "horizon": 2.0,
    "dayofweek": 1.0,
}


def shap_rows(values, **keys):
    base = dict(
        dataset="cta",
        history_days=60,
        n_series=12,
        missing_rate=0.0,
        missing_pattern="none",
        model="lightgbm",
        seed=0,
        fold=0,
    )
    base.update(keys)
    return pd.DataFrame([{"feature": f, "mean_abs_shap": v, **base} for f, v in values.items()])


# Lookback parsing.


@pytest.mark.parametrize(
    "name, days",
    [
        ("lag_1", 1),
        ("lag_364", 364),
        ("mean_91", 91),
        ("std_7", 7),
        ("zero_rate_28", 28),
        ("horizon", 0),
        ("dayofweek", 0),
        ("days_since_demand", 0),
        ("stationname", 0),
    ],
)
def test_feature_lookback(name, days):
    assert feature_lookback(name) == days


def test_available_at_excludes_what_the_history_cannot_support():
    features = ["lag_1", "lag_28", "lag_91", "mean_364", "horizon"]
    assert available_at(features, 60) == ["lag_1", "lag_28", "horizon"]
    assert available_at(features, 730) == features


# Seed stability.


def test_identical_seeds_agree_perfectly():
    frame = pd.concat([shap_rows(SHORT, seed=0), shap_rows(SHORT, seed=1)])
    out = seed_stability(frame).iloc[0]
    assert out["rho"] == pytest.approx(1.0)
    assert out["top_k_jaccard"] == pytest.approx(1.0)


def test_reversed_seeds_disagree_completely():
    reversed_ = {f: 7.0 - v for f, v in SHORT.items()}
    frame = pd.concat([shap_rows(SHORT, seed=0), shap_rows(reversed_, seed=1)])
    assert seed_stability(frame).iloc[0]["rho"] == pytest.approx(-1.0)


def test_seed_stability_ignores_features_the_history_cannot_support():
    """Two seeds that agree on everything real must not be marked unstable
    because of features that are NaN in this cell anyway."""
    a = {**SHORT, "lag_364": 100.0, "mean_364": 0.0}
    b = {**SHORT, "lag_364": 0.0, "mean_364": 100.0}
    frame = pd.concat([shap_rows(a, seed=0), shap_rows(b, seed=1)])
    out = seed_stability(frame).iloc[0]
    assert out["rho"] == pytest.approx(1.0)
    assert out["n_features"] == len(SHORT)


def test_folds_are_averaged_before_ranking():
    noisy = pd.concat(
        [
            shap_rows({**SHORT, "lag_1": 0.0}, seed=0, fold=0),
            shap_rows({**SHORT, "lag_1": 12.0}, seed=0, fold=1),  # mean 6.0, as in SHORT
            shap_rows(SHORT, seed=1),
        ]
    )
    assert seed_stability(noisy).iloc[0]["rho"] == pytest.approx(1.0)


# Regime stability.


def test_history_regime_compares_only_shared_features():
    """At 730 days the long lags dominate. At 60 they do not exist. Agreement
    must be judged on what both regimes could actually see."""
    long_ = {**SHORT, "lag_364": 50.0, "mean_364": 40.0}
    frame = pd.concat(
        [shap_rows(SHORT, history_days=60), shap_rows(long_, history_days=730)]
    )
    out = regime_stability(frame, axis="history_days").iloc[0]
    assert out["regime"] == 60
    assert out["reference"] == 730
    assert out["rho"] == pytest.approx(1.0)
    assert out["n_features"] == len(SHORT)


def test_history_regime_detects_a_genuine_reordering():
    reordered = {**SHORT, "lag_1": 1.0, "dayofweek": 6.0}
    frame = pd.concat(
        [shap_rows(reordered, history_days=60), shap_rows(SHORT, history_days=730)]
    )
    assert regime_stability(frame, axis="history_days").iloc[0]["rho"] < 1.0


def test_missingness_regime_uses_the_clean_panel_as_reference():
    frame = pd.concat(
        [
            shap_rows(SHORT, missing_rate=0.0, missing_pattern="none"),
            shap_rows(SHORT, missing_rate=0.15, missing_pattern="mcar"),
        ]
    )
    out = regime_stability(frame, axis="missingness").iloc[0]
    assert out["reference"] == (0.0, "none")
    assert out["regime"] == (0.15, "mcar")
    assert out["rho"] == pytest.approx(1.0)


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown axis"):
        regime_stability(shap_rows(SHORT), axis="n_series")


# The model itself.


def test_lightgbm_attributions_cover_every_feature():
    from forecast_scarce.models import build

    rng = np.random.default_rng(0)
    ds = pd.date_range("2023-01-01", periods=200, freq="D")
    train = pd.DataFrame({"series_id": "a", "ds": ds, "y": 10 + rng.normal(0, 1, 200)})
    targets = pd.DataFrame(
        {"series_id": "a", "ds": pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=28)}
    )

    model = build("lightgbm")
    model.fit(train)
    out = model.attributions(targets)

    assert set(out["feature"]) == set(model._columns)
    assert (out["mean_abs_shap"] >= 0).all()
    assert out["mean_abs_shap"].sum() > 0
