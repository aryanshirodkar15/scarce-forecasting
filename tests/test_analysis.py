import numpy as np
import pandas as pd
import pytest

from forecast_scarce.analysis import (
    benjamini_hochberg,
    bootstrap_difference,
    compare_cell,
    compare_grid,
    diebold_mariano,
    summarise,
)

# Diebold-Mariano.


def test_identical_forecasts_give_the_null():
    loss = np.array([1.0, 2.0, 3.0, 2.0, 1.0] * 8)
    assert diebold_mariano(loss, loss, horizon=1) == (0.0, 1.0)


def test_uniformly_worse_model_diverges():
    """A constant nonzero gap on every date is certainty, not missing evidence."""
    statistic, p_value = diebold_mariano(np.full(40, 2.0), np.full(40, 1.0), horizon=1)
    assert statistic == np.inf
    assert p_value == 0.0


def test_better_model_gives_a_negative_statistic():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 160)
    statistic, p_value = diebold_mariano(1.0 + noise, 1.5 + noise, horizon=1)
    assert statistic < 0
    assert p_value < 0.01


def test_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.3, 160)
    b = rng.normal(1.0, 0.3, 160)
    assert diebold_mariano(a, b, horizon=1)[1] > 0.05


def test_hac_correction_widens_the_variance():
    """Ignoring autocorrelation understates variance and overstates significance."""
    rng = np.random.default_rng(2)
    smooth = pd.Series(rng.normal(0.1, 0.5, 200)).rolling(14, min_periods=1).mean().to_numpy()
    zero = np.zeros_like(smooth)
    assert diebold_mariano(smooth, zero, 28)[1] > diebold_mariano(smooth, zero, 1)[1]


def test_too_few_observations_is_undefined():
    assert np.isnan(diebold_mariano(np.array([1.0]), np.array([2.0]))[0])


def test_all_missing_is_undefined():
    nans = np.array([np.nan] * 10)
    assert np.isnan(diebold_mariano(nans, nans)[0])


# Benjamini-Hochberg.


def test_bh_known_example():
    out = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)
    assert np.allclose(out["p_adjusted"], 0.05)
    assert out["reject"].all()


def test_bh_rejects_only_the_strong_result():
    out = benjamini_hochberg([0.001, 0.5, 0.6], alpha=0.05)
    assert out["reject"].tolist() == [True, False, False]
    assert out["p_adjusted"].iloc[0] == pytest.approx(0.003)


def test_bh_adjusted_values_are_monotone():
    rng = np.random.default_rng(0)
    out = benjamini_hochberg(rng.uniform(0, 1, 50))
    ordered = out.sort_values("p_value")["p_adjusted"].to_numpy()
    assert np.all(np.diff(ordered) >= -1e-12)


def test_bh_is_conservative_about_an_isolated_result():
    """One modest result among many nulls is what multiplicity correction catches.

    Uncorrected, p=0.04 passes at alpha=0.05. Among twenty tests it is exactly
    what chance produces, so BH declines it.
    """
    out = benjamini_hochberg([0.04] + [0.9] * 19, alpha=0.05)
    assert not out["reject"].any()
    assert out["p_adjusted"].iloc[0] == pytest.approx(0.8)


def test_bh_rejects_a_block_of_consistent_results():
    """BH is not uniformly stricter: when every test agrees, it rejects them all,
    because the expected false discovery proportion is still controlled."""
    out = benjamini_hochberg([0.04] * 20, alpha=0.05)
    assert out["reject"].all()


def test_bh_ignores_undefined_tests():
    out = benjamini_hochberg([0.001, np.nan, 0.6])
    assert out["reject"].tolist() == [True, False, False]
    assert np.isnan(out["p_adjusted"].iloc[1])


# Bootstrap.


def per_series(values_a, values_b):
    rows = []
    for i, (a, b) in enumerate(zip(values_a, values_b, strict=True)):
        rows.append({"series_id": f"s{i}", "model": "a", "mase": a})
        rows.append({"series_id": f"s{i}", "model": "b", "mase": b})
    return pd.DataFrame(rows)


def test_bootstrap_interval_contains_zero_when_models_tie():
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 0.2, 80)
    point, lower, upper = bootstrap_difference(per_series(values, values), "a", "b")
    assert point == pytest.approx(0.0)
    assert lower <= 0 <= upper


def test_bootstrap_interval_excludes_zero_for_a_clear_winner():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 120)
    point, _, upper = bootstrap_difference(per_series(a, a + 0.5), "a", "b")
    assert point == pytest.approx(-0.5, abs=0.01)
    assert upper < 0


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    frame = per_series(rng.normal(1, 0.2, 50), rng.normal(1.1, 0.2, 50))
    assert bootstrap_difference(frame, "a", "b", seed=7) == bootstrap_difference(
        frame, "a", "b", seed=7
    )


def test_bootstrap_on_missing_model_is_undefined():
    assert np.isnan(bootstrap_difference(per_series([1.0] * 5, [1.0] * 5), "a", "missing")[0])


# Cell and grid comparison.


def daily_frame(models, n=60, **keys):
    rng = np.random.default_rng(0)
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.concat(
        [
            pd.DataFrame(
                {"ds": ds, "model": name, "scaled_error": level + rng.normal(0, 0.05, n), **keys}
            )
            for name, level in models.items()
        ],
        ignore_index=True,
    )


def test_compare_cell_covers_every_pair():
    out = compare_cell(daily_frame({"a": 1.0, "b": 1.5, "c": 2.0}), horizon=1)
    assert {(c.model_a, c.model_b) for c in out} == {("a", "b"), ("a", "c"), ("b", "c")}


def test_compare_cell_orders_by_loss():
    out = {(c.model_a, c.model_b): c for c in compare_cell(daily_frame({"a": 1.0, "b": 1.5}), 1)}
    assert out[("a", "b")].mean_difference < 0  # a loses less


def test_compare_grid_applies_correction_across_cells():
    frames = [
        daily_frame(
            {"a": 1.0, "b": 1.02},
            dataset="cta",
            history_days=h,
            n_series=10,
            missing_rate=0.0,
            missing_pattern="none",
            seed=0,
        )
        for h in (60, 120, 365, 730)
    ]
    table = compare_grid(pd.concat(frames, ignore_index=True), horizon=1)
    assert len(table) == 4
    assert (table["p_adjusted"] >= table["p_value"]).all()


def test_summarise_counts_the_series_behind_each_number():
    results = pd.DataFrame(
        {
            "dataset": "cta",
            "history_days": 60,
            "n_series": 2,
            "missing_rate": 0.0,
            "missing_pattern": "none",
            "seed": 0,
            "model": ["a", "a"],
            "quadrant": "smooth",
            "series_id": ["s1", "s2"],
            "mase": [1.0, 2.0],
        }
    )
    out = summarise(results)
    assert out["mase"].iloc[0] == 1.5
    assert out["n_series_scored"].iloc[0] == 2
