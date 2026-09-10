import json

import numpy as np
import pandas as pd
import pytest

from forecast_scarce.data.base import Dataset
from forecast_scarce.intermittency import classify
from forecast_scarce.sampling import STRATA, ScarcitySpec, sample

DAYS = 1000
PER_STRATUM = 10


def _values(kind, rng, n):
    if kind == "smooth":
        return 10 + rng.integers(0, 2, n).astype("float32")
    if kind == "erratic":
        return rng.choice([1.0, 60.0], n).astype("float32")
    out = np.zeros(n, dtype="float32")
    hits = np.arange(0, n, 3)
    if kind == "intermittent":
        out[hits] = 5.0
    else:  # lumpy
        out[hits] = rng.choice([1.0, 100.0], len(hits))
    return out


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=DAYS, freq="D")
    frames = []
    for kind in STRATA:
        for i in range(PER_STRATUM):
            frames.append(
                pd.DataFrame(
                    {"series_id": f"{kind}_{i}", "ds": ds, "y": _values(kind, rng, DAYS)}
                )
            )
    panel = pd.concat(frames, ignore_index=True)
    static = pd.DataFrame(index=pd.Index(panel["series_id"].unique(), name="series_id"))
    static["kind"] = [s.rsplit("_", 1)[0] for s in static.index]
    return Dataset(name="synthetic", panel=panel, static=static)


def test_fixture_lands_in_the_intended_quadrants(dataset):
    """If this fails every other test in the file is meaningless."""
    stats = classify(dataset.panel)
    for kind in STRATA:
        got = stats.loc[[f"{kind}_{i}" for i in range(PER_STRATUM)], "quadrant"]
        assert set(got) == {kind}, f"{kind} generator produced {set(got)}"


def spec(**kwargs):
    base = dict(dataset="synthetic", history_days=365, n_series=20)
    return ScarcitySpec(**{**base, **kwargs})


def test_identical_specs_produce_identical_samples(dataset):
    a = sample(dataset, spec())
    b = sample(dataset, spec())
    assert a.manifest.series_ids == b.manifest.series_ids
    assert a.manifest.panel_sha256 == b.manifest.panel_sha256


def test_selection_is_invariant_to_history_length(dataset):
    """The paired-design property: history length must not reshuffle the series."""
    short = sample(dataset, spec(history_days=60))
    long = sample(dataset, spec(history_days=730))
    assert short.manifest.series_ids == long.manifest.series_ids


def test_selection_is_invariant_to_missingness(dataset):
    clean = sample(dataset, spec())
    gappy = sample(dataset, spec(missing_rate=0.30, missing_pattern="mcar"))
    assert clean.manifest.series_ids == gappy.manifest.series_ids


def test_different_seed_changes_the_selection(dataset):
    a = sample(dataset, spec(seed=0))
    b = sample(dataset, spec(seed=1))
    assert a.manifest.series_ids != b.manifest.series_ids


def test_equal_allocation_across_quadrants(dataset):
    s = sample(dataset, spec(n_series=20))
    assert s.manifest.actual_per_stratum == {q: 5 for q in STRATA}
    assert s.manifest.n_series_actual == 20


def test_remainder_is_spread_not_dropped(dataset):
    s = sample(dataset, spec(n_series=22))
    assert sum(s.manifest.actual_per_stratum.values()) == 22
    assert sorted(s.manifest.actual_per_stratum.values()) == [5, 5, 6, 6]


def test_shortfall_is_recorded_when_a_quadrant_runs_out(dataset):
    s = sample(dataset, spec(n_series=40))  # only 10 exist per quadrant
    assert s.manifest.n_series_actual == 40
    s2 = sample(dataset, spec(n_series=60))
    assert s2.manifest.n_series_actual == 40
    assert s2.manifest.shortfall == {q: 5 for q in STRATA}


def test_window_carries_training_plus_the_shared_test_period(dataset):
    for days in (60, 120, 365, 730):
        s = sample(dataset, spec(history_days=days))
        per_series = s.panel.groupby("series_id", observed=True).size().unique()
        assert per_series.tolist() == [days + 168]


def test_test_period_is_identical_across_history_lengths(dataset):
    """The whole point of the split design: every cell is scored on the same days."""
    starts = {
        sample(dataset, spec(history_days=d)).manifest.test_start
        for d in (60, 120, 365, 730)
    }
    assert len(starts) == 1


def test_training_span_matches_history_days(dataset):
    s = sample(dataset, spec(history_days=120))
    test_start = pd.Timestamp(s.manifest.test_start)
    train = s.panel.loc[s.panel["ds"] < test_start]
    assert train.groupby("series_id", observed=True).size().unique().tolist() == [120]


def test_test_period_length_matches_spec(dataset):
    s = sample(dataset, spec(history_days=120))
    test_start = pd.Timestamp(s.manifest.test_start)
    held = s.panel.loc[s.panel["ds"] >= test_start]
    assert held.groupby("series_id", observed=True).size().unique().tolist() == [168]


def test_window_ends_at_the_frame_end(dataset):
    s = sample(dataset, spec(history_days=60))
    assert s.panel["ds"].max() == dataset.panel["ds"].max()
    assert s.manifest.window_end == str(dataset.panel["ds"].max().date())


def test_mcar_hits_the_requested_rate(dataset):
    s = sample(dataset, spec(missing_rate=0.15, missing_pattern="mcar"))
    assert abs(s.manifest.final_absent_rate - 0.15) < 0.01
    assert abs(s.manifest.injected_rate - 0.15) < 0.01


def test_zero_missingness_leaves_the_panel_untouched(dataset):
    s = sample(dataset, spec())
    assert s.manifest.injected_rate == 0.0
    assert s.panel["y"].isna().sum() == 0


def _mean_gap_run(panel):
    runs = []
    for _, group in panel.groupby("series_id", observed=True):
        flags = group["y"].isna().to_numpy()
        length = 0
        for flag in flags:
            if flag:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    return float(np.mean(runs)) if runs else 0.0


def test_burst_gaps_are_contiguous_unlike_mcar(dataset):
    scattered = sample(dataset, spec(missing_rate=0.15, missing_pattern="mcar"))
    bursty = sample(dataset, spec(missing_rate=0.15, missing_pattern="burst"))

    assert abs(bursty.manifest.final_absent_rate - 0.15) < 0.02
    # Same amount of absence, very different shape.
    assert _mean_gap_run(bursty.panel) > 2 * _mean_gap_run(scattered.panel)


def test_manifest_serializes(dataset):
    s = sample(dataset, spec(missing_rate=0.05, missing_pattern="burst"))
    payload = json.loads(s.manifest.to_json())
    assert payload["spec"]["history_days"] == 365
    assert payload["n_series_actual"] == 20
    assert len(payload["series_ids"]) == 20
    assert payload["panel_sha256"]


def test_static_is_restricted_to_the_sample(dataset):
    s = sample(dataset, spec(n_series=8))
    assert len(s.static) == 8
    assert set(s.static.index) == set(s.manifest.series_ids)


def test_drift_rate_is_reported(dataset):
    s = sample(dataset, spec(history_days=60))
    assert 0.0 <= s.manifest.quadrant_drift_rate <= 1.0


def test_history_plus_test_longer_than_frame_is_rejected():
    with pytest.raises(ValueError, match="exceeds frame_days"):
        spec(history_days=730, test_days=168, frame_days=730)


def test_rate_without_pattern_is_rejected():
    with pytest.raises(ValueError, match="missing_pattern is 'none'"):
        spec(missing_rate=0.1)


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="unknown missing_pattern"):
        spec(missing_rate=0.1, missing_pattern="drifting")


def test_n_series_below_stratum_count_is_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        spec(n_series=3)


def test_proportional_allocation_mirrors_the_pool(dataset):
    """The fixture is balanced, so proportional and equal agree on it."""
    s = sample(dataset, spec(n_series=20, allocation="proportional"))
    assert s.manifest.actual_per_stratum == {q: 5 for q in STRATA}


def test_proportional_allocation_on_a_lopsided_pool():
    """CTA is 98 percent smooth; equal allocation would return a handful of series."""
    rng = np.random.default_rng(1)
    ds = pd.date_range("2020-01-01", periods=DAYS, freq="D")
    frames = [
        pd.DataFrame({"series_id": f"smooth_{i}", "ds": ds, "y": _values("smooth", rng, DAYS)})
        for i in range(40)
    ]
    frames.append(
        pd.DataFrame({"series_id": "lumpy_0", "ds": ds, "y": _values("lumpy", rng, DAYS)})
    )
    panel = pd.concat(frames, ignore_index=True)
    static = pd.DataFrame(index=pd.Index(panel["series_id"].unique(), name="series_id"))
    static["kind"] = 1
    lopsided = Dataset(name="lopsided", panel=panel, static=static)

    equal = sample(lopsided, spec(dataset="lopsided", n_series=20))
    assert equal.manifest.n_series_actual < 20  # three strata are nearly empty

    proportional = sample(
        lopsided, spec(dataset="lopsided", n_series=20, allocation="proportional")
    )
    assert proportional.manifest.n_series_actual == 20
    assert proportional.manifest.actual_per_stratum["smooth"] >= 19


def test_largest_remainder_sums_exactly(dataset):
    for n in (7, 13, 19, 23):
        s = sample(dataset, spec(n_series=n, allocation="proportional"))
        assert sum(s.manifest.requested_per_stratum.values()) == n


def test_allocation_changes_the_selection_key(dataset):
    a = sample(dataset, spec(n_series=20))
    b = sample(dataset, spec(n_series=20, allocation="proportional"))
    assert a.manifest.selection_seed != b.manifest.selection_seed


def test_unknown_allocation_is_rejected():
    with pytest.raises(ValueError, match="unknown allocation"):
        spec(allocation="stratified-ish")
