import pandas as pd
import pytest

from forecast_scarce.splits import rolling_origin, split

TEST_START = pd.Timestamp("2024-01-01")


def test_folds_tile_the_evaluation_period_without_overlap():
    folds = rolling_origin(TEST_START, history_days=365, test_days=168, horizon=28)
    assert len(folds) == 6
    for earlier, later in zip(folds[:-1], folds[1:], strict=True):
        assert earlier.test_end == later.test_start
    assert folds[0].test_start == TEST_START
    assert folds[-1].test_end == TEST_START + pd.Timedelta(days=168)


def test_training_window_is_the_same_width_in_every_fold():
    """The training window slides; it must never expand as the origin advances."""
    for history in (60, 120, 365, 730):
        folds = rolling_origin(TEST_START, history_days=history, test_days=168)
        widths = {(f.train_end - f.train_start).days for f in folds}
        assert widths == {history}


def test_training_never_reaches_its_own_targets():
    for f in rolling_origin(TEST_START, history_days=60, test_days=168):
        assert f.train_end <= f.test_start


def test_later_folds_may_train_on_earlier_folds_targets():
    """Standard rolling-origin behaviour: actuals become available over time."""
    folds = rolling_origin(TEST_START, history_days=60, test_days=168)
    assert folds[1].train_start < folds[0].test_end
    assert folds[1].origin == folds[0].test_end


def test_test_period_is_identical_regardless_of_history_length():
    spans = {
        (f.test_start, f.test_end)
        for history in (60, 730)
        for f in rolling_origin(TEST_START, history_days=history, test_days=168)
    }
    assert len(spans) == 6  # six distinct folds, shared across both histories


def test_indivisible_test_period_is_rejected():
    with pytest.raises(ValueError, match="not divisible by horizon"):
        rolling_origin(TEST_START, history_days=365, test_days=100, horizon=28)


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon must be positive"):
        rolling_origin(TEST_START, history_days=365, test_days=168, horizon=0)


def panel(start, periods):
    ds = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"series_id": "a", "ds": ds, "y": range(periods)})


def test_split_returns_the_right_row_counts():
    frame = panel("2023-01-01", 600)
    fold = rolling_origin(TEST_START, history_days=60, test_days=168)[0]
    train, test = split(frame, fold)
    assert len(train) == 60
    assert len(test) == 28


def test_split_boundaries_are_half_open():
    frame = panel("2023-01-01", 600)
    fold = rolling_origin(TEST_START, history_days=60, test_days=168)[0]
    train, test = split(frame, fold)
    assert train["ds"].max() == TEST_START - pd.Timedelta(days=1)
    assert test["ds"].min() == TEST_START
