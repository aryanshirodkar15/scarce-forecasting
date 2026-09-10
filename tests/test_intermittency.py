import numpy as np
import pandas as pd

from forecast_scarce.intermittency import classify


def series(name, values, start="2020-01-01"):
    ds = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {"series_id": name, "ds": ds, "y": np.asarray(values, dtype="float32")}
    )


def quadrant_of(values):
    return classify(series("a", values)).loc["a", "quadrant"]


def test_smooth_series():
    assert quadrant_of([10, 11] * 10) == "smooth"


def test_intermittent_series():
    # Demand every third day, always the same size.
    assert quadrant_of([5, 0, 0] * 7) == "intermittent"


def test_erratic_series():
    # Sells daily, but the size swings wildly.
    assert quadrant_of([1, 50] * 10) == "erratic"


def test_lumpy_series():
    assert quadrant_of([1, 0, 0, 100, 0, 0] * 4) == "lumpy"


def test_too_few_demands_is_insufficient():
    assert quadrant_of([7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7] + [0] * 10) == "insufficient"


def test_all_zero_series_is_insufficient():
    assert quadrant_of([0] * 30) == "insufficient"


def test_adi_and_cv2_values():
    stats = classify(series("a", [5, 0, 0] * 7)).loc["a"]
    assert stats["adi"] == 3.0  # 21 periods / 7 demands
    assert stats["cv2"] == 0.0  # every demand is the same size
    assert stats["n_observed"] == 21
    assert stats["n_nonzero"] == 7


def test_proportional_missingness_leaves_adi_unchanged():
    """The reason NaN periods are dropped rather than spanned.

    Masking one demand and two zeros out of every seven periods removes the same
    share of both, so ADI must come out identical. Had unobserved periods been
    counted as intervals, this would read 21/6 = 3.5 and the series would drift
    toward the intermittent cutoff purely because of the missingness treatment.
    """
    full = [5.0, 0.0, 0.0] * 7
    masked = list(full)
    masked[0] = np.nan  # one demand
    masked[1] = np.nan  # two zeros
    masked[2] = np.nan

    assert classify(series("a", full)).loc["a", "adi"] == 3.0
    assert classify(series("a", masked)).loc["a", "adi"] == 3.0
    assert classify(series("a", masked)).loc["a", "n_observed"] == 18


def test_negative_values_are_not_demand_occurrences():
    # Favorita encodes returns as negative sales; a return is not a demand.
    stats = classify(series("a", [5, -2] * 10)).loc["a"]
    assert stats["n_observed"] == 20
    assert stats["n_nonzero"] == 10
    assert stats["adi"] == 2.0


def test_classifies_many_series_independently():
    panel = pd.concat(
        [series("smooth", [10, 11] * 10), series("inter", [5, 0, 0] * 7)],
        ignore_index=True,
    )
    out = classify(panel)
    assert out.loc["smooth", "quadrant"] == "smooth"
    assert out.loc["inter", "quadrant"] == "intermittent"
    assert out.index.name == "series_id"
