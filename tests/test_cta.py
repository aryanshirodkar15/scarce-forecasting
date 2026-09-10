import numpy as np
import pandas as pd
import pytest

from forecast_scarce.data.base import to_daily_grid
from forecast_scarce.data.cta import _resolve_conflicting_days


def frame(rows):
    df = pd.DataFrame(rows, columns=["series_id", "ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def test_conflicting_counts_for_one_day_are_blanked():
    # CTA reported ~600 station-days twice in summer 2011 with different totals.
    out = _resolve_conflicting_days(
        frame([("40010", "2011-07-18", 2021.0), ("40010", "2011-07-18", 2017.0)])
    )
    assert len(out) == 1
    assert np.isnan(out["y"].iloc[0])


def test_identical_repeats_are_collapsed_not_blanked():
    out = _resolve_conflicting_days(
        frame([("40010", "2011-07-18", 2021.0), ("40010", "2011-07-18", 2021.0)])
    )
    assert len(out) == 1
    assert out["y"].iloc[0] == 2021.0


def test_clean_days_are_untouched():
    rows = frame([("40010", "2011-07-18", 2021.0), ("40010", "2011-07-19", 1900.0)])
    out = _resolve_conflicting_days(rows)
    assert len(out) == 2
    assert out["y"].tolist() == [2021.0, 1900.0]


def test_conflict_in_one_station_does_not_affect_another():
    out = _resolve_conflicting_days(
        frame([
            ("40010", "2011-07-18", 2021.0),
            ("40010", "2011-07-18", 2017.0),
            ("40020", "2011-07-18", 3781.0),
        ])
    ).set_index("series_id")
    assert np.isnan(out.loc["40010", "y"])
    assert out.loc["40020", "y"] == 3781.0


def test_daily_grid_rejects_unresolved_duplicates():
    """The raw pandas error for this is cryptic; fail with something actionable."""
    rows = frame([("a", "2020-01-01", 1.0), ("a", "2020-01-01", 2.0)])
    with pytest.raises(ValueError, match="duplicate \\(series_id, ds\\) rows"):
        to_daily_grid(rows, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
