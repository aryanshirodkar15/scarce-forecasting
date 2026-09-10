import numpy as np
import pandas as pd
import pytest

from forecast_scarce.data.base import Dataset, mask_leading_absence, to_daily_grid


def panel_from(rows):
    df = pd.DataFrame(rows, columns=["series_id", "ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype("float32")
    return df


def test_leading_zeros_become_absent_but_later_zeros_survive():
    panel = panel_from([
        ("a", "2020-01-01", 0.0),
        ("a", "2020-01-02", 0.0),
        ("a", "2020-01-03", 3.0),
        ("a", "2020-01-04", 0.0),
        ("a", "2020-01-05", 1.0),
    ])
    out = mask_leading_absence(panel)

    assert out["y"].isna().tolist() == [True, True, False, False, False]
    # The zero on the 4th is real zero demand and must not be blanked.
    assert out.loc[out["ds"] == pd.Timestamp("2020-01-04"), "y"].item() == 0.0


def test_series_that_never_sells_is_dropped():
    panel = panel_from([
        ("a", "2020-01-01", 0.0),
        ("a", "2020-01-02", 0.0),
        ("b", "2020-01-01", 0.0),
        ("b", "2020-01-02", 5.0),
    ])
    out = mask_leading_absence(panel)
    assert set(out["series_id"]) == {"b"}


def test_leading_mask_is_per_series():
    panel = panel_from([
        ("a", "2020-01-01", 2.0),
        ("a", "2020-01-02", 1.0),
        ("b", "2020-01-01", 0.0),
        ("b", "2020-01-02", 4.0),
    ])
    out = mask_leading_absence(panel).set_index(["series_id", "ds"])["y"]
    assert not np.isnan(out[("a", pd.Timestamp("2020-01-01"))])
    assert np.isnan(out[("b", pd.Timestamp("2020-01-01"))])


def test_daily_grid_fills_absent_dates_with_nan_not_zero():
    panel = panel_from([
        ("a", "2020-01-01", 5.0),
        ("a", "2020-01-04", 2.0),
    ])
    out = to_daily_grid(panel, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04"))

    assert len(out) == 4
    assert out["y"].isna().sum() == 2
    assert out["y"].sum() == 7.0


def test_daily_grid_preserves_explicit_zero():
    panel = panel_from([
        ("a", "2020-01-01", 0.0),
        ("a", "2020-01-03", 1.0),
    ])
    out = to_daily_grid(panel, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    ys = out.set_index("ds")["y"]
    assert ys[pd.Timestamp("2020-01-01")] == 0.0
    assert np.isnan(ys[pd.Timestamp("2020-01-02")])


def test_daily_grid_spans_every_series_equally():
    panel = panel_from([
        ("a", "2020-01-01", 1.0),
        ("b", "2020-01-03", 2.0),
    ])
    out = to_daily_grid(panel, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    assert out.groupby("series_id", observed=True).size().unique().tolist() == [3]


def static_for(ids):
    s = pd.DataFrame(index=pd.Index(ids, name="series_id"))
    s["kind"] = "x"
    return s


def test_dataset_rejects_panel_without_static_row():
    panel = panel_from([("a", "2020-01-01", 1.0)])
    with pytest.raises(ValueError, match="no static row"):
        Dataset(name="t", panel=panel, static=static_for(["b"]))


def test_dataset_rejects_non_datetime_ds():
    panel = pd.DataFrame({"series_id": ["a"], "ds": ["2020-01-01"], "y": [1.0]})
    with pytest.raises(ValueError, match="datetime64"):
        Dataset(name="t", panel=panel, static=static_for(["a"]))


def test_summary_separates_absence_from_zero():
    panel = panel_from([
        ("a", "2020-01-01", np.nan),
        ("a", "2020-01-02", 0.0),
        ("a", "2020-01-03", 0.0),
        ("a", "2020-01-04", 4.0),
    ])
    summary = Dataset(name="t", panel=panel, static=static_for(["a"])).summary()
    assert summary["pct_absent"] == 25.0
    # Two of the three observed values are zero.
    assert summary["pct_zero_when_observed"] == 66.67
    assert summary["n_series"] == 1


def test_parquet_roundtrip_preserves_nan(tmp_path):
    panel = panel_from([("a", "2020-01-01", np.nan), ("a", "2020-01-02", 2.0)])
    original = Dataset(name="t", panel=panel, static=static_for(["a"]))
    original.to_parquet(tmp_path)

    restored = Dataset.from_parquet(tmp_path, name="t")
    assert restored.panel["y"].isna().tolist() == [True, False]
    assert restored.summary() == original.summary()
