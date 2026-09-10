"""M5 loader against a miniature stand-in for the real files."""

import pandas as pd
import pytest

from forecast_scarce.data import m5

CALENDAR = "d,date\nd_1,2011-01-29\nd_2,2011-01-30\nd_3,2011-01-31\nd_4,2011-02-01\n"
SALES = (
    "id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3,d_4\n"
    "a_CA_1,ITEM_A,FOODS_1,FOODS,CA_1,CA,0,0,3,0\n"
    "b_CA_1,ITEM_B,FOODS_1,FOODS,CA_1,CA,1,2,0,4\n"
    "a_TX_1,ITEM_A,FOODS_1,FOODS,TX_1,TX,0,0,0,0\n"
)


@pytest.fixture
def raw(tmp_path):
    (tmp_path / "calendar.csv").write_text(CALENDAR)
    (tmp_path / "sales_train_evaluation.csv").write_text(SALES)
    return tmp_path


def test_item_store_level_shape_and_dates(raw):
    d = m5.load(level="item_store", raw_dir=raw)
    assert d.name == "m5_item_store"
    # ITEM_A_TX_1 never sells and is dropped.
    assert set(d.panel["series_id"].unique()) == {"ITEM_A_CA_1", "ITEM_B_CA_1"}
    assert d.span == (pd.Timestamp("2011-01-29"), pd.Timestamp("2011-02-01"))


def test_leading_zeros_masked_but_interior_zero_kept(raw):
    d = m5.load(level="item_store", raw_dir=raw)
    a = d.panel.query("series_id == 'ITEM_A_CA_1'").set_index("ds")["y"]
    assert a[pd.Timestamp("2011-01-29")] != a[pd.Timestamp("2011-01-29")]  # NaN
    assert a[pd.Timestamp("2011-01-31")] == 3.0
    assert a[pd.Timestamp("2011-02-01")] == 0.0


def test_aggregation_sums_the_children(raw):
    d = m5.load(level="dept_store", raw_dir=raw)
    ca = d.panel.query("series_id == 'FOODS_1_CA_1'").set_index("ds")["y"]
    assert ca[pd.Timestamp("2011-01-29")] == 1.0  # 0 + 1
    assert ca[pd.Timestamp("2011-02-01")] == 4.0  # 0 + 4


def test_attributes_that_vary_within_a_series_become_null(raw):
    d = m5.load(level="dept_store", raw_dir=raw)
    static = d.static.loc["FOODS_1_CA_1"]
    # This series pools ITEM_A and ITEM_B, so item_id is not well defined.
    assert pd.isna(static["item_id"])
    # These are constant within the series and must survive.
    assert static["store_id"] == "CA_1"
    assert static["state_id"] == "CA"
    assert static["cat_id"] == "FOODS"


def test_unknown_level_rejected(raw):
    with pytest.raises(ValueError, match="unknown level"):
        m5.load(level="nonsense", raw_dir=raw)
