"""Corporacion Favorita grocery sales.

Two quirks drive the handling here. Favorita omits rows entirely for days an
item-store combination recorded no sale, so a missing row is ambiguous between "sold
zero" and "store shut"; we reindex to a daily grid and leave those NaN rather
than guessing. And unit_sales goes negative for returns, which we keep as-is
because clipping would distort the intermittency statistics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.csv as pv

from .base import Dataset, mask_leading_absence, to_daily_grid
from .download import RAW_DIR, kaggle_competition

SLUG = "favorita-grocery-sales-forecasting"

# The full file is ~125M rows. Most of the useful perishable structure is in the
# later years and the scarcity protocol never asks for more than 730 days, so
# the default trims the head. Pass since=None to load everything.
DEFAULT_SINCE = "2015-01-01"


def load(since: str | None = DEFAULT_SINCE, raw_dir: Path | None = None) -> Dataset:
    raw = raw_dir or (RAW_DIR / "favorita")
    kaggle_competition(SLUG, raw)

    table = pv.read_csv(
        raw / "train.csv",
        convert_options=pv.ConvertOptions(
            include_columns=["date", "store_nbr", "item_nbr", "unit_sales"],
            column_types={"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
        ),
    )
    sales = table.to_pandas()
    del table

    sales["ds"] = pd.to_datetime(sales["date"])
    sales = sales.drop(columns="date")
    if since is not None:
        sales = sales.loc[sales["ds"] >= pd.Timestamp(since)]

    sales["series_id"] = (
        sales["store_nbr"].astype(str) + "_" + sales["item_nbr"].astype(str)
    )

    items = pd.read_csv(raw / "items.csv")
    stores = pd.read_csv(raw / "stores.csv")

    static = sales[["series_id", "store_nbr", "item_nbr"]].drop_duplicates("series_id")
    static = static.merge(items, on="item_nbr", how="left").merge(
        stores, on="store_nbr", how="left"
    )
    static = static.set_index("series_id").astype("category")

    panel = sales[["series_id", "ds", "unit_sales"]].rename(columns={"unit_sales": "y"})
    panel = to_daily_grid(panel, panel["ds"].min(), panel["ds"].max())
    panel = mask_leading_absence(panel)

    static = static.loc[static.index.isin(panel["series_id"].unique())]
    static.index.name = "series_id"

    return Dataset(name="favorita", panel=panel, static=static)
