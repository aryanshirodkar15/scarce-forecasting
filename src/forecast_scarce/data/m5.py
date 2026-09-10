"""M5 / Walmart hierarchical retail sales.

Bottom level is 3,049 items x 10 stores = 30,490 series of 1,941 daily
observations. The file is a complete rectangle with explicit zeros, so there are
no absent rows to reindex; the only absence M5 encodes is the leading run of
zeros before an item was stocked, which we blank out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Dataset, mask_leading_absence
from .download import RAW_DIR, kaggle_competition

SLUG = "m5-forecasting-accuracy"

# Which static columns define a series at each aggregation level.
LEVELS = {
    "item_store": ["item_id", "store_id"],
    "item": ["item_id"],
    "dept_store": ["dept_id", "store_id"],
    "cat_store": ["cat_id", "store_id"],
    "store": ["store_id"],
    "state": ["state_id"],
}


def load(level: str = "item_store", raw_dir: Path | None = None) -> Dataset:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {sorted(LEVELS)}")

    raw = raw_dir or (RAW_DIR / "m5")
    kaggle_competition(SLUG, raw)

    calendar = pd.read_csv(raw / "calendar.csv", usecols=["d", "date"])
    day_to_date = dict(zip(calendar["d"], pd.to_datetime(calendar["date"]), strict=True))

    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    keys = LEVELS[level]
    sales["series_id"] = sales[keys].agg("_".join, axis=1)

    static = (
        sales[["series_id", *id_cols]]
        .drop_duplicates("series_id")
        .set_index("series_id")
    )
    # Attributes that vary within an aggregated series are no longer meaningful.
    # Checked against `sales`, not `static`: static has already been deduplicated
    # down to one arbitrary row per series, which hides the variation.
    for col in id_cols:
        if col in keys:
            continue
        varies = sales.groupby("series_id", observed=True)[col].nunique() > 1
        static.loc[varies[varies].index, col] = pd.NA
    static = static.astype("category")

    panel = sales.groupby("series_id", observed=True)[day_cols].sum()
    panel = (
        panel.stack()
        .rename("y")
        .reset_index()
        .rename(columns={"level_1": "d"})
    )
    panel["ds"] = panel["d"].map(day_to_date)
    panel["y"] = panel["y"].astype("float32")
    panel["series_id"] = panel["series_id"].astype("category")
    panel = panel[["series_id", "ds", "y"]].sort_values(["series_id", "ds"])

    panel = mask_leading_absence(panel)
    static = static.loc[static.index.isin(panel["series_id"].unique())]
    static.index.name = "series_id"

    return Dataset(name=f"m5_{level}", panel=panel, static=static)
