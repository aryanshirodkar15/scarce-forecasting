"""Canonical panel representation shared by every loader.

The panel is long format: one row per (series_id, ds). `y` is float32 and uses
NaN for "this series had no observable demand process on this date" as opposed
to 0.0 for "the demand process was observed and produced zero". Keeping those
apart is load-bearing: the scarcity protocol injects its own gaps later, and
intermittency statistics computed over pre-launch zeros are meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_COLUMNS = ("series_id", "ds", "y")


@dataclass
class Dataset:
    name: str
    panel: pd.DataFrame
    static: pd.DataFrame
    freq: str = "D"

    def __post_init__(self) -> None:
        missing = set(PANEL_COLUMNS) - set(self.panel.columns)
        if missing:
            raise ValueError(f"{self.name}: panel is missing columns {sorted(missing)}")
        if not pd.api.types.is_datetime64_any_dtype(self.panel["ds"]):
            raise ValueError(f"{self.name}: ds must be datetime64")
        if self.static.index.name != "series_id":
            raise ValueError(f"{self.name}: static must be indexed by series_id")

        panel_ids = set(self.panel["series_id"].unique())
        static_ids = set(self.static.index)
        if panel_ids - static_ids:
            n = len(panel_ids - static_ids)
            raise ValueError(f"{self.name}: {n} series in panel have no static row")

    @property
    def n_series(self) -> int:
        return self.panel["series_id"].nunique()

    @property
    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.panel["ds"].min(), self.panel["ds"].max()

    def summary(self) -> dict[str, object]:
        y = self.panel["y"].to_numpy()
        observed = ~np.isnan(y)
        start, end = self.span
        return {
            "name": self.name,
            "n_series": self.n_series,
            "n_rows": len(self.panel),
            "start": str(start.date()),
            "end": str(end.date()),
            "pct_absent": round(float(100 * (1 - observed.mean())), 2),
            "pct_zero_when_observed": round(float((y[observed] == 0).mean() * 100), 2),
        }

    def to_parquet(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.panel.to_parquet(directory / "panel.parquet", index=False)
        self.static.to_parquet(directory / "static.parquet")

    @classmethod
    def from_parquet(cls, directory: Path, name: str, freq: str = "D") -> Dataset:
        return cls(
            name=name,
            panel=pd.read_parquet(directory / "panel.parquet"),
            static=pd.read_parquet(directory / "static.parquet"),
            freq=freq,
        )


def to_daily_grid(long: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex a ragged long frame onto a contiguous daily grid, NaN for absent dates.

    Used by sources that simply omit rows rather than recording an explicit zero.
    """
    clashes = long.duplicated(subset=["series_id", "ds"]).sum()
    if clashes:
        raise ValueError(
            f"{clashes} duplicate (series_id, ds) rows reached to_daily_grid. "
            f"Resolve them in the loader: a grid cannot hold two values for one day."
        )

    dates = pd.date_range(start, end, freq="D")
    ids = pd.Index(long["series_id"].unique(), name="series_id")
    full = pd.MultiIndex.from_product([ids, dates], names=["series_id", "ds"])

    out = (
        long.set_index(["series_id", "ds"])["y"]
        .reindex(full)
        .astype("float32")
        .reset_index()
    )
    out["series_id"] = out["series_id"].astype("category")
    return out


def mask_leading_absence(panel: pd.DataFrame) -> pd.DataFrame:
    """Blank out each series before its first strictly positive observation.

    A run of zeros at the head of a retail series is almost always a product that
    was not stocked yet, not a product nobody bought. Series that are positive
    nowhere are dropped entirely, since they carry no demand signal at all.
    """
    positive = panel.loc[panel["y"] > 0]
    first_sale = positive.groupby("series_id", observed=True)["ds"].min()

    alive = panel["series_id"].map(first_sale)
    keep = alive.notna()
    dropped = panel.loc[~keep, "series_id"].nunique()
    if dropped:
        print(f"  dropped {dropped} series with no positive observation")

    out = panel.loc[keep].copy()
    out.loc[out["ds"] < alive[keep], "y"] = np.nan
    return out.reset_index(drop=True)
