"""Chicago CTA rail station daily entries.

The non-retail control. Around 145 stations with daily counts back to 2001,
served by an open Socrata endpoint with no authentication. Unlike a single
national ridership aggregate this is a genuine panel, so the scarcity protocol
applies to it unchanged. It also carries a real structural break in 2020, which
is a useful stress test for models that assume a stable level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Dataset, to_daily_grid
from .download import RAW_DIR, socrata

DOMAIN = "data.cityofchicago.org"
RESOURCE = "5neh-572f"


def load(raw_dir: Path | None = None) -> Dataset:
    raw = raw_dir or (RAW_DIR / "cta")
    path = socrata(DOMAIN, RESOURCE, raw / "l_station_entries.ndjson")

    rides = pd.read_json(path, lines=True)
    rides["ds"] = pd.to_datetime(rides["date"])
    rides["series_id"] = rides["station_id"].astype(str)
    rides["y"] = rides["rides"].astype("float32")

    static = (
        rides[["series_id", "stationname"]]
        .drop_duplicates("series_id")
        .set_index("series_id")
        .astype("category")
    )

    panel = _resolve_conflicting_days(rides[["series_id", "ds", "y"]])
    # Stations open and close over a 20+ year window, so gaps here are genuine
    # absence rather than zero ridership. No leading-zero mask: a station that
    # opens mid-panel simply has no rows before it opened.
    panel = to_daily_grid(panel, panel["ds"].min(), panel["ds"].max())

    static.index.name = "series_id"
    return Dataset(name="cta", panel=panel, static=static)


def _resolve_conflicting_days(rides: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated station-days, blanking the ones that disagree.

    CTA re-reported roughly 600 station-days between 2011-07-01 and 2011-08-10
    with two different ride counts for the same day. There is no basis for
    preferring either figure, and averaging them would invent an observation
    that was never recorded, so the affected days are marked unobserved. It is
    under 0.1 percent of the panel.
    """
    rides = rides.drop_duplicates(subset=["series_id", "ds", "y"])

    repeated = rides.duplicated(subset=["series_id", "ds"], keep=False)
    if repeated.any():
        n_days = rides.loc[repeated].groupby(["series_id", "ds"], observed=True).ngroups
        print(f"  blanked {n_days} station-days with conflicting counts")
        rides = rides.copy()
        rides.loc[repeated, "y"] = np.nan
        rides = rides.drop_duplicates(subset=["series_id", "ds"])

    return rides
