"""Chicago CTA rail station daily entries.

The non-retail control. Around 145 stations with daily counts back to 2001,
served by an open Socrata endpoint with no authentication. Unlike a single
national ridership aggregate this is a genuine panel, so the scarcity protocol
applies to it unchanged. It also carries a real structural break in 2020, which
is a useful stress test for models that assume a stable level.
"""

from __future__ import annotations

from pathlib import Path

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

    panel = rides[["series_id", "ds", "y"]]
    # Stations open and close over a 20+ year window, so gaps here are genuine
    # absence rather than zero ridership. No leading-zero mask: a station that
    # opens mid-panel simply has no rows before it opened.
    panel = to_daily_grid(panel, panel["ds"].min(), panel["ds"].max())

    static.index.name = "series_id"
    return Dataset(name="cta", panel=panel, static=static)
