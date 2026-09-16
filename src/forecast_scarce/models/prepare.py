"""Shared preprocessing applied identically to every model's training input."""

from __future__ import annotations

import pandas as pd


def impute(train: pd.DataFrame) -> pd.DataFrame:
    """Fill interior gaps by linear interpolation, per series.

    Leading gaps are dropped rather than back-filled: before a series' first
    observation there is nothing to interpolate from, and inventing a value there
    would fabricate history that the scarcity protocol deliberately withheld.
    Trailing gaps are carried forward, since the most recent observation is the
    best available statement about the present.
    """
    out = []
    for _, group in train.groupby("series_id", observed=True):
        group = group.sort_values("ds").copy()
        first = group["y"].first_valid_index()
        if first is None:
            continue  # nothing observed at all; the series cannot be fitted
        group = group.loc[first:]
        group["y"] = group["y"].interpolate(method="linear", limit_direction="forward")
        out.append(group)

    if not out:
        return train.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def imputation_rate(train: pd.DataFrame) -> float:
    """Share of the training window that imputation had to invent."""
    total = len(train)
    return float(train["y"].isna().sum() / total) if total else 0.0
