"""Feature construction for the tabular models.

Everything is anchored at the forecast origin rather than at the target date,
with the horizon index carried as its own feature. That keeps short lags usable
at a 28-day horizon, needs one model instead of one per horizon, avoids the error
compounding of recursive forecasting, and matches how the deep models produce a
28-step forecast natively, so the comparison is like for like.

The feature set is identical in every cell of the grid. A window too short to
support a lag yields NaN, which LightGBM routes natively and which costs a
zero-gain split. Letting the feature set shrink with history instead would make
the 60-day and 730-day models different models, so part of any performance gap
would be the features rather than the data, and SHAP rankings would no longer be
comparable across regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 28, 91, 182, 364)
WINDOWS = (7, 14, 28, 91, 182, 364)


def build(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: pd.Timestamp,
    static: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (series, target date), all features known as of `origin`.

    `history` must already be restricted to observations strictly before the
    origin; this function does not trim it for you, and passing anything later
    would leak the answer into the features.
    """
    if not history.empty and history["ds"].max() >= origin:
        raise ValueError(
            f"history reaches {history['ds'].max().date()} but the origin is "
            f"{origin.date()}; features would leak future observations"
        )

    per_series = {
        sid: g.sort_values("ds")
        for sid, g in history.groupby("series_id", observed=True)
    }

    rows = targets[["series_id", "ds"]].copy()
    rows["horizon"] = (rows["ds"] - origin).dt.days + 1

    stats = pd.DataFrame(
        [_series_features(per_series.get(sid)) for sid in rows["series_id"]],
        index=rows.index,
    )
    out = pd.concat([rows, stats], axis=1)

    for name, values in _calendar(out["ds"]).items():
        out[name] = values

    if static is not None and not static.empty:
        out = out.merge(
            static.reset_index(), on="series_id", how="left", suffixes=("", "_static")
        )
    return out


def _series_features(history: pd.DataFrame | None) -> dict[str, float]:
    if history is None or history.empty:
        return _empty_features()

    y = history["y"].to_numpy(dtype="float64")
    observed = y[~np.isnan(y)]
    features: dict[str, float] = {}

    # Lags count back from the origin, so lag_1 is the most recent day before it.
    for lag in LAGS:
        features[f"lag_{lag}"] = y[-lag] if len(y) >= lag else np.nan

    for window in WINDOWS:
        tail = y[-window:] if len(y) >= window else np.array([])
        seen = tail[~np.isnan(tail)] if tail.size else tail
        # A window is only reported when the history actually spans it, otherwise
        # a 60-day cell would report a "364-day mean" computed from 60 days.
        features[f"mean_{window}"] = seen.mean() if seen.size else np.nan
        features[f"std_{window}"] = seen.std(ddof=1) if seen.size > 1 else np.nan
        features[f"zero_rate_{window}"] = (seen == 0).mean() if seen.size else np.nan

    features["days_since_demand"] = _days_since_demand(y)
    features["n_observed"] = float(observed.size)
    features["overall_mean"] = observed.mean() if observed.size else np.nan
    return features


def _days_since_demand(y: np.ndarray) -> float:
    """Distance from the origin back to the last strictly positive observation."""
    positive = np.flatnonzero(y > 0)
    if positive.size == 0:
        return np.nan
    return float(len(y) - positive[-1])


def _empty_features() -> dict[str, float]:
    features = {f"lag_{lag}": np.nan for lag in LAGS}
    for window in WINDOWS:
        features[f"mean_{window}"] = np.nan
        features[f"std_{window}"] = np.nan
        features[f"zero_rate_{window}"] = np.nan
    features["days_since_demand"] = np.nan
    features["n_observed"] = 0.0
    features["overall_mean"] = np.nan
    return features


def _calendar(ds: pd.Series) -> dict[str, np.ndarray]:
    """Target-date calendar attributes, which are genuinely known in advance."""
    return {
        "dayofweek": ds.dt.dayofweek.to_numpy(),
        "dayofmonth": ds.dt.day.to_numpy(),
        "month": ds.dt.month.to_numpy(),
        "weekofyear": ds.dt.isocalendar().week.astype("int64").to_numpy(),
        "is_weekend": (ds.dt.dayofweek >= 5).astype("int64").to_numpy(),
    }


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs, excluding the identifiers and the target."""
    return [c for c in frame.columns if c not in ("series_id", "ds", "y")]
