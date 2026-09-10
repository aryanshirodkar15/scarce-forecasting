"""Forecast accuracy metrics.

Every metric here ignores unobserved actuals rather than treating them as zeros.
The scarcity protocol injects gaps across the whole window including the
evaluation period, so a metric that scored a NaN target as a zero would reward
whichever model degrades most gracefully on missing inputs, which is not the
quantity anyone wants measured.

MAPE is deliberately absent. It is undefined wherever demand is zero, and zero
demand is the normal case across most of the intermittent regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEASONALITY = 7


def _paired(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Actual/forecast pairs where both are present."""
    actual = np.asarray(y_true, dtype="float64")
    forecast = np.asarray(y_pred, dtype="float64")
    if actual.shape != forecast.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {forecast.shape}")
    keep = ~(np.isnan(actual) | np.isnan(forecast))
    return actual[keep], forecast[keep]


def seasonal_diffs(y_train, seasonality: int = DEFAULT_SEASONALITY) -> np.ndarray:
    """In-sample seasonal differences over pairs where both ends are observed.

    Gaps in the training series remove the pairs that straddle them instead of
    contributing a spurious difference against a NaN.
    """
    train = np.asarray(y_train, dtype="float64")
    if seasonality < 1:
        raise ValueError(f"seasonality must be at least 1, got {seasonality}")
    if len(train) <= seasonality:
        return np.empty(0)

    later, earlier = train[seasonality:], train[:-seasonality]
    keep = ~(np.isnan(later) | np.isnan(earlier))
    return later[keep] - earlier[keep]


def mase(y_true, y_pred, y_train, seasonality: int = DEFAULT_SEASONALITY) -> float:
    """Mean absolute scaled error, scaled by in-sample seasonal naive MAE."""
    actual, forecast = _paired(y_true, y_pred)
    diffs = seasonal_diffs(y_train, seasonality)
    if actual.size == 0 or diffs.size == 0:
        return float("nan")

    scale = np.mean(np.abs(diffs))
    if scale == 0:
        # A training window with no seasonal variation at all gives no yardstick.
        # Returning NaN keeps it out of the aggregate rather than dividing by zero.
        return float("nan")
    return float(np.mean(np.abs(actual - forecast)) / scale)


def rmsse(y_true, y_pred, y_train, seasonality: int = DEFAULT_SEASONALITY) -> float:
    """Root mean squared scaled error."""
    actual, forecast = _paired(y_true, y_pred)
    diffs = seasonal_diffs(y_train, seasonality)
    if actual.size == 0 or diffs.size == 0:
        return float("nan")

    scale = np.mean(diffs**2)
    if scale == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / scale))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: total error over total demand."""
    actual, forecast = _paired(y_true, y_pred)
    if actual.size == 0:
        return float("nan")

    total = np.sum(np.abs(actual))
    if total == 0:
        # No demand at all in the evaluation window, so there is nothing to be
        # wrong about in proportion to.
        return float("nan")
    return float(np.sum(np.abs(actual - forecast)) / total)


def pinball(y_true, y_pred, quantile: float) -> float:
    """Pinball (quantile) loss at a single quantile."""
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    actual, forecast = _paired(y_true, y_pred)
    if actual.size == 0:
        return float("nan")

    delta = actual - forecast
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def evaluate(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    train: pd.DataFrame,
    seasonality: int = DEFAULT_SEASONALITY,
) -> pd.DataFrame:
    """Per-series metrics for one fold, in tidy long format.

    All three frames are long panels with series_id, ds and y; `forecast` carries
    predictions in its y column.
    """
    merged = actual.merge(forecast, on=["series_id", "ds"], suffixes=("", "_pred"))
    train_by_series = {
        sid: g["y"].to_numpy() for sid, g in train.groupby("series_id", observed=True)
    }

    rows = []
    for series_id, group in merged.groupby("series_id", observed=True):
        history = train_by_series.get(series_id, np.empty(0))
        y, yhat = group["y"].to_numpy(), group["y_pred"].to_numpy()
        rows.append(
            {
                "series_id": series_id,
                "n_scored": int((~np.isnan(y) & ~np.isnan(yhat)).sum()),
                "mase": mase(y, yhat, history, seasonality),
                "rmsse": rmsse(y, yhat, history, seasonality),
                "wape": wape(y, yhat),
            }
        )
    return pd.DataFrame(rows)
