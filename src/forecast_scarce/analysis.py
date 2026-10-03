"""Statistical comparison of forecasts across the grid.

Three pieces. Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample
correction tests whether two models differ on a given cell. A cluster bootstrap
over series puts an interval on the size of that difference, resampling whole
series rather than individual observations because a series' folds are
correlated with each other. Benjamini-Hochberg controls the false discovery rate
across the grid, which matters here: the full grid runs thousands of pairwise
tests, and at 5 percent uncorrected a few hundred would read as significant by
chance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class Comparison:
    model_a: str
    model_b: str
    mean_difference: float
    dm_statistic: float
    p_value: float
    n_obs: int


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """DM statistic and two-sided p-value, Harvey-Leybourne-Newbold corrected.

    The HAC variance uses lags up to horizon-1, since an h-step forecast error
    series is autocorrelated up to order h-1 by construction. Without that, the
    variance is understated and almost everything looks significant.
    """
    d = np.asarray(loss_a, dtype="float64") - np.asarray(loss_b, dtype="float64")
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2 or horizon < 1:
        return float("nan"), float("nan")

    mean = d.mean()
    centred = d - mean
    gamma0 = np.sum(centred**2) / n
    gammas = [np.sum(centred[k:] * centred[:-k]) / n for k in range(1, min(horizon, n))]
    variance = (gamma0 + 2 * sum(gammas)) / n

    if variance <= 0:
        # Identical forecasts give a degenerate differential; that is not evidence
        # of a difference, so report the null rather than dividing by zero. A
        # constant nonzero gap is the opposite case: one model is uniformly worse
        # on every date, and the statistic genuinely diverges.
        if np.isclose(mean, 0):
            return 0.0, 1.0
        return float(np.inf * np.sign(mean)), 0.0

    statistic = mean / np.sqrt(variance)

    adjustment = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if adjustment <= 0:
        return float("nan"), float("nan")
    statistic *= np.sqrt(adjustment)

    p_value = 2 * (1 - sps.t.cdf(abs(statistic), df=n - 1))
    return float(statistic), float(p_value)


def benjamini_hochberg(p_values, alpha: float = 0.05) -> pd.DataFrame:
    """Adjusted p-values and rejection flags controlling the false discovery rate."""
    p = np.asarray(p_values, dtype="float64")
    out = pd.DataFrame({"p_value": p, "p_adjusted": np.nan, "reject": False})

    finite = np.flatnonzero(~np.isnan(p))
    if finite.size == 0:
        return out

    ordered = finite[np.argsort(p[finite])]
    m = finite.size
    ranks = np.arange(1, m + 1)

    adjusted = p[ordered] * m / ranks
    # Enforce monotonicity from the largest p-value downward.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out.loc[ordered, "p_adjusted"] = adjusted
    out.loc[ordered, "reject"] = adjusted <= alpha
    return out


def bootstrap_difference(
    per_series: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "mase",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval for mean(metric_a - metric_b).

    Resamples series, not rows. A series contributes several folds whose errors
    are correlated, and resampling rows would treat those as independent and
    produce intervals that are too narrow.
    """
    wide = (
        per_series.loc[per_series["model"].isin([model_a, model_b])]
        .pivot_table(index="series_id", columns="model", values=metric, observed=True)
        .dropna()
    )
    if wide.empty or model_a not in wide or model_b not in wide:
        return float("nan"), float("nan"), float("nan")

    difference = (wide[model_a] - wide[model_b]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, difference.size, size=(n_boot, difference.size))
    means = difference[draws].mean(axis=1)

    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(difference.mean()), float(lower), float(upper)


def compare_cell(daily: pd.DataFrame, horizon: int = 28) -> list[Comparison]:
    """Pairwise DM tests between every model on one cell."""
    wide = daily.pivot_table(index="ds", columns="model", values="scaled_error", observed=True)
    models = sorted(wide.columns)

    out = []
    for a, b in combinations(models, 2):
        pair = wide[[a, b]].dropna()
        statistic, p_value = diebold_mariano(pair[a].to_numpy(), pair[b].to_numpy(), horizon)
        out.append(
            Comparison(
                model_a=a,
                model_b=b,
                mean_difference=float((pair[a] - pair[b]).mean()),
                dm_statistic=statistic,
                p_value=p_value,
                n_obs=len(pair),
            )
        )
    return out


CELL_KEYS = ["dataset", "history_days", "n_series", "missing_rate", "missing_pattern", "seed"]


def compare_grid(daily: pd.DataFrame, horizon: int = 28, alpha: float = 0.05) -> pd.DataFrame:
    """Every pairwise test on every cell, with the FDR controlled across all of them."""
    rows = []
    for keys, group in daily.groupby(CELL_KEYS, observed=True):
        for comparison in compare_cell(group, horizon):
            rows.append({**dict(zip(CELL_KEYS, keys, strict=True)), **vars(comparison)})

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    corrected = benjamini_hochberg(table["p_value"], alpha=alpha)
    table["p_adjusted"] = corrected["p_adjusted"].to_numpy()
    table["significant"] = corrected["reject"].to_numpy()
    return table


def summarise(results: pd.DataFrame, metric: str = "mase") -> pd.DataFrame:
    """Mean metric per cell and model, with the series count behind each number."""
    return (
        results.dropna(subset=[metric])
        .groupby([*CELL_KEYS, "model", "quadrant"], observed=True)
        .agg(**{metric: (metric, "mean"), "n_series_scored": ("series_id", "nunique")})
        .reset_index()
    )
