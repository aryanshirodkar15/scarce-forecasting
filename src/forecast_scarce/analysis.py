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

import re
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


# Attribution stability.
#
# The question is whether the features a model relies on stay put as data gets
# scarce. Two kinds of comparison: across seeds within one cell configuration,
# which measures how much of a ranking is noise, and across regimes, which
# measures whether scarcity changes what the model attends to.

_WINDOWED = re.compile(r"^(?:lag|mean|std|zero_rate)_(\d+)$")
TOP_K = 10


def feature_lookback(name: str) -> int:
    """Days of history a feature needs before it can be anything but NaN."""
    match = _WINDOWED.match(name)
    return int(match.group(1)) if match else 0


def available_at(features, history_days: int) -> list[str]:
    return [f for f in features if feature_lookback(f) <= history_days]


def _ranking(group: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP per feature, averaged over whatever the group spans."""
    return group.groupby("feature")["mean_abs_shap"].mean()


def _agreement(a: pd.Series, b: pd.Series, features: list[str]) -> tuple[float, float]:
    """Spearman correlation of two rankings and Jaccard overlap of their top k."""
    a, b = a.reindex(features).fillna(0.0), b.reindex(features).fillna(0.0)
    if len(features) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan"), float("nan")

    rho = float(sps.spearmanr(a, b).statistic)
    k = min(TOP_K, len(features))
    top_a, top_b = set(a.nlargest(k).index), set(b.nlargest(k).index)
    return rho, len(top_a & top_b) / len(top_a | top_b)


def seed_stability(shap: pd.DataFrame) -> pd.DataFrame:
    """How much two seeds of the same cell agree on what matters.

    Only features the history can support are compared. A feature that is NaN
    everywhere has an attribution near zero by construction, and including it
    would inflate agreement with ties that carry no information.
    """
    keys = ["dataset", "history_days", "n_series", "missing_rate", "missing_pattern", "model"]
    rows = []
    for config, group in shap.groupby(keys, observed=True):
        history = dict(zip(keys, config, strict=True))["history_days"]
        features = available_at(group["feature"].unique(), history)
        rankings = {seed: _ranking(g) for seed, g in group.groupby("seed")}

        pairs = [
            _agreement(rankings[a], rankings[b], features)
            for a, b in combinations(sorted(rankings), 2)
        ]
        if not pairs:
            continue
        rhos, jaccards = zip(*pairs, strict=True)
        rows.append(
            {
                **dict(zip(keys, config, strict=True)),
                "rho": float(np.nanmean(rhos)),
                "top_k_jaccard": float(np.nanmean(jaccards)),
                "n_pairs": len(pairs),
                "n_features": len(features),
            }
        )
    return pd.DataFrame(rows)


def regime_stability(shap: pd.DataFrame, axis: str = "history_days") -> pd.DataFrame:
    """Agreement between each regime's ranking and the richest regime's.

    Along history, the reference is the longest history; along missingness, it
    is the clean panel. Rankings are compared only over features available on
    both sides. Comparing a 60-day ranking against a 730-day ranking over all
    features would report the long lags "dropping out" at 60 days, which is
    true by construction and says nothing about the model.
    """
    if axis == "history_days":
        keys = ["dataset", "n_series", "missing_rate", "missing_pattern", "model"]
        regime = shap["history_days"]
    elif axis == "missingness":
        keys = ["dataset", "history_days", "n_series", "model"]
        regime = list(zip(shap["missing_rate"], shap["missing_pattern"], strict=True))
    else:
        raise ValueError(f"unknown axis {axis!r}, expected 'history_days' or 'missingness'")

    frame = shap.assign(_regime=regime)
    rows = []
    for config, group in frame.groupby(keys, observed=True):
        regimes = {r: g for r, g in group.groupby("_regime")}
        reference = max(regimes) if axis == "history_days" else (0.0, "none")
        if reference not in regimes:
            continue

        ref_ranking = _ranking(regimes[reference])
        for label, g in regimes.items():
            if label == reference:
                continue
            shortest = min(g["history_days"].min(), regimes[reference]["history_days"].min())
            features = available_at(g["feature"].unique(), shortest)
            rho, jaccard = _agreement(_ranking(g), ref_ranking, features)
            rows.append(
                {
                    **dict(zip(keys, config, strict=True)),
                    "regime": label,
                    "reference": reference,
                    "rho": rho,
                    "top_k_jaccard": jaccard,
                    "n_features": len(features),
                }
            )
    return pd.DataFrame(rows)
