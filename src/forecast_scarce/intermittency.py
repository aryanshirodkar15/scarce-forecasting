"""Syntetos-Boylan classification of demand patterns.

ADI is the average inter-demand interval, periods divided by demand
occurrences. CV2 is the squared coefficient of variation of the demand sizes.
The 1.32 / 0.49 cutoffs are the standard ones from Syntetos, Boylan and Croston
(2005); using the literature's boundaries rather than data-driven quantiles is
what keeps quadrants comparable across three very different datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49

# Below this many demand occurrences CV2 is not an estimate of anything.
MIN_NONZERO = 5

QUADRANTS = ("smooth", "erratic", "intermittent", "lumpy", "insufficient")


def classify(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-series ADI, CV2 and quadrant, indexed by series_id.

    NaN periods are dropped rather than counted as intervals. That is deliberate:
    if unobserved periods stretched the measured interval, injecting missingness
    would push series across the ADI cutoff and the missingness axis of the grid
    would silently turn into an intermittency axis.

    Only strictly positive values count as demand occurrences. Favorita encodes
    returns as negative sales, and a return is not a demand.
    """
    observed = panel.dropna(subset=["y"])
    n_observed = observed.groupby("series_id", observed=True)["y"].size()

    positive = observed.loc[observed["y"] > 0]
    sizes = positive.groupby("series_id", observed=True)["y"]

    n_nonzero = sizes.size().reindex(n_observed.index, fill_value=0)
    mean_size = sizes.mean().reindex(n_observed.index)
    # ddof=1 because these are small samples and the bias matters in exactly the
    # regime this benchmark studies.
    std_size = sizes.std(ddof=1).reindex(n_observed.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        adi = n_observed / n_nonzero
        cv2 = (std_size / mean_size) ** 2

    out = pd.DataFrame(
        {
            "adi": adi.astype("float64"),
            "cv2": cv2.astype("float64"),
            "n_observed": n_observed.astype("int64"),
            "n_nonzero": n_nonzero.astype("int64"),
        }
    )
    out["quadrant"] = _quadrant(out)
    out.index.name = "series_id"
    return out


def _quadrant(stats: pd.DataFrame) -> pd.Series:
    intermittent = stats["adi"] >= ADI_CUT
    erratic = stats["cv2"] >= CV2_CUT

    label = pd.Series("smooth", index=stats.index, dtype=object)
    label[~intermittent & erratic] = "erratic"
    label[intermittent & ~erratic] = "intermittent"
    label[intermittent & erratic] = "lumpy"

    unusable = (stats["n_nonzero"] < MIN_NONZERO) | stats["cv2"].isna()
    label[unusable] = "insufficient"
    return pd.Categorical(label, categories=QUADRANTS)


def quadrant_counts(stats: pd.DataFrame) -> dict[str, int]:
    counts = stats["quadrant"].value_counts()
    return {q: int(counts.get(q, 0)) for q in QUADRANTS}
