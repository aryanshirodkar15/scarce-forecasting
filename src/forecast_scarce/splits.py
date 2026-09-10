"""Rolling-origin folds over a shared evaluation period.

Every cell of the scarcity grid is scored on the same calendar days for the same
series. `history_days` controls only the width of a training window that slides
forward with the origin, so the amount of training data is held at exactly
`history_days` in every fold rather than growing as the origin advances. That is
what makes a metric difference between a 60-day cell and a 730-day cell
attributable to training volume rather than to a different test set or a
different effective sample size.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_HORIZON = 28


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive, equals the origin
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive

    @property
    def origin(self) -> pd.Timestamp:
        return self.train_end


def rolling_origin(
    test_start: pd.Timestamp,
    history_days: int,
    test_days: int,
    horizon: int = DEFAULT_HORIZON,
) -> list[Fold]:
    """Non-overlapping folds tiling the shared test period."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if test_days % horizon:
        raise ValueError(
            f"test_days {test_days} is not divisible by horizon {horizon}; "
            f"folds would not tile the evaluation period evenly"
        )

    test_start = pd.Timestamp(test_start)
    folds = []
    for i in range(test_days // horizon):
        origin = test_start + pd.Timedelta(days=i * horizon)
        folds.append(
            Fold(
                index=i,
                train_start=origin - pd.Timedelta(days=history_days),
                train_end=origin,
                test_start=origin,
                test_end=origin + pd.Timedelta(days=horizon),
            )
        )
    return folds


def split(panel: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test frames for one fold.

    Note that from the second fold onward the training window legitimately
    contains days that were the previous fold's test targets. That is what
    rolling-origin evaluation means: at each origin the forecaster may use every
    actual observed before that origin. No fold ever sees its own targets.
    """
    ds = panel["ds"]
    train = panel.loc[(ds >= fold.train_start) & (ds < fold.train_end)]
    test = panel.loc[(ds >= fold.test_start) & (ds < fold.test_end)]
    return train, test
