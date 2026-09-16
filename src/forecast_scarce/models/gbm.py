"""LightGBM over the origin-anchored feature table.

One model per cell rather than one per horizon: features are computed as of the
forecast origin and the distance to the target is passed as the `horizon`
feature. The same feature columns appear at every history length, with NaN where
a window is too short, which LightGBM routes natively.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import build, feature_columns
from .prepare import impute

DEFAULT_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}
DEFAULT_ROUNDS = 300


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = DEFAULT_ROUNDS,
        seed: int = 0,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
        self.num_boost_round = num_boost_round
        self._booster: lgb.Booster | None = None
        self._columns: list[str] | None = None
        self._history: pd.DataFrame | None = None
        self._static: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame, static: pd.DataFrame | None = None) -> None:
        self._history = impute(train)
        self._static = static

        table = self._training_table(self._history, static)
        if table.empty:
            raise ValueError("lightgbm: no usable training rows after windowing")

        self._columns = feature_columns(table)
        dataset = lgb.Dataset(
            table[self._columns],
            label=table["y"],
            categorical_feature=self._categoricals(table),
            free_raw_data=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._booster = lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

    def predict(self, targets: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None or self._columns is None:
            raise RuntimeError("lightgbm: predict called before fit")

        origin = self._history["ds"].max() + pd.Timedelta(days=1)
        table = build(self._history, targets, origin, self._static)
        table = self._align(table)

        predicted = self._booster.predict(table[self._columns])
        out = targets[["series_id", "ds"]].copy()
        out["y"] = np.clip(predicted, 0, None)  # demand cannot be negative
        return out

    def _training_table(self, history: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
        """Rolling origins inside the training window, so the model sees the same
        shape of problem it will face at prediction time."""
        dates = np.sort(history["ds"].unique())
        horizon = 28
        # Step by the horizon so training origins do not overlap each other.
        origins = dates[len(dates) % horizon :: horizon][1:]

        frames = []
        for origin in origins:
            origin = pd.Timestamp(origin)
            past = history.loc[history["ds"] < origin]
            future = history.loc[
                (history["ds"] >= origin) & (history["ds"] < origin + pd.Timedelta(days=horizon))
            ]
            if past.empty or future.empty:
                continue
            table = build(past, future[["series_id", "ds"]], origin, static)
            table["y"] = future["y"].to_numpy()
            frames.append(table)

        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, ignore_index=True)
        return table.loc[table["y"].notna()]

    def _categoricals(self, table: pd.DataFrame) -> list[str]:
        return [c for c in self._columns if str(table[c].dtype) in ("category", "object")]

    def _align(self, table: pd.DataFrame) -> pd.DataFrame:
        """Prediction-time columns must match training exactly, in order."""
        for column in self._columns:
            if column not in table.columns:
                table[column] = np.nan
        return table
