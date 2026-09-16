"""Forecasters behind one interface.

Every model takes the same training window and produces a value for every
(series, date) pair in the evaluation window. `fit` and `predict` are separate so
that fitting cost can be measured on its own, though the statistical models do
their work lazily since statsforecast fits and forecasts in a single call.

All models receive the same imputed training series. Classical and deep models
require a gap-free regular grid, and giving each model its own missing-data
handling would confound forecast quality with imputation quality. Equalizing it
means the benchmark measures the former.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .classical import CLASSICAL
from .gbm import LightGBMForecaster


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, static: pd.DataFrame | None = None) -> None: ...

    def predict(self, targets: pd.DataFrame) -> pd.DataFrame: ...


MODELS: dict[str, type] = {**CLASSICAL, "lightgbm": LightGBMForecaster}


def available() -> list[str]:
    return list(MODELS)


def build(name: str, **kwargs) -> Forecaster:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["Forecaster", "MODELS", "available", "build", "LightGBMForecaster"]
