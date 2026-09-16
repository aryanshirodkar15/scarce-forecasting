"""Statistical baselines, via statsforecast.

Croston and TSB are here because the intermittent quadrants are where the
benchmark expects classical methods to hold their ground longest; they encode
structural assumptions that compensate for exactly the data scarcity under study.
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    TSB,
    AutoARIMA,
    AutoETS,
    CrostonOptimized,
    SeasonalNaive,
)

from .prepare import impute

SEASON_LENGTH = 7


class StatsForecastModel:
    """Thin adapter. statsforecast fits and forecasts in one call, so `fit` only
    stores the window and the work happens in `predict`."""

    name = "statsforecast"

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._train: pd.DataFrame | None = None

    def _model(self):
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, static: pd.DataFrame | None = None) -> None:
        self._train = impute(train)

    def predict(self, targets: pd.DataFrame) -> pd.DataFrame:
        if self._train is None:
            raise RuntimeError(f"{self.name}: predict called before fit")

        horizon = targets["ds"].nunique()
        frame = self._train.rename(columns={"series_id": "unique_id"})[["unique_id", "ds", "y"]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = StatsForecast(models=[self._model()], freq="D", n_jobs=1)
            forecast = engine.forecast(df=frame, h=horizon)

        column = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]
        forecast = forecast.rename(columns={"unique_id": "series_id", column: "y"})
        return targets[["series_id", "ds"]].merge(
            forecast[["series_id", "ds", "y"]], on=["series_id", "ds"], how="left"
        )


class SeasonalNaiveForecaster(StatsForecastModel):
    name = "seasonal_naive"

    def _model(self):
        return SeasonalNaive(season_length=SEASON_LENGTH)


class ETSForecaster(StatsForecastModel):
    name = "ets"

    def _model(self):
        return AutoETS(season_length=SEASON_LENGTH)


class ARIMAForecaster(StatsForecastModel):
    name = "auto_arima"

    def _model(self):
        return AutoARIMA(season_length=SEASON_LENGTH)


class CrostonForecaster(StatsForecastModel):
    name = "croston"

    def _model(self):
        return CrostonOptimized()


class TSBForecaster(StatsForecastModel):
    name = "tsb"

    def _model(self):
        return TSB(
            alpha_d=self._kwargs.get("alpha_d", 0.2),
            alpha_p=self._kwargs.get("alpha_p", 0.2),
        )


CLASSICAL = {
    m.name: m
    for m in (
        SeasonalNaiveForecaster,
        ETSForecaster,
        ARIMAForecaster,
        CrostonForecaster,
        TSBForecaster,
    )
}
