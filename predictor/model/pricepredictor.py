#!/usr/bin/python3

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from .auxdatastore import AuxDataStore
from .entsoedatastore import EntsoeDataStore
from .gaspricestore import GasPriceStore
from .marketfeaturestore import MarketFeatureStore
from .priceregion import PriceRegion
from .pricestore import PriceStore
from .weatherstore import WeatherStore

log = logging.getLogger(__name__)


class PricePredictor:
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    entsoestore: EntsoeDataStore
    marketstore: MarketFeatureStore
    auxstore: AuxDataStore
    gasstore: GasPriceStore

    traindata: pd.DataFrame | None = None

    predictor: lgb.Booster | None = None
    quantile_models: dict[str, lgb.Booster]
    spike_classifier: lgb.Booster | None
    spike_uplift_model: lgb.Booster | None
    model_version: str | None
    last_train_start: datetime | None
    last_train_end: datetime | None
    feature_columns: list[str] | None

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region, storage_dir)
        self.entsoestore = EntsoeDataStore(region, storage_dir)
        self.marketstore = MarketFeatureStore(region, storage_dir)
        self.gasstore = GasPriceStore(region, storage_dir)

        self.quantile_models = {}
        self.spike_classifier = None
        self.spike_uplift_model = None
        self.model_version = None
        self.last_train_start = None
        self.last_train_end = None
        self.feature_columns = None

    async def load_from_persistence(self):
        await asyncio.gather(
            self.weatherstore.load(),
            self.pricestore.load(),
            self.auxstore.load(),
            self.entsoestore.load(),
            self.marketstore.load(),
            self.gasstore.load(),
        )
        return self

    def last_data_update(self) -> datetime:
        return max(
            self.weatherstore.last_updated,
            self.pricestore.last_updated,
            self.entsoestore.last_updated,
            self.marketstore.last_updated,
            self.gasstore.last_updated,
        )

    def use_datastores_from(self, other: "PricePredictor"):
        assert self.region.bidding_zone_entsoe == other.region.bidding_zone_entsoe
        self.weatherstore = other.weatherstore
        self.pricestore = other.pricestore
        self.auxstore = other.auxstore
        self.entsoestore = other.entsoestore
        self.marketstore = other.marketstore
        self.gasstore = other.gasstore

    def is_trained(self) -> bool:
        return self.predictor is not None

    async def train(self, start: datetime, end: datetime):
        self.traindata = await self.prepare_dataframe(start, end, prediction_generated_at=end)
        if self.traindata is None:
            return

        train_frame = self.traindata.dropna(subset=["price"]).copy()
        if train_frame.empty:
            return

        params = self._to_numeric_features(train_frame.drop(columns=["price"]))
        output = pd.to_numeric(train_frame["price"], errors="coerce")
        valid_rows = output.notna()
        params = params.loc[valid_rows]
        output = output.loc[valid_rows]
        if params.empty:
            return

        self.feature_columns = params.columns.to_list()

        weights = self._build_training_weights(params.index, end)
        dataset = lgb.Dataset(params, label=output, weight=weights)

        self.predictor = await asyncio.to_thread(
            lgb.train,
            params=self._lgb_params(),
            train_set=dataset,
        )

        self.quantile_models = {}
        self.spike_classifier = None
        self.spike_uplift_model = None
        if self.region.use_market_features:
            for quantile_name, alpha in {"q10": 0.1, "q50": 0.5, "q90": 0.9}.items():
                self.quantile_models[quantile_name] = await asyncio.to_thread(
                    lgb.train,
                    params=self._lgb_params(objective="quantile", alpha=alpha),
                    train_set=lgb.Dataset(params, label=output, weight=weights),
                )

            spike_target = (output >= output.quantile(0.9)).astype(int)
            if spike_target.sum() >= 48 and spike_target.nunique() > 1:
                self.spike_classifier = await asyncio.to_thread(
                    lgb.train,
                    params=self._lgb_params(objective="binary"),
                    train_set=lgb.Dataset(params, label=spike_target, weight=weights),
                )

                base_prediction = pd.Series(self.predictor.predict(params), index=params.index)
                residual = output - base_prediction
                spike_rows = spike_target.astype(bool)
                if spike_rows.sum() >= 48:
                    spike_params = params.loc[spike_rows]
                    spike_weights = weights.loc[spike_rows]
                    self.spike_uplift_model = await asyncio.to_thread(
                        lgb.train,
                        params=self._lgb_params(),
                        train_set=lgb.Dataset(
                            spike_params,
                            label=residual.loc[spike_rows],
                            weight=spike_weights,
                        ),
                    )

        self.model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.last_train_start = start
        self.last_train_end = end

    async def predict(
        self,
        start: datetime,
        end: datetime,
        fill_known: bool = True,
        generated_at: datetime | None = None,
    ) -> pd.DataFrame:
        details = await self.predict_with_details(start, end, fill_known=fill_known, generated_at=generated_at)
        return details["point"]

    async def predict_with_details(
        self,
        start: datetime,
        end: datetime,
        fill_known: bool = True,
        generated_at: datetime | None = None,
    ) -> dict[str, Any]:
        assert self.is_trained() and self.predictor is not None

        generation_time = generated_at or start
        df = await self.prepare_dataframe(start, end, prediction_generated_at=generation_time)
        assert df is not None

        prices_known = df["price"].copy()
        params = self._to_numeric_features(df.drop(columns=["price"]))
        params = self._align_prediction_features(params)

        point_forecast = pd.DataFrame(index=params.index)
        point_forecast["price"] = self.predictor.predict(params)

        spike_probability = pd.Series(0.0, index=params.index)
        if self.spike_classifier is not None:
            spike_probability = pd.Series(self.spike_classifier.predict(params), index=params.index)
            if self.spike_uplift_model is not None:
                uplift = pd.Series(self.spike_uplift_model.predict(params), index=params.index)
                point_forecast["price"] = point_forecast["price"] + uplift * spike_probability.clip(0.0, 1.0)

        quantiles = pd.DataFrame(index=params.index)
        if self.quantile_models:
            for name, model in self.quantile_models.items():
                quantiles[name] = model.predict(params)
            quantiles["q50"] = quantiles["q50"] if "q50" in quantiles else point_forecast["price"]
            if {"q10", "q50", "q90"} <= set(quantiles.columns):
                quantiles["q10"] = np.minimum(quantiles["q10"], quantiles["q50"])
                quantiles["q90"] = np.maximum(quantiles["q90"], quantiles["q50"])
                point_forecast["price"] = point_forecast["price"] * 0.85 + quantiles["q50"] * 0.15

        point_forecast = self._apply_post_model_blend(df, point_forecast, spike_probability)

        if fill_known:
            point_forecast.update(prices_known)

        return {
            "point": point_forecast,
            "quantiles": quantiles,
            "features": params,
            "spike_probability": spike_probability.to_frame("spike_probability"),
        }

    def to_price_dict(self, df: pd.DataFrame) -> Dict[datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result

    async def prepare_dataframe(
        self,
        actual_start: datetime,
        end: datetime,
        prediction_generated_at: datetime | None = None,
    ) -> pd.DataFrame | None:
        lookback_days = 7 if self.region.use_market_features else 1
        generated_at = prediction_generated_at or actual_start
        start = min(
            datetime.now(timezone.utc) - timedelta(days=14),
            actual_start - timedelta(days=lookback_days),
            generated_at - timedelta(days=lookback_days),
        )

        weather, prices, auxdata = await asyncio.gather(
            self.weatherstore.get_data(start, end),
            self.pricestore.get_data(start, end),
            self.auxstore.get_data(start, end),
        )

        df = pd.concat([weather, auxdata], axis=1, sort=True)

        if self.region.use_entsoe_load_forecast:
            entsoedata = await self.entsoestore.get_data(start, end)
            if len(entsoedata) > 0:
                df = pd.concat([df, entsoedata], axis=1, sort=True)

        if self.region.use_market_features:
            marketdata = await self.marketstore.get_data(start, end)
            if len(marketdata) > 0:
                market_features = self._build_market_features(
                    marketdata,
                    prices["price"],
                    prediction_generated_at,
                )
                if not market_features.empty:
                    df = pd.concat([df, market_features], axis=1, sort=True)

        if self.region.use_de_nat_gas_price:
            gasprices = await self.gasstore.get_data(start, end)
            gasprices = gasprices.reindex(weather.index).ffill()
            df = pd.concat([df, gasprices], axis=1, sort=True)

        df = pd.concat([df, prices], axis=1, sort=True)
        df = self._to_numeric_frame(df)
        df = df[actual_start:]
        return df

    def _build_market_features(
        self,
        marketdata: pd.DataFrame,
        own_prices: pd.Series,
        prediction_generated_at: datetime | None,
    ) -> pd.DataFrame:
        market = self._to_numeric_frame(marketdata)
        features = pd.DataFrame(index=market.index)

        load = self._coalesce_columns(
            market,
            [
                "fingrid_load_forecast",
                "entsoe_load_forecast_forecasted_load",
                "entsoe_load_forecast",
            ],
        )
        if "fingrid_wind_forecast" in market.columns:
            wind = market["fingrid_wind_forecast"]
        else:
            wind = self._coalesce_columns(
                market,
                [
                    "entsoe_wind_solar_wind_onshore",
                    "entsoe_wind_solar_wind_offshore",
                ],
                combine="sum",
            )
        solar = self._coalesce_columns(
            market,
            [
                "fingrid_solar_forecast",
                "entsoe_wind_solar_solar",
            ],
        )
        generation = self._coalesce_columns(
            market,
            [
                "entsoe_generation_forecast_actual_aggregated",
                "entsoe_generation_forecast",
            ],
        )

        features["market_load_forecast"] = load
        features["market_wind_forecast"] = wind
        features["market_solar_forecast"] = solar
        features["market_generation_forecast"] = generation
        features["market_residual_load"] = load - wind - solar
        features["market_dispatchable_gap"] = generation - wind - solar

        capacity_import = self._coalesce_columns(
            market,
            [
                "fingrid_capacity_ee_fi",
            ],
        )
        features["available_import_headroom"] = capacity_import
        features["available_export_headroom"] = self._coalesce_columns(
            market,
            ["fingrid_capacity_fi_ee"],
        )

        own_price_known = self._mask_known_series(own_prices.reindex(features.index), prediction_generated_at)
        state_sources = {
            "imbalance_long": market.get("imbalance_prices_long"),
            "imbalance_short": market.get("imbalance_prices_short"),
            "flow_se3_to_fi": market.get("flow_se3_to_fi"),
            "flow_ee_to_fi": market.get("flow_ee_to_fi"),
            "boiler_consumption": market.get("fingrid_electric_boiler"),
            "commercial_flow_fi_ee": market.get("fingrid_commercial_flow_fi_ee"),
        }

        for name, series in state_sources.items():
            if series is None:
                continue
            self._append_state_features(features, name, series, prediction_generated_at)

        for shadow_region in self.region.shadow_regions:
            column = f"shadow_price_{shadow_region.lower()}"
            if column not in market.columns:
                continue
            shadow_price = self._mask_known_series(market[column], prediction_generated_at)
            features[f"{column}_lag_1d"] = self._lag_to_index(shadow_price, features.index, timedelta(days=1))
            shadow_spread = shadow_price - own_price_known
            self._append_state_features(features, f"{column}_spread", shadow_spread, prediction_generated_at)

        return self._to_numeric_frame(features)

    def _append_state_features(
        self,
        features: pd.DataFrame,
        prefix: str,
        series: pd.Series,
        prediction_generated_at: datetime | None,
    ) -> None:
        known_series = self._mask_known_series(series.reindex(features.index), prediction_generated_at)
        lagged = known_series.shift(1)
        mean = lagged.rolling(96, min_periods=4).mean()
        std = lagged.rolling(96, min_periods=4).std()
        last = lagged.ffill()

        if prediction_generated_at is not None:
            cutoff = pd.Timestamp(prediction_generated_at)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            mean = self._carry_forward_after_cutoff(mean, cutoff)
            std = self._carry_forward_after_cutoff(std.fillna(0.0), cutoff)
            last = self._carry_forward_after_cutoff(last, cutoff)

        features[f"{prefix}_recent"] = last
        features[f"{prefix}_recent_mean_24h"] = mean
        features[f"{prefix}_recent_std_24h"] = std

    def _carry_forward_after_cutoff(self, series: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
        result = series.copy()
        history = result.loc[:cutoff].dropna()
        if history.empty:
            return result
        result.loc[result.index > cutoff] = history.iloc[-1]
        return result

    def _mask_known_series(self, series: pd.Series, prediction_generated_at: datetime | None) -> pd.Series:
        result = pd.to_numeric(series, errors="coerce")
        if prediction_generated_at is None:
            return result
        cutoff = pd.Timestamp(prediction_generated_at)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return result.where(result.index <= cutoff)

    def _lag_to_index(self, series: pd.Series, index: pd.DatetimeIndex, delta: timedelta) -> pd.Series:
        lagged = series.reindex(index - delta)
        lagged.index = index
        return lagged

    def _coalesce_columns(
        self,
        frame: pd.DataFrame,
        columns: list[str],
        combine: str = "first",
    ) -> pd.Series:
        existing = [frame[column] for column in columns if column in frame.columns]
        if not existing:
            return pd.Series(index=frame.index, dtype=float)
        if combine == "sum":
            result = existing[0].copy()
            for series in existing[1:]:
                result = result.add(series, fill_value=0.0)
            return result
        result = existing[0].copy()
        for series in existing[1:]:
            result = result.combine_first(series)
        return result

    def _to_numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def _to_numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = self._to_numeric_frame(frame)
        return features.astype(float)

    def _align_prediction_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            return frame
        return frame.reindex(columns=self.feature_columns, fill_value=np.nan)

    def _build_training_weights(self, index: pd.Index, train_end: datetime) -> pd.Series:
        timestamps = pd.DatetimeIndex(index)
        train_end_ts = pd.Timestamp(train_end)
        train_end_ts = train_end_ts.tz_localize("UTC") if train_end_ts.tzinfo is None else train_end_ts.tz_convert("UTC")
        age_days = (train_end_ts - timestamps).total_seconds() / (24 * 60 * 60)
        decay_days = 90.0 if self.region.use_market_features else 180.0
        weights = np.exp(-age_days / decay_days)
        return pd.Series(weights, index=timestamps)

    def _lgb_params(self, objective: str = "regression", alpha: float | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": objective,
            "force_col_wise": True,
            "verbosity": -1,
            "seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "data_random_seed": 42,
        }
        if alpha is not None:
            params["alpha"] = alpha
        return params

    def _apply_post_model_blend(
        self,
        _: pd.DataFrame,
        predictions: pd.DataFrame,
        spike_probability: pd.Series | None = None,
    ) -> pd.DataFrame:
        weight = self.region.yesterday_blend_weight
        if weight <= 0.0 or self.pricestore.data.empty:
            return predictions

        price_data = self.pricestore.data
        if self.pricestore.horizon_cutoff is not None:
            price_data = price_data[:self.pricestore.horizon_cutoff]
        if price_data.empty:
            return predictions

        result = predictions.copy()
        yesterday_baseline = price_data["price"].sort_index().reindex(result.index - timedelta(days=1))
        yesterday_baseline.index = result.index
        mask = yesterday_baseline.notna()
        if mask.any():
            dynamic_weight = pd.Series(weight, index=result.index)
            if spike_probability is not None and not spike_probability.empty:
                dynamic_weight = weight * (1.0 - spike_probability.clip(0.0, 1.0))
            result.loc[mask, "price"] = (
                result.loc[mask, "price"] * (1.0 - dynamic_weight.loc[mask])
                + yesterday_baseline.loc[mask] * dynamic_weight.loc[mask]
            )
        return result

    async def refresh_forecasts(self, start: datetime, end: datetime):
        """
        Will re-fetch everything starting from yesterday during next training.
        """
        await self.weatherstore.refresh_range(start, end)
        await self.entsoestore.refresh_range(start, end)
        await self.marketstore.refresh_range(start, end)

    def cleanup(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.region.retention_days)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)
        self.entsoestore.drop_before(cutoff)
        self.marketstore.drop_before(cutoff)
        self.gasstore.drop_before(cutoff)

    def get_model_artifacts(self) -> dict[str, lgb.Booster]:
        artifacts: dict[str, lgb.Booster] = {}
        if self.predictor is not None:
            artifacts["point"] = self.predictor
        for name, model in self.quantile_models.items():
            artifacts[name] = model
        if self.spike_classifier is not None:
            artifacts["spike_classifier"] = self.spike_classifier
        if self.spike_uplift_model is not None:
            artifacts["spike_uplift"] = self.spike_uplift_model
        return artifacts
