#!/usr/bin/python3

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, cast

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet

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
    lear_model: ElasticNet | None = None
    calibration_residuals: pd.Series | None = None
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

        self.lear_model = None
        self.calibration_residuals = None
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
        return self.predictor is not None and self.lear_model is not None

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

        # Use raw prices to preserve spike magnitude (no variance-stabilizing transform)
        output_transformed = output

        weights = self._build_training_weights(params.index, end)
        params = params.sort_index()
        output_transformed = output_transformed.sort_index()
        weights = weights.sort_index()

        # Phase 1: LEAR structural baseline (lags + calendar + price-regime anchor)
        # Including rolling_mean gives LEAR a price-level anchor so it doesn't need
        # to reconstruct the current regime purely from 2d-ago prices.
        structural_cols = [c for c in [
            "own_price_lag_2d",
            "own_price_lag_7d",
            "own_price_rolling_mean_24h",
            "weekday",
            "hour_of_day",
        ] if c in params.columns]
        structural_params = params[structural_cols].fillna(0)
        self.lear_model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
        self.lear_model.fit(structural_params, output_transformed, sample_weight=weights)
        lear_predictions = pd.Series(self.lear_model.predict(structural_params), index=params.index)

        cat_features = [c for c in ["weekday", "month"] if c in params.columns]

        # Phase 2: LightGBM learns the nonlinear residual
        residual_target = output_transformed - lear_predictions

        # Temporal split: last 10% as held-out validation
        split_idx = int(len(params) * 0.9)
        if split_idx == 0 or split_idx == len(params):
            train_set = lgb.Dataset(params, label=residual_target, weight=weights, categorical_feature=cat_features)
            valid_sets = [train_set]
            train_x, val_x = params, params
            train_y, val_y = residual_target, residual_target
            train_w = weights
        else:
            train_x, val_x = params.iloc[:split_idx], params.iloc[split_idx:]
            train_y, val_y = residual_target.iloc[:split_idx], residual_target.iloc[split_idx:]
            train_w, val_w = weights.iloc[:split_idx], weights.iloc[split_idx:]
            train_set = lgb.Dataset(train_x, label=train_y, weight=train_w, categorical_feature=cat_features)
            val_set = lgb.Dataset(val_x, label=val_y, weight=val_w, reference=train_set)
            valid_sets = [train_set, val_set]


        best_params = self._lgb_params()
        if split_idx > 0:
            try:
                best_params = await asyncio.to_thread(
                    self._optimize_hyperparameters, train_x, train_y, train_w, cat_features
                )
            except Exception as e:
                log.warning("%s: Optuna optimization failed, using defaults: %s", self.region.bidding_zone_entsoe, e)

        self.predictor = await asyncio.to_thread(
            lgb.train,
            params=best_params,
            train_set=train_set,
            num_boost_round=1500,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # Phase 3: Conformal calibration from held-out validation errors
        if split_idx > 0 and split_idx < len(params):
            val_structural = val_x[structural_cols].fillna(0)
            val_base = self.lear_model.predict(val_structural)
            val_res = self.predictor.predict(val_x)
            val_preds = val_base + val_res
            self.calibration_residuals = pd.Series(np.abs(output_transformed.iloc[split_idx:] - val_preds))
        else:
            self.calibration_residuals = pd.Series([1.0])

        # Keep legacy spike_classifier/quantile_models stubs so artifact store stays happy
        self.quantile_models = {}
        self.spike_classifier = None
        self.spike_uplift_model = None

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
        assert self.is_trained() and self.predictor is not None and self.lear_model is not None

        generation_time = generated_at or start
        df = await self.prepare_dataframe(start, end, prediction_generated_at=generation_time)
        assert df is not None

        prices_known = df["price"].copy()
        params = self._to_numeric_features(df.drop(columns=["price"]))
        params = self._align_prediction_features(params)

        # LEAR structural baseline
        structural_cols = [c for c in [
            "own_price_lag_2d",
            "own_price_lag_7d",
            "own_price_rolling_mean_24h",
            "weekday",
            "hour_of_day",
        ] if c in params.columns]
        structural_params = params[structural_cols].fillna(0)
        lear_preds = self.lear_model.predict(structural_params)

        # LightGBM residual correction
        res_preds = self.predictor.predict(params)
        point_transformed = lear_preds + res_preds

        point_forecast = pd.DataFrame(index=params.index)
        point_forecast["price"] = point_transformed

        spike_probability = pd.Series(0.0, index=params.index)

        # Conformal prediction intervals
        quantiles = pd.DataFrame(index=params.index)
        if self.calibration_residuals is not None and not self.calibration_residuals.empty:
            q90_err = float(np.quantile(self.calibration_residuals, 0.90))
            quantiles["q10"] = point_transformed - q90_err
            quantiles["q90"] = point_transformed + q90_err
            quantiles["q50"] = point_forecast["price"]
        else:
            quantiles["q50"] = point_forecast["price"]
            quantiles["q10"] = point_forecast["price"] * 0.9
            quantiles["q90"] = point_forecast["price"] * 1.1

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

        # Cold-morning demand-spike interaction: colder temp × closer to morning peak.
        # This is the primary driver of Finnish pre-dawn price spikes (06-09 EET) in
        # spring/autumn — electric heating surges when temperatures drop unexpectedly.
        if "temp_0" in df.columns and "morningpeak" in df.columns:
            cold_factor = np.clip(15.0 - df["temp_0"], 0.0, 30.0)
            morning_proximity = np.exp(-np.abs(df["morningpeak"]) / (2 * 3600))
            df["cold_morning_peak"] = cold_factor * morning_proximity

        # Evening cold interaction (19:00 EET demand peak)
        if "temp_0" in df.columns and "eveningpeak" in df.columns:
            cold_factor = np.clip(15.0 - df["temp_0"], 0.0, 30.0)
            evening_proximity = np.exp(-np.abs(df["eveningpeak"]) / (2 * 3600))
            df["cold_evening_peak"] = cold_factor * evening_proximity

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

        # Autoregressive price lags — use 2d lag as the always-known safe lag.
        # lag_1d for D+1 slots maps to D+0 22:00–23:45 UTC, still future when model
        # runs at ~17:00 UTC → NaN → fillna(0) → LEAR baseline collapses to ~zero.
        # lag_2d (same hour D-1) is fully known at any realistic forecast time.
        features["own_price_lag_2d"] = self._lag_to_index(own_price_known, features.index, timedelta(days=2))
        features["own_price_lag_7d"] = self._lag_to_index(own_price_known, features.index, timedelta(days=7))

        # Rolling statistics on the 2d lag
        lag_2d = features["own_price_lag_2d"]
        features["own_price_rolling_mean_24h"] = lag_2d.rolling(96, min_periods=24).mean()
        features["own_price_rolling_max_24h"] = lag_2d.rolling(96, min_periods=24).max()
        features["own_price_rolling_min_24h"] = lag_2d.rolling(96, min_periods=24).min()
        # 3-day rolling mean as a slower price-regime indicator
        features["own_price_rolling_mean_72h"] = lag_2d.rolling(96 * 3, min_periods=48).mean()

        # Normalised residual load: deviation from 7-day rolling baseline normalised by std.
        # Raw load in MW has a spurious negative coefficient (high load ≈ winter ≈ lower wind),
        # but load deviation from normal (cold snap, workday peak) directly pressures price.
        load_baseline = load.rolling(96 * 7, min_periods=96).mean()
        load_std = load.rolling(96 * 7, min_periods=96).std().replace(0, np.nan)
        features["load_deviation_norm"] = (load - load_baseline) / load_std

        # Wind ramp: how much the forecasted wind changed vs. 24h ago (sudden drought → price spike).
        wind_24h_ago = self._lag_to_index(wind, features.index, timedelta(hours=24))
        features["wind_ramp_24h"] = wind - wind_24h_ago

        # Load ramp
        load_24h_ago = self._lag_to_index(load, features.index, timedelta(hours=24))
        features["load_ramp_24h"] = load - load_24h_ago

        features["market_residual_load_ramp_3h"] = features["market_residual_load"].diff(periods=12)

        # Renewable penetration & thermal burden
        features["market_renewable_penetration"] = (wind + solar) / load.replace(0, np.nan)
        features["market_thermal_burden"] = load - wind - solar - capacity_import

        # Explicit spatial cross-border features
        for col in market.columns:
            if col.startswith("capacity_"):
                features[col] = market[col]

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
            # Use lag_2d (not lag_1d) — same masking reason as own_price_lag.
            # lag_1d for D+1 slots maps back to D+0 which isn't published yet.
            features[f"{column}_lag_2d"] = self._lag_to_index(shadow_price, features.index, timedelta(days=2))
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
        # 180-day half-life: focuses the model on the recent price regime (spring 2026)
        # while still retaining a full year of seasonal signal via the long history window.
        # Winter 2025 data with different price levels gets downweighted ~7x vs. last month.
        decay_days = 180.0
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
            "learning_rate": 0.05,
        }
        if alpha is not None:
            params["alpha"] = alpha
        return params

    def _optimize_hyperparameters(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        train_w: pd.Series,
        cat_features: list[str],
    ) -> dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            param = {
                "objective": "regression",
                "force_col_wise": True,
                "verbosity": -1,
                "seed": 42,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }
            split_idx = int(len(train_x) * 0.8)
            cv_tx, cv_vx = train_x.iloc[:split_idx], train_x.iloc[split_idx:]
            cv_ty, cv_vy = train_y.iloc[:split_idx], train_y.iloc[split_idx:]
            cv_tw = train_w.iloc[:split_idx]
            cv_vw = train_w.iloc[split_idx:]
            cv_train = lgb.Dataset(cv_tx, label=cv_ty, weight=cv_tw, categorical_feature=cat_features)
            cv_val = lgb.Dataset(cv_vx, label=cv_vy, weight=cv_vw, reference=cv_train)
            gbm = lgb.train(
                param, cv_train,
                num_boost_round=800,
                valid_sets=[cv_val],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            preds = gbm.predict(cv_vx)
            return float(np.mean(np.abs(preds - cv_vy)))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=8, timeout=120)
        best = study.best_params
        best.update({"objective": "regression", "force_col_wise": True, "verbosity": -1, "seed": 42})
        return best

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

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        if self.predictor is not None:
            artifacts["point"] = self.predictor
        if self.lear_model is not None:
            artifacts["lear_baseline"] = self.lear_model
        return artifacts
