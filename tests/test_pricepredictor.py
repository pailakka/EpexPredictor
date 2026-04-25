"""Tests for predictor.model.pricepredictor module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegionName


class TestPricePredictorInit:
    """Tests for PricePredictor initialization."""

    def test_init_creates_stores(self, sample_region):
        """Test that initialization creates all required stores."""
        predictor = PricePredictor(sample_region)
        assert predictor.region == sample_region
        assert predictor.weatherstore is not None
        assert predictor.pricestore is not None
        assert predictor.auxstore is not None
        assert predictor.entsoestore is not None
        assert predictor.marketstore is not None

    def test_init_with_storage_dir(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        predictor = PricePredictor(sample_region, temp_storage_dir)
        assert predictor.weatherstore.storage_dir == temp_storage_dir
        assert predictor.pricestore.storage_dir == temp_storage_dir


class TestPricePredictorGetLastKnownPrice:
    """Tests for get_last_known_price method."""

    def test_get_last_known_price_empty_store(self, sample_region):
        """Test get_last_known_price with empty price store."""
        predictor = PricePredictor(sample_region)
        result = predictor.pricestore.get_last_known()
        assert result is None

    def test_get_last_known_price_with_data(self, sample_region):
        """Test get_last_known_price with data in store."""
        predictor = PricePredictor(sample_region)

        # Add price data
        dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
        df.index.name = "time"
        predictor.pricestore._update_data(df)

        result = predictor.pricestore.get_last_known()
        assert result is not None
        assert isinstance(result, datetime)


class TestPricePredictorToPriceDict:
    """Tests for to_price_dict method."""

    def test_to_price_dict(self, sample_region):
        """Test conversion of DataFrame to price dictionary."""
        predictor = PricePredictor(sample_region)

        dates = pd.date_range(
            start="2025-11-01", end="2025-11-01T01:00", freq="15min", tz="UTC"
        )
        df = pd.DataFrame({"price": [8.0, 9.0, 10.0, 11.0, 12.0]}, index=dates)
        df.index.name = "time"

        result = predictor.to_price_dict(df)

        assert isinstance(result, dict)
        assert len(result) == len(df)
        for dt, price in result.items():
            assert isinstance(dt, datetime)
            assert isinstance(price, float)


class TestPricePredictorPrepareDataframe:
    """Tests for prepare_dataframe method."""

    @pytest.mark.asyncio
    async def test_prepare_dataframe_combines_data(
        self, sample_region, sample_weather_data, sample_price_data, sample_aux_data, sample_entsoe_data, sample_gas_price_data, sample_market_data
    ):
        """Test that prepare_dataframe combines all data sources."""
        predictor = PricePredictor(sample_region)

        # Mock the stores to return our sample data
        predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
        predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
        predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
        predictor.entsoestore.get_data = AsyncMock(return_value=sample_entsoe_data)
        predictor.marketstore.get_data = AsyncMock(return_value=sample_market_data)
        predictor.gasstore.get_data = AsyncMock(return_value=sample_gas_price_data)

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        result = await predictor.prepare_dataframe(start, end)

        assert result is not None
        assert not result.empty
        assert "gasprice" in result.columns


class TestPricePredictorTrain:
    """Tests for train method."""

    @pytest.mark.asyncio
    async def test_train_creates_model(self, mocked_predictor):
        """Test that training creates a model."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await mocked_predictor.train(start, end)

        assert mocked_predictor.predictor is not None


class TestPricePredictorPredict:
    """Tests for predict method."""

    @pytest.mark.asyncio
    async def test_predict_returns_dataframe(self, mocked_predictor):
        """Test that predict returns a DataFrame with predictions."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict
        result = await mocked_predictor.predict(start, end, generated_at=start)

        assert result is not None
        assert not result.empty
        assert "price" in result.columns

    @pytest.mark.asyncio
    async def test_predict_fill_known_true(self, mocked_predictor):
        """Test that predict with fill_known=True uses known prices."""
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        # Train first
        await mocked_predictor.train(start, end)

        # Predict with fill_known=True
        result = await mocked_predictor.predict(start, end, fill_known=True, generated_at=start)

        assert result is not None
        assert not result.empty

    @pytest.mark.asyncio
    async def test_predict_aligns_features_to_training_columns(self):
        """Prediction should reindex features to the training contract when columns differ."""
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.DatetimeIndex([pd.Timestamp("2025-11-01T00:00:00Z")])
        predictor.prepare_dataframe = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "price": [float("nan")],
                    "feature_a": [1.0],
                    "feature_c": [3.0],
                },
                index=index,
            )
        )
        predictor.predictor = MagicMock()
        predictor.predictor.predict.return_value = np.array([42.0])
        predictor.lear_model = MagicMock()
        predictor.lear_model.predict.return_value = np.array([0.0])
        predictor.feature_columns = ["feature_a", "feature_b"]

        details = await predictor.predict_with_details(
            index[0].to_pydatetime(),
            index[0].to_pydatetime(),
            fill_known=False,
            generated_at=index[0].to_pydatetime(),
        )

        assert list(details["features"].columns) == ["feature_a", "feature_b"]
        assert details["features"].loc[index[0], "feature_a"] == pytest.approx(1.0)
        assert pd.isna(details["features"].loc[index[0], "feature_b"])
        predictor.predictor.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_applies_fi_blend_only_to_unknown_rows(self):
        """FI blend should be 95/5 model-to-yesterday and known prices should still overwrite output."""
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-11-01T00:00:00Z"),
                pd.Timestamp("2025-11-01T00:15:00Z"),
                pd.Timestamp("2025-11-01T00:30:00Z"),
            ]
        )
        df = pd.DataFrame(
            {
                "price": [20.0, float("nan"), float("nan")],
                "feature": [1.0, 2.0, 3.0],
            },
            index=index,
        )
        predictor.pricestore.data = pd.DataFrame(
            {"price": [12.0]},
            index=pd.DatetimeIndex([index[1] - timedelta(days=1)]),
        )
        predictor.prepare_dataframe = AsyncMock(return_value=df)
        predictor.predictor = MagicMock()
        predictor.predictor.predict.return_value = np.array([30.0, 40.0, 50.0])
        predictor.lear_model = MagicMock()
        predictor.lear_model.predict.return_value = np.array([0.0, 0.0, 0.0])

        result = await predictor.predict(index[0].to_pydatetime(), index[-1].to_pydatetime(), fill_known=True, generated_at=index[0].to_pydatetime())

        assert result.loc[index[0], "price"] == pytest.approx(20.0)
        assert result.loc[index[1], "price"] == pytest.approx(40.0 * 0.95 + 12.0 * 0.05)
        assert result.loc[index[2], "price"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_predict_blend_respects_price_horizon_cutoff(self):
        """Yesterday baseline must not use prices beyond the current known horizon."""
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.DatetimeIndex([pd.Timestamp("2025-11-02T00:15:00Z")])
        df = pd.DataFrame({"price": [float("nan")], "feature": [1.0]}, index=index)
        predictor.pricestore.data = pd.DataFrame(
            {"price": [12.0]},
            index=pd.DatetimeIndex([index[0] - timedelta(days=1)]),
        )
        predictor.pricestore.horizon_cutoff = index[0] - timedelta(days=1, minutes=15)
        predictor.prepare_dataframe = AsyncMock(return_value=df)
        predictor.predictor = MagicMock()
        predictor.predictor.predict.return_value = np.array([40.0])
        predictor.lear_model = MagicMock()
        predictor.lear_model.predict.return_value = np.array([0.0])

        result = await predictor.predict(index[0].to_pydatetime(), index[0].to_pydatetime(), fill_known=False, generated_at=index[0].to_pydatetime())

        assert result.loc[index[0], "price"] == pytest.approx(40.0)

    def test_build_market_features_masks_future_state(self):
        """Recent state features should carry forward known values instead of leaking future rows."""
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2025-11-01T00:00:00Z", periods=8, freq="15min", tz="UTC")
        marketdata = pd.DataFrame(
            {
                "imbalance_prices_long": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
                "shadow_price_se_3": [10.0, 10.0, 10.0, 10.0, 50.0, 60.0, 70.0, 80.0],
                "entsoe_load_forecast_forecasted_load": [10000.0] * len(index),
                "entsoe_wind_solar_solar": [100.0] * len(index),
                "entsoe_wind_solar_wind_onshore": [1000.0] * len(index),
                "entsoe_generation_forecast_actual_aggregated": [9500.0] * len(index),
            },
            index=index,
        )
        own_prices = pd.Series([20.0, 21.0, 22.0, 23.0, 99.0, 99.0, 99.0, 99.0], index=index)
        generated_at = index[3].to_pydatetime()

        features = predictor._build_market_features(marketdata, own_prices, generated_at)

        assert features.loc[index[4], "imbalance_long_recent"] == pytest.approx(features.loc[index[3], "imbalance_long_recent"])
        assert pd.isna(features.loc[index[4], "shadow_price_se_3_lag_2d"])

    def test_build_market_features_uses_full_target_index_for_price_lags(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        target_index = pd.date_range("2026-04-20T00:00:00Z", periods=8, freq="15min", tz="UTC")
        market_index = target_index[:4]
        marketdata = pd.DataFrame(
            {
                "entsoe_load_forecast_forecasted_load": [10000.0] * len(market_index),
                "entsoe_wind_solar_wind_onshore": [1000.0] * len(market_index),
                "entsoe_wind_solar_solar": [100.0] * len(market_index),
            },
            index=market_index,
        )
        lag_index = target_index - timedelta(days=7)
        own_prices = pd.Series(np.arange(len(target_index), dtype=float) + 20.0, index=lag_index)

        features = predictor._build_market_features(
            marketdata,
            own_prices,
            target_index[0].to_pydatetime(),
            target_index=target_index,
        )

        assert features.index.equals(target_index)
        assert features.loc[target_index[-1], "own_price_lag_7d"] == pytest.approx(27.0)
        assert pd.isna(features.loc[target_index[-1], "market_load_forecast"])

    def test_select_model_excluded_features_drops_stale_weekly_lag_from_recent_data(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=120, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "own_price_lag_2d": np.linspace(1.0, 3.0, len(index)),
                "own_price_lag_7d": np.linspace(11.0, 13.0, len(index)),
                "market_wind_forecast": [3000.0] * len(index),
            },
            index=index,
        )
        output = pd.Series(np.linspace(1.1, 3.1, len(index)), index=index)

        predictor.model_excluded_features = predictor._select_model_excluded_features(
            frame,
            output,
            index[-1].to_pydatetime(),
        )
        features = predictor._to_numeric_features(frame)

        assert predictor.model_excluded_features == {"own_price_lag_7d"}
        assert "own_price_lag_2d" in features.columns
        assert "market_wind_forecast" in features.columns
        assert "own_price_lag_7d" not in features.columns

    def test_select_model_excluded_features_keeps_supported_weekly_lag(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=120, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "own_price_lag_2d": np.linspace(8.0, 10.0, len(index)),
                "own_price_lag_7d": np.linspace(1.0, 3.0, len(index)),
            },
            index=index,
        )
        output = pd.Series(np.linspace(1.1, 3.1, len(index)), index=index)

        excluded = predictor._select_model_excluded_features(frame, output, index[-1].to_pydatetime())

        assert excluded == set()

    def test_build_market_features_falls_back_from_fingrid_to_entsoe_wind_rowwise(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=3, freq="15min", tz="UTC")
        marketdata = pd.DataFrame(
            {
                "fingrid_wind_power_forecast": [1200.0, np.nan, 1400.0],
                "fingrid_wind_forecast": [1100.0, np.nan, 1300.0],
                "entsoe_wind_solar_wind_onshore": [900.0, 950.0, 1000.0],
                "entsoe_wind_solar_solar": [100.0, 100.0, 100.0],
                "entsoe_load_forecast_forecasted_load": [10000.0, 10000.0, 10000.0],
            },
            index=index,
        )
        own_prices = pd.Series([10.0] * len(index), index=index)

        features = predictor._build_market_features(marketdata, own_prices, index[0].to_pydatetime())

        assert features.loc[index[0], "market_wind_forecast"] == pytest.approx(1200.0)
        assert features.loc[index[1], "market_wind_forecast"] == pytest.approx(950.0)
        assert features.loc[index[2], "market_wind_forecast"] == pytest.approx(1400.0)

    def test_build_market_features_prefers_jao_import_capacity_total(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=2, freq="15min", tz="UTC")
        marketdata = pd.DataFrame(
            {
                "jao_import_capacity_total": [2500.0, np.nan],
                "capacity_se_1_to_fi": [100.0, 100.0],
                "capacity_se_3_to_fi": [200.0, 200.0],
                "capacity_ee_to_fi": [300.0, 300.0],
                "entsoe_load_forecast_forecasted_load": [10000.0, 10000.0],
                "entsoe_wind_solar_wind_onshore": [1000.0, 1000.0],
                "entsoe_wind_solar_solar": [100.0, 100.0],
            },
            index=index,
        )
        own_prices = pd.Series([10.0] * len(index), index=index)

        features = predictor._build_market_features(marketdata, own_prices, index[0].to_pydatetime())

        assert features.loc[index[0], "available_import_headroom"] == pytest.approx(2500.0)
        assert features.loc[index[1], "available_import_headroom"] == pytest.approx(600.0)

    def test_build_market_features_keeps_residual_load_when_solar_forecast_missing(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=1, freq="15min", tz="UTC")
        marketdata = pd.DataFrame(
            {
                "fingrid_load_forecast": [10000.0],
                "fingrid_wind_power_forecast": [1200.0],
                "fingrid_solar_forecast": [np.nan],
                "entsoe_generation_forecast_actual_aggregated": [9500.0],
                "jao_import_capacity_total": [0.0],
            },
            index=index,
        )
        own_prices = pd.Series([10.0], index=index)

        features = predictor._build_market_features(marketdata, own_prices, index[0].to_pydatetime())

        assert pd.isna(features.loc[index[0], "market_solar_forecast"])
        assert features.loc[index[0], "market_residual_load"] == pytest.approx(8800.0)
        assert features.loc[index[0], "market_dispatchable_gap"] == pytest.approx(8300.0)
        assert features.loc[index[0], "market_renewable_penetration"] == pytest.approx(0.12)
        assert features.loc[index[0], "market_thermal_burden"] == pytest.approx(8800.0)

    def test_fi_blend_treats_missing_core_market_forecasts_as_weak_inputs(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        generated_at = pd.Timestamp("2026-04-20T00:00:00Z")
        index = pd.DatetimeIndex(
            [
                generated_at + pd.Timedelta(hours=12),
                generated_at + pd.Timedelta(hours=30),
                generated_at + pd.Timedelta(hours=54),
            ]
        )
        dynamic_weight = pd.Series(0.05, index=index)
        feature_frame = pd.DataFrame(
            {
                "available_import_headroom": [3500.0, 3500.0, 3500.0],
                "fi_nuclear_available_mw": [4300.0, 4300.0, 4300.0],
                "market_load_forecast": [np.nan, np.nan, np.nan],
                "market_wind_forecast": [np.nan, np.nan, np.nan],
                "market_residual_load": [np.nan, np.nan, np.nan],
            },
            index=index,
        )

        adjusted = predictor._adjust_blend_weight_for_feature_availability(
            dynamic_weight,
            feature_frame,
            generated_at.to_pydatetime(),
        )

        assert adjusted.loc[index[0]] == pytest.approx(0.05)
        assert adjusted.loc[index[1]] == pytest.approx(0.30)
        assert adjusted.loc[index[2]] == pytest.approx(0.60)

    def test_fi_blend_keeps_weight_when_core_market_forecasts_are_available(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        generated_at = pd.Timestamp("2026-04-20T00:00:00Z")
        index = pd.DatetimeIndex([generated_at + pd.Timedelta(hours=54)])
        dynamic_weight = pd.Series(0.05, index=index)
        feature_frame = pd.DataFrame(
            {
                "market_load_forecast": [10000.0],
                "market_wind_forecast": [2000.0],
                "market_residual_load": [7800.0],
                "available_import_headroom": [3500.0],
            },
            index=index,
        )

        adjusted = predictor._adjust_blend_weight_for_feature_availability(
            dynamic_weight,
            feature_frame,
            generated_at.to_pydatetime(),
        )

        assert adjusted.loc[index[0]] == pytest.approx(0.05)

    def test_low_wind_scaler_gates_and_caps(self):
        predictor = PricePredictor(PriceRegionName.DE.to_region())
        index = pd.date_range("2026-04-20T03:00:00Z", periods=96, freq="15min", tz="UTC")
        features = pd.DataFrame({"market_wind_forecast": [1000.0] * len(index)}, index=index)
        features.loc[index[20], "market_wind_forecast"] = 100.0
        predictions = pd.DataFrame({"price": np.linspace(10.0, 100.0, len(index))}, index=index)
        predictions.loc[index[20], "price"] = 200.0
        predictor.traindata = pd.DataFrame(
            {"market_wind_forecast": np.linspace(100.0, 2000.0, 96)},
            index=index - timedelta(days=1),
        )

        scaled, multiplier = predictor._apply_low_wind_scaler(
            features,
            predictions,
            pd.Timestamp("2026-04-20T00:00:00Z").to_pydatetime(),
        )

        assert float(multiplier.max()) <= 1.30
        assert multiplier.loc[index[20]] > 1.0
        assert scaled.loc[index[20], "price"] == pytest.approx(predictions.loc[index[20], "price"] * multiplier.loc[index[20]])
        assert multiplier.loc[index[0]] == pytest.approx(1.0)

    def test_low_price_scaler_gates_future_surplus_regime_from_available_signals(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        generated_at = pd.Timestamp("2026-04-20T00:00:00Z")
        history_index = pd.date_range(generated_at - pd.Timedelta(days=1), periods=100, freq="15min", tz="UTC")
        predictor.traindata = pd.DataFrame(
            {
                "price": np.linspace(1.0, 10.0, len(history_index)),
                "market_thermal_burden": np.linspace(0.0, 10000.0, len(history_index)),
                "market_renewable_penetration": np.linspace(0.0, 1.0, len(history_index)),
                "available_import_headroom": np.linspace(1000.0, 5000.0, len(history_index)),
                "own_price_rolling_mean_24h": np.linspace(0.0, 10.0, len(history_index)),
                "wind_mean": np.linspace(0.0, 40.0, len(history_index)),
            },
            index=history_index,
        )
        index = pd.DatetimeIndex(
            [
                generated_at - pd.Timedelta(minutes=15),
                generated_at + pd.Timedelta(minutes=15),
                generated_at + pd.Timedelta(minutes=30),
                generated_at + pd.Timedelta(minutes=45),
            ]
        )
        features = pd.DataFrame(
            {
                "market_thermal_burden": [100.0, 100.0, 100.0, 100.0],
                "market_renewable_penetration": [0.9, 0.9, 0.9, 0.9],
                "available_import_headroom": [4500.0, 4500.0, 4500.0, 4500.0],
                "own_price_rolling_mean_24h": [1.0, 1.0, 1.0, 1.0],
                "wind_mean": [35.0, 35.0, 35.0, 35.0],
            },
            index=index,
        )
        predictions = pd.DataFrame({"price": [5.0, 5.0, 8.0, 5.0]}, index=index)
        low_wind_multiplier = pd.Series([1.0, 1.0, 1.0, 1.2], index=index)

        scaled, multiplier = predictor._apply_low_price_regime_scaler(
            features,
            predictions,
            generated_at.to_pydatetime(),
            low_wind_multiplier,
        )

        assert multiplier.loc[index[0]] == pytest.approx(1.0)
        assert 0.0 < multiplier.loc[index[1]] < 1.0
        assert multiplier.loc[index[2]] == pytest.approx(1.0)
        assert multiplier.loc[index[3]] == pytest.approx(1.0)
        assert scaled.loc[index[1], "price"] == pytest.approx(5.0 * multiplier.loc[index[1]])

    def test_weather_aggregates_added_for_current_and_legacy_columns(self):
        predictor = PricePredictor(PriceRegionName.FI.to_region())
        index = pd.date_range("2026-04-20T00:00:00Z", periods=1, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "temp_0": [0.0],
                "temperature_2m_1": [2.0],
                "wind_0": [4.0],
                "wind_speed_80m_1": [8.0],
            },
            index=index,
        )

        predictor._append_weather_aggregate_features(frame)

        assert frame.loc[index[0], "temp_mean"] == pytest.approx(1.0)
        assert frame.loc[index[0], "temp_variance"] == pytest.approx(2.0)
        assert frame.loc[index[0], "wind_mean"] == pytest.approx(6.0)
        assert frame.loc[index[0], "wind_variance"] == pytest.approx(8.0)


class TestPricePredictorCleanup:
    """Tests for cleanup method."""

    def test_cleanup_removes_old_data(self, sample_region):
        """Test that cleanup removes data older than 1 year."""
        predictor = PricePredictor(sample_region)

        # Add old data (2 years ago)
        old_dates = pd.date_range(
            start="2023-01-01", end="2023-01-02", freq="15min", tz="UTC"
        )
        old_df = pd.DataFrame({"price": [8.0] * len(old_dates)}, index=old_dates)
        old_df.index.name = "time"
        predictor.pricestore._update_data(old_df)

        # Add recent data
        recent_dates = pd.date_range(
            start="2025-11-01", end="2025-11-02", freq="15min", tz="UTC"
        )
        recent_df = pd.DataFrame({"price": [9.0] * len(recent_dates)}, index=recent_dates)
        recent_df.index.name = "time"
        predictor.pricestore._update_data(recent_df)

        # Cleanup
        predictor.cleanup()

        # Old data should be removed
        assert predictor.pricestore.data.index.min() > pd.Timestamp("2024-01-01", tz="UTC")


class TestPricePredictorRefreshMethods:
    """Tests for refresh_prices and refresh_forecasts methods."""


    @pytest.mark.asyncio
    async def test_refresh_forecasts(self, sample_region):
        """Test refresh_forecasts method."""
        predictor = PricePredictor(sample_region)

        # Mock the weather store's refresh_range method (not fetch_missing_data)
        predictor.weatherstore.refresh_range = AsyncMock()
        predictor.entsoestore.refresh_range = AsyncMock()
        predictor.marketstore.refresh_range = AsyncMock()

        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 8, tzinfo=timezone.utc)

        await predictor.refresh_forecasts(start, end)

        # Should have called refresh_range
        assert predictor.weatherstore.refresh_range.called
        assert predictor.entsoestore.refresh_range.called
        assert predictor.marketstore.refresh_range.called
