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
        assert pd.isna(features.loc[index[4], "shadow_price_se_3_lag_1d"])


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
