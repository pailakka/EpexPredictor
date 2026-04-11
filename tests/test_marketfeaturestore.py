"""Tests for predictor.model.marketfeaturestore module."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predictor.model.marketfeaturestore import MarketFeatureStore
from predictor.model.priceregion import PriceRegionName


class TestMarketFeatureStore:
    def test_normalize_fingrid_payload(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        payload = [
            {"startTime": "2026-04-10T00:00:00Z", "value": 100.0},
            {"startTime": "2026-04-10T01:00:00Z", "value": 120.0},
        ]

        frame = store._normalize_fingrid_payload(payload, "fingrid_load_forecast")

        assert not frame.empty
        assert frame.loc[pd.Timestamp("2026-04-10T00:15:00Z"), "fingrid_load_forecast"] == 100.0
        assert frame.loc[pd.Timestamp("2026-04-10T01:00:00Z"), "fingrid_load_forecast"] == 120.0

    def test_normalize_fingrid_payload_dataset_specific_value_key(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        payload = {
            "data": [
                {
                    "startTime": "2026-04-10T00:00:00Z",
                    "endTime": "2026-04-10T00:15:00Z",
                    "Electricity consumption forecast - updated every 15 minutes": 9968.6,
                },
                {
                    "startTime": "2026-04-10T00:15:00Z",
                    "endTime": "2026-04-10T00:30:00Z",
                    "Electricity consumption forecast - updated every 15 minutes": 9981.4,
                },
            ]
        }

        frame = store._normalize_fingrid_payload(payload, "fingrid_load_forecast")

        assert not frame.empty
        assert frame.loc[pd.Timestamp("2026-04-10T00:00:00Z"), "fingrid_load_forecast"] == 9968.6
        assert frame.loc[pd.Timestamp("2026-04-10T00:15:00Z"), "fingrid_load_forecast"] == 9981.4

    def test_normalize_entsoe_series_name(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        index = pd.date_range("2026-04-10T00:00:00Z", periods=2, freq="15min", tz="UTC")
        series = pd.Series([1.0, 2.0], index=index, name="Forecasted Load")

        frame = store._normalize_entsoe_frame(series, "entsoe_load_forecast")

        assert list(frame.columns) == ["entsoe_load_forecast_forecasted_load"]

    @pytest.mark.asyncio
    async def test_fetch_fingrid_range_skips_far_historical_requests(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        store.fingrid_api_key = "configured"

        far_start = datetime.now(timezone.utc) - timedelta(days=30)
        far_end = far_start + timedelta(days=1)

        frame = await store._fetch_fingrid_range(far_start, far_end)

        assert frame.empty

    @pytest.mark.asyncio
    async def test_fetch_fingrid_range_uses_cooldown(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        store.fingrid_api_key = "configured"
        store.last_fingrid_refresh = datetime.now(timezone.utc)

        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=1)

        frame = await store._fetch_fingrid_range(start, end)

        assert frame.empty
