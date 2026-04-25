"""Tests for predictor.model.marketfeaturestore module."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predictor.model.marketfeaturestore import MarketFeatureStore
from predictor.model.priceregion import PriceRegionName


class TestMarketFeatureStore:
    def test_fingrid_dataset_ids_include_reference_wind_sources(self):
        assert MarketFeatureStore.FINGRID_DATASETS["fingrid_wind_power_forecast"] == 245
        assert MarketFeatureStore.FINGRID_DATASETS["fingrid_wind_forecast"] == 246
        assert MarketFeatureStore.FINGRID_DATASETS["fingrid_wind_power_realtime"] == 181
        assert MarketFeatureStore.FINGRID_DATASETS["fingrid_wind_capacity"] == 268
        assert MarketFeatureStore.FINGRID_DATASETS["fingrid_nuclear_production"] == 188

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

    def test_normalize_fingrid_wind_power_forecast_payload(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        payload = {
            "data": [
                {
                    "startTime": "2026-04-10T00:00:00Z",
                    "endTime": "2026-04-10T00:15:00Z",
                    "Wind power generation forecast - updated every 15 minutes": 1450.5,
                },
                {
                    "startTime": "2026-04-10T00:15:00Z",
                    "endTime": "2026-04-10T00:30:00Z",
                    "Wind power generation forecast - updated every 15 minutes": 1460.0,
                },
            ]
        }

        frame = store._normalize_fingrid_payload(payload, "fingrid_wind_power_forecast")

        assert not frame.empty
        assert frame.loc[pd.Timestamp("2026-04-10T00:00:00Z"), "fingrid_wind_power_forecast"] == 1450.5
        assert frame.loc[pd.Timestamp("2026-04-10T00:15:00Z"), "fingrid_wind_power_forecast"] == 1460.0

    def test_normalize_jao_payload_calculates_import_total(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        payload = {
            "data": [
                {
                    "dateTimeUtc": "2026-04-10T00:00:00Z",
                    "border_SE1_FI": 100.0,
                    "border_SE3_FI": 200.0,
                    "border_EE_FI": 50.0,
                },
                {
                    "dateTimeUtc": "2026-04-10T01:00:00Z",
                    "border_SE1_FI": 0.0,
                    "border_SE3_FI": 0.0,
                    "border_EE_FI": 0.0,
                },
            ]
        }

        frame = store._normalize_jao_payload(payload)

        assert frame.loc[pd.Timestamp("2026-04-10T00:00:00Z"), "jao_import_capacity_se1_fi"] == 100.0
        assert frame.loc[pd.Timestamp("2026-04-10T00:00:00Z"), "jao_import_capacity_total"] == 350.0
        assert frame.loc[pd.Timestamp("2026-04-10T00:15:00Z"), "jao_import_capacity_total"] == 350.0
        assert frame.loc[pd.Timestamp("2026-04-10T01:00:00Z"), "jao_import_capacity_total"] == 350.0

    def test_compute_hydrology_frame_missing_source_generates_columns(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        index = pd.date_range("2026-04-10T00:00:00Z", periods=2, freq="15min", tz="UTC")

        frame = store._compute_hydrology_frame({}, index)

        expected_columns = {
            "HydroPrecip_5d_median",
            "HydroPrecip_5d_p10",
            "HydroSWE_median",
            "HydroSWE_p10",
        }
        assert expected_columns.issubset(frame.columns)
        assert frame[list(expected_columns)].isna().all().all()

    def test_compute_hydrology_frame_daily_quantiles(self, temp_storage_dir):
        store = MarketFeatureStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        index = pd.date_range("2026-04-10T00:00:00Z", periods=2, freq="15min", tz="UTC")
        rows = pd.DataFrame(
            {
                "Aika": [
                    pd.Timestamp("2026-04-10T00:00:00Z"),
                    pd.Timestamp("2026-04-10T00:00:00Z"),
                    pd.Timestamp("2026-04-10T00:00:00Z"),
                ],
                "Paikka_Id": [848, 852, 810],
                "Arvo": [10.0, 20.0, 30.0],
            }
        )

        frame = store._compute_hydrology_frame({"HydroPrecip_5d": rows}, index)

        assert frame["HydroPrecip_5d_median"].iloc[0] == pytest.approx(20.0)
        assert frame["HydroPrecip_5d_p10"].iloc[0] == pytest.approx(12.0)

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
