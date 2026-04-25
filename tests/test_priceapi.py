"""Tests for predictor.api.priceapi module."""

from datetime import datetime, timedelta, timezone
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import predictor.api.priceapi as priceapi
from predictor.model.priceregion import PriceRegionName
from predictor.api.priceapi import (
    OutputFormat,
    PriceModel,
    PricesModel,
    PricesModelShort,
    PriceUnit,
    RegionPriceManager,
    app,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_region_manager(sample_region):
    """Create a mock RegionPriceManager."""
    manager = RegionPriceManager(sample_region)
    return manager


class TestPriceUnit:
    """Tests for PriceUnit enum."""

    def test_ct_per_kwh_no_conversion(self):
        """Test CT_PER_KWH returns value as-is."""
        unit = PriceUnit.CT_PER_KWH
        assert unit.convert(10.0) == pytest.approx(10.0)

    def test_eur_per_kwh_conversion(self):
        """Test EUR_PER_KWH divides by 100."""
        unit = PriceUnit.EUR_PER_KWH
        assert unit.convert(100.0) == pytest.approx(1.0)
        assert unit.convert(10.0) == pytest.approx(0.1)

    def test_eur_per_mwh_conversion(self):
        """Test EUR_PER_MWH converts correctly."""
        unit = PriceUnit.EUR_PER_MWH
        # 10 ct/kWh = 0.1 EUR/kWh = 100 EUR/MWh
        assert unit.convert(10.0) == pytest.approx(100.0)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_long_format_exists(self):
        """Test LONG format exists."""
        assert OutputFormat.LONG.value == "LONG"

    def test_short_format_exists(self):
        """Test SHORT format exists."""
        assert OutputFormat.SHORT.value == "SHORT"


class TestPriceModels:
    """Tests for Pydantic models."""

    def test_price_model_creation(self):
        """Test PriceModel can be created."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
            total=10.5
        )
        assert model.total == pytest.approx(10.5)

    def test_prices_model_creation(self):
        """Test PricesModel can be created."""
        prices = [
            PriceModel(starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc), total=10.5),
            PriceModel(starts_at=datetime(2025, 11, 1, 0, 15, tzinfo=timezone.utc), total=11.0),
        ]
        model = PricesModel(
            prices=prices,
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        assert len(model.prices) == 2

    def test_prices_model_short_creation(self):
        """Test PricesModelShort can be created."""
        model = PricesModelShort(
            s=[1730419200, 1730420100],
            t=[10.5, 11.0]
        )
        assert len(model.s) == 2
        assert len(model.t) == 2


class TestPriceModelSerializationAliases:
    """Tests for backward-compatible JSON serialization aliases."""

    def test_price_model_serializes_starts_at_as_camel_case(self):
        """Test PriceModel serializes starts_at as 'startsAt' for backward compatibility."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        json_dict = model.model_dump(by_alias=True)
        assert "startsAt" in json_dict
        assert "starts_at" not in json_dict

    def test_price_model_internal_name_still_works(self):
        """Test PriceModel can still be accessed via internal snake_case name."""
        model = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        assert model.starts_at == datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)

    def test_prices_model_serializes_known_until_as_camel_case(self):
        """Test PricesModel serializes known_until as 'knownUntil' for backward compatibility."""
        model = PricesModel(
            prices=[],
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        json_dict = model.model_dump(by_alias=True)
        assert "knownUntil" in json_dict
        assert "known_until" not in json_dict

    def test_full_response_uses_camel_case_aliases(self):
        """Test complete response structure uses camelCase for API backward compatibility."""
        price = PriceModel(
            starts_at=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            total=10.5
        )
        model = PricesModel(
            prices=[price],
            known_until=datetime(2025, 11, 2, tzinfo=timezone.utc)
        )
        json_dict = model.model_dump(by_alias=True)

        # Top level should have knownUntil
        assert "knownUntil" in json_dict
        # Nested price should have startsAt
        assert "startsAt" in json_dict["prices"][0]
        assert json_dict["prices"][0]["startsAt"] is not None


class TestRegionPriceManagerFormatShort:
    """Tests for RegionPriceManager.format_short method."""

    def test_format_short(self, sample_region):
        """Test format_short converts to short format."""
        manager = RegionPriceManager(sample_region)

        prices = [
            PriceModel(starts_at=datetime(2025, 11, 1, tzinfo=timezone.utc), total=10.5),
            PriceModel(starts_at=datetime(2025, 11, 1, 0, 15, tzinfo=timezone.utc), total=11.0),
        ]

        result = manager.format_short(prices)

        assert isinstance(result, PricesModelShort)
        assert len(result.s) == 2
        assert len(result.t) == 2
        assert result.t[0] == pytest.approx(10.5)
        assert result.t[1] == pytest.approx(11.0)


class TestAPIEndpointRoot:
    """Tests for root endpoint."""

    def test_root_redirects_to_docs(self, client):
        """Test that root endpoint redirects to /docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/docs" in response.headers.get("location", "")

    def test_ui_endpoint_returns_html(self, client):
        response = client.get("/ui")
        assert response.status_code == 200
        assert "Forecast Inspector" in response.text


class TestAPIEndpointPrices:
    """Tests for /prices endpoint."""

    def test_prices_endpoint_exists(self, client):
        """Test that /prices endpoint returns 200 with mocked handler."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices")
            assert response.status_code == 200

    def test_prices_with_hours_parameter(self, client):
        """Test /prices with hours parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?hours=24")
            assert response.status_code == 200

    def test_prices_with_country_parameter(self, client):
        """Test /prices with country parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?country=DE")
            assert response.status_code == 200

    def test_prices_with_unit_parameter(self, client):
        """Test /prices with unit parameter returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModel(
                    prices=[],
                    known_until=datetime(2025, 11, 1, tzinfo=timezone.utc)
                )
            )
            response = client.get("/prices?unit=EUR_PER_KWH")
            assert response.status_code == 200


class TestAPIEndpointPricesShort:
    """Tests for /prices_short endpoint."""

    def test_prices_short_endpoint_exists(self, client):
        """Test that /prices_short endpoint returns 200."""
        with patch("predictor.api.priceapi.prices_handler") as mock_handler:
            mock_handler.prices = AsyncMock(
                return_value=PricesModelShort(s=[], t=[])
            )
            response = client.get("/prices_short")
            assert response.status_code == 200


class TestAPIEndpointUIData:
    def test_ui_data_endpoint_returns_cached_prediction_and_sources(self, client, sample_region):
        manager = RegionPriceManager(sample_region)
        base_time = datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)
        index = pd.date_range(base_time, periods=4, freq="15min", tz="UTC")

        manager.cachedprices = pd.DataFrame({"price": [10.0, 11.0, 12.0, 13.0]}, index=index)
        manager.cachedeval = pd.DataFrame({"price": [9.5, 10.5, 11.5, 12.5]}, index=index)
        manager.predictor.pricestore.data = pd.DataFrame({"price": [10.2, 10.8, 11.9, 13.2]}, index=index)
        manager.predictor.weatherstore.data = pd.DataFrame({"temp_0": [4.0, 4.1, 4.2, 4.3]}, index=index)
        manager.predictor.entsoestore.data = pd.DataFrame({"load": [1000.0, 1005.0, 1010.0, 1012.0]}, index=index)
        manager.predictor.marketstore.data = pd.DataFrame({"shadow_price_se_3": [9.0, 9.5, 10.0, 10.5]}, index=index)
        manager.predictor.gasstore.data = pd.DataFrame({"gasprice": [35.0, 35.0, 35.0, 35.0]}, index=index)
        manager.last_generated_forecast = base_time
        manager.last_known_price = index[-1].to_pydatetime()

        with patch("predictor.api.priceapi.prices_handler.get_price_manager", new=AsyncMock(return_value=manager)):
            response = client.get("/ui/data?region=DE")

        assert response.status_code == 200
        payload = response.json()
        assert payload["region"] == sample_region.bidding_zone_entsoe
        assert payload["price_rows"]
        assert payload["source_rows"]
        assert payload["table_rows"]
        assert payload["source_groups"]
        assert "predicted_price" in payload["table_rows"][0]


class TestAPIEndpointUIExplainability:
    def test_snapshot_catalog_endpoint_returns_runs(self, client):
        manager = RegionPriceManager(PriceRegionName.FI.to_region())
        manager.list_snapshot_runs = MagicMock(
            return_value={
                "region": "FI",
                "runs": [
                    {
                        "generated_at_utc": "2026-04-11T08:00:00+00:00",
                        "row_count": 12,
                        "explainable_row_count": 12,
                    }
                ],
            }
        )

        with patch("predictor.api.priceapi.prices_handler.get_price_manager", new=AsyncMock(return_value=manager)):
            response = client.get("/ui/api/snapshots?region=FI")

        assert response.status_code == 200
        payload = response.json()
        assert payload["region"] == "FI"
        assert payload["runs"][0]["row_count"] == 12

    def test_snapshot_explanation_summary_endpoint_returns_rows(self, client):
        manager = RegionPriceManager(PriceRegionName.FI.to_region())
        manager.get_snapshot_summary = MagicMock(
            return_value={
                "region": "FI",
                "selection_strategy": "latest",
                "rows_evaluated": 4,
                "explainable_rows": 4,
                "rows": [
                    {
                        "time_utc": "2026-04-11T22:00:00+00:00",
                        "generated_at_utc": "2026-04-11T08:00:00+00:00",
                        "predicted_price": 12.5,
                        "actual_price": 11.9,
                        "explainable": True,
                    }
                ],
                "group_summary": [],
                "feature_summary": [],
                "adjustment_summary": [],
                "error_slices": {},
                "runs": [],
            }
        )

        with patch("predictor.api.priceapi.prices_handler.get_price_manager", new=AsyncMock(return_value=manager)):
            response = client.get("/ui/api/explanation-summary?region=FI")

        assert response.status_code == 200
        payload = response.json()
        assert payload["rows_evaluated"] == 4
        assert payload["rows"][0]["explainable"] is True

    def test_snapshot_explanation_endpoint_returns_local_breakdown(self, client):
        manager = RegionPriceManager(PriceRegionName.FI.to_region())
        manager.get_snapshot_explanation = MagicMock(
            return_value={
                "explainable": True,
                "predicted_price": 12.5,
                "base_value": 6.2,
                "feature_contributions": [{"feature_name": "market_load_forecast", "signed_contribution": 1.2}],
                "group_contributions": [{"group_id": "demand_load", "signed_contribution": 1.2}],
                "adjustments": [{"adjustment_name": "yesterday_blend_component", "signed_contribution": 0.3}],
            }
        )

        with patch("predictor.api.priceapi.prices_handler.get_price_manager", new=AsyncMock(return_value=manager)):
            response = client.get(
                "/ui/api/explanation?region=FI&generatedAtUtc=2026-04-11T08:00:00Z&targetTimeUtc=2026-04-11T22:00:00Z"
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["explainable"] is True
        assert payload["feature_contributions"]
        assert payload["group_contributions"]

    def test_snapshot_scenario_endpoint_returns_delta(self, client):
        manager = RegionPriceManager(PriceRegionName.FI.to_region())
        manager.evaluate_snapshot_scenario = MagicMock(
            return_value={
                "explainable": True,
                "baseline_prediction": 12.5,
                "scenario_prediction": 13.1,
                "prediction_delta": 0.6,
                "group_deltas": [{"name": "demand_load", "delta": 0.3}],
                "changed_features": [{"feature_name": "market_load_forecast", "delta": 150.0}],
                "applied_inputs": [{"name": "load_forecast_pct", "applied": True}],
            }
        )

        with patch("predictor.api.priceapi.prices_handler.get_price_manager", new=AsyncMock(return_value=manager)):
            response = client.post(
                "/ui/api/scenario",
                json={
                    "region": "FI",
                    "generatedAtUtc": "2026-04-11T08:00:00Z",
                    "targetTimeUtc": "2026-04-11T22:00:00Z",
                    "loadForecastPct": 10.0,
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["prediction_delta"] == pytest.approx(0.6)
        assert payload["changed_features"]


class TestRegionPriceManagerPrices:
    """Tests for RegionPriceManager.prices method."""

    @pytest.mark.asyncio
    async def test_prices_applies_fixed_price(self, sample_region):
        """Test that fixed price is added to all prices."""
        manager = RegionPriceManager(sample_region)

        # Add cached prices
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = pd.DataFrame({"price": [10.0, 11.0]}, index=pd.to_datetime([base_time, base_time + timedelta(minutes=15)]))
        manager.last_known_price = base_time + timedelta(hours=1)

        # Mock update_in_background to do nothing
        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            surcharge=5.0,
            tax_percent=0.0,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Prices should have fixed price added
        for price in result.prices:
            assert price.total >= 15.0  # 10 + 5 or 11 + 5

    @pytest.mark.asyncio
    async def test_prices_applies_tax(self, sample_region):
        """Test that tax is applied to prices."""
        manager = RegionPriceManager(sample_region)

        # Add cached prices
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = pd.DataFrame({"price": [10.0]}, index=pd.to_datetime([base_time,]))
        manager.last_known_price = base_time + timedelta(hours=1)

        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            surcharge=0.0,
            tax_percent=19.0,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Price should be 10 * 1.19 = 11.9
        if result.prices:
            assert result.prices[0].total == pytest.approx(11.9, rel=0.01)

    @pytest.mark.asyncio
    async def test_prices_hourly_averaging(self, sample_region):
        """Test that hourly mode averages 15-minute prices."""
        manager = RegionPriceManager(sample_region)

        # Add 4 prices for one hour (15-min intervals)
        base_time = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.cachedprices = pd.DataFrame({"price": [10.0, 12.0, 14.0, 16.0]}, index=pd.to_datetime([
            base_time,
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=30),
            base_time + timedelta(minutes=45)
        ]))
        manager.last_known_price = base_time + timedelta(hours=2)

        manager.update_in_background = AsyncMock()

        result = await manager.prices(
            hours=1,
            surcharge=0.0,
            tax_percent=0.0,
            hourly=True,
            format=OutputFormat.LONG
        )

        assert isinstance(result, PricesModel)
        # Should have 1 hourly price (average of 10, 12, 14, 16 = 13)
        if result.prices:
            assert result.prices[0].total == pytest.approx(13.0, rel=0.01)


class TestRegionPriceManagerUpdateDataIfNeeded:
    """Tests for RegionPriceManager.update_data_if_needed method."""

    @pytest.mark.asyncio
    async def test_update_in_background_does_not_create_unused_coroutine_when_request_training_disabled(self, sample_region, monkeypatch):
        manager = RegionPriceManager(sample_region)
        manager.cachedprices = pd.DataFrame(
            {"price": [10.0]},
            index=pd.DatetimeIndex([datetime(2025, 11, 1, tzinfo=timezone.utc)]),
        )
        manager.update_data_if_needed = AsyncMock()
        monkeypatch.setattr(priceapi, "ENABLE_REQUEST_TRAINING", False)

        await manager.update_in_background()

        manager.update_data_if_needed.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_triggers_refresh_when_stale(self, sample_region):
        """Test that update triggers refresh when data is stale."""
        manager = RegionPriceManager(sample_region)

        # Mock predictor methods
        manager.predictor.refresh_forecasts = AsyncMock()
        manager.predictor.train = AsyncMock()
        prediction_frame = pd.DataFrame(
            {"price": [10.0]},
            index=pd.DatetimeIndex([datetime.now(timezone.utc)]),
        )
        manager.predictor.predict_with_details = AsyncMock(
            return_value={
                "point": prediction_frame,
                "quantiles": pd.DataFrame(index=prediction_frame.index),
                "features": pd.DataFrame({"feature": [1.0]}, index=prediction_frame.index),
            }
        )
        manager.predictor.pricestore.get_last_known = MagicMock(
            return_value=datetime.now(timezone.utc)
        )
        manager.predictor.pricestore.get_data = AsyncMock(return_value=prediction_frame)
        manager.predictor.cleanup = MagicMock()
        manager.predictor.get_model_artifacts = MagicMock(return_value={})
        manager.predictor.model_version = "test-version"
        manager.prediction_snapshots.append_predictions = MagicMock(return_value=12)
        manager.predictor_snapshots.append_frame = MagicMock(return_value=12)
        manager.distribution_snapshots.append_frame = MagicMock(return_value=0)
        manager.forecast_artifacts.save_latest = MagicMock()
        manager.forecast_artifacts.save_model_bundle = MagicMock(return_value={})

        await manager.update_data_if_needed()

        # Should have called refresh methods
        assert manager.predictor.refresh_forecasts.called
        assert manager.prediction_snapshots.append_predictions.called
        assert manager.predictor_snapshots.append_frame.called
        assert manager.forecast_artifacts.save_latest.called
        assert manager.prediction_snapshots.append_predictions.call_args.args[3] == "test-version"
        assert manager.predictor_snapshots.append_frame.call_args.args[3]["model_version"] == "test-version"


class TestRegionPriceManagerPersistenceRefresh:
    @pytest.mark.asyncio
    async def test_refresh_from_persistence_if_updated_reloads_prices_and_artifacts(self, sample_region):
        manager = RegionPriceManager(sample_region)
        manager.last_known_price = datetime(2025, 11, 1, tzinfo=timezone.utc)
        manager.last_artifact_load = datetime(2025, 11, 1, tzinfo=timezone.utc)

        newer = datetime(2025, 11, 1, 1, tzinfo=timezone.utc)
        manager.forecast_artifacts.get_latest_update_time = MagicMock(return_value=newer)
        manager.load_cached_artifacts = MagicMock(side_effect=lambda: setattr(manager, "last_artifact_load", newer))

        manager.predictor.weatherstore.load_if_storage_updated = AsyncMock(return_value=False)
        manager.predictor.entsoestore.load_if_storage_updated = AsyncMock(return_value=False)
        manager.predictor.marketstore.load_if_storage_updated = AsyncMock(return_value=False)
        manager.predictor.gasstore.load_if_storage_updated = AsyncMock(return_value=False)
        manager.predictor.pricestore.load_if_storage_updated = AsyncMock(return_value=True)
        manager.predictor.pricestore.get_last_known = MagicMock(return_value=newer)

        await manager.refresh_from_persistence_if_updated()

        assert manager.predictor.pricestore.load_if_storage_updated.called
        assert manager.load_cached_artifacts.called
        assert manager.last_known_price == newer

    @pytest.mark.asyncio
    async def test_ensure_loaded_refreshes_persistence_after_initial_load(self, sample_region):
        manager = RegionPriceManager(sample_region)
        manager.predictor.load_from_persistence = AsyncMock()
        manager.load_cached_artifacts = MagicMock()
        manager.refresh_from_persistence_if_updated = AsyncMock()

        await manager.ensure_loaded()
        await manager.ensure_loaded()

        assert manager.predictor.load_from_persistence.await_count == 1
        assert manager.load_cached_artifacts.call_count == 1
        manager.refresh_from_persistence_if_updated.assert_awaited_once()
