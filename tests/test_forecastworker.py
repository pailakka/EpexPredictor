"""Tests for predictor.forecastworker module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from predictor.forecastworker import refresh_regions
from predictor.model.priceregion import PriceRegionName


class TestForecastWorker:
    @pytest.mark.asyncio
    async def test_refresh_regions_forces_manager_update(self):
        manager = MagicMock()
        manager.ensure_loaded = AsyncMock()
        manager.update_data_if_needed = AsyncMock()

        with patch("predictor.forecastworker.RegionPriceManager", return_value=manager):
            await refresh_regions([PriceRegionName.FI], force=True)

        assert manager.ensure_loaded.called
        assert manager.update_data_if_needed.await_args.kwargs["force"] is True
