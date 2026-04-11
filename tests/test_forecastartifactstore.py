"""Tests for predictor.model.forecastartifactstore module."""

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from predictor.model.forecastartifactstore import ForecastArtifactStore
from predictor.model.priceregion import PriceRegionName


class TestForecastArtifactStore:
    def test_save_and_load_latest_frames(self, temp_storage_dir):
        region = PriceRegionName.FI.to_region()
        store = ForecastArtifactStore(region, temp_storage_dir)
        index = pd.date_range("2026-04-10T00:00:00Z", periods=2, freq="15min", tz="UTC")
        point = pd.DataFrame({"price": [10.0, 11.0]}, index=index)
        evaluation = pd.DataFrame({"price": [9.5, 10.5]}, index=index)
        quantiles = pd.DataFrame({"q10": [8.0, 9.0], "q50": [10.0, 11.0], "q90": [12.0, 13.0]}, index=index)

        store.save_latest(
            point,
            evaluation,
            quantiles,
            {
                "generated_at_utc": datetime(2026, 4, 10, 1, 0, tzinfo=timezone.utc).isoformat(),
                "known_until_utc": datetime(2026, 4, 10, 0, 45, tzinfo=timezone.utc).isoformat(),
            },
        )

        loaded_point, loaded_eval, loaded_quantiles, metadata = store.load_latest()

        assert loaded_point.equals(point)
        assert loaded_eval.equals(evaluation)
        assert loaded_quantiles.equals(quantiles)
        assert metadata["generated_at_utc"] == "2026-04-10T01:00:00+00:00"

    def test_get_model_files_and_prune_unreferenced_versions(self, temp_storage_dir):
        region = PriceRegionName.FI.to_region()
        store = ForecastArtifactStore(region, temp_storage_dir)

        old_dir = Path(temp_storage_dir) / "models" / region.bidding_zone_entsoe / "20240101T120000Z"
        recent_dir = Path(temp_storage_dir) / "models" / region.bidding_zone_entsoe / "20260401T120000Z"
        old_dir.mkdir(parents=True)
        recent_dir.mkdir(parents=True)
        (old_dir / "point.txt").write_text("old", encoding="utf-8")
        (recent_dir / "point.txt").write_text("recent", encoding="utf-8")

        model_files = store.get_model_files("20260401T120000Z")
        removed = store.prune_model_bundles({"20260401T120000Z"})

        assert model_files["point"].endswith("point.txt")
        assert removed == ["20240101T120000Z"]
        assert not old_dir.exists()
        assert recent_dir.exists()
