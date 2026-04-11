"""Tests for predictor.model.dataframesnapshotstore module."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from predictor.model.dataframesnapshotstore import DataFrameSnapshotStore
from predictor.model.priceregion import PriceRegionName


class TestDataFrameSnapshotStore:
    def test_append_frame_persists_model_version(self, temp_storage_dir):
        store = DataFrameSnapshotStore(
            PriceRegionName.FI.to_region(),
            temp_storage_dir,
            "predictor_snapshots_v1",
            extra_columns=["model_version", "train_start_utc", "train_end_utc"],
        )
        generated_at = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        known_until = generated_at
        index = pd.DatetimeIndex(
            [
                generated_at + timedelta(minutes=15),
                generated_at + timedelta(minutes=30),
            ]
        )
        frame = pd.DataFrame({"market_load_forecast": [100.0, 101.0]}, index=index)

        stored = store.append_frame(
            frame,
            generated_at,
            known_until,
            {
                "model_version": "20260401T120000Z",
                "train_start_utc": generated_at - timedelta(days=120),
                "train_end_utc": generated_at + timedelta(days=7),
            },
        )

        loaded = store.load()
        assert stored == 2
        assert len(loaded) == 2
        assert loaded["model_version"].tolist() == ["20260401T120000Z", "20260401T120000Z"]

    def test_load_legacy_frame_without_model_version_column(self, temp_storage_dir):
        store = DataFrameSnapshotStore(
            PriceRegionName.FI.to_region(),
            temp_storage_dir,
            "predictor_snapshots_v1",
            extra_columns=["model_version", "train_start_utc", "train_end_utc"],
        )
        path = store.get_storage_file()
        assert path is not None

        legacy = pd.DataFrame(
            {
                "generated_at_utc": ["2026-04-01T12:00:00Z"],
                "target_time_utc": ["2026-04-01T12:30:00Z"],
                "region": ["FI"],
                "train_start_utc": ["2025-12-01T12:00:00Z"],
                "train_end_utc": ["2026-04-08T12:00:00Z"],
                "market_load_forecast": [100.0],
            }
        )
        legacy.to_csv(path, index=False, compression="gzip")

        loaded = store.load()

        assert "model_version" in loaded.columns
        assert pd.isna(loaded.iloc[0]["model_version"])
