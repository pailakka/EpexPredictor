"""Tests for predictor.model.predictionsnapshotstore module."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from predictor.model.predictionsnapshotstore import PredictionSnapshotStore
from predictor.model.priceregion import PriceRegionName


class TestPredictionSnapshotStore:
    """Tests for append-only prediction snapshot persistence."""

    def test_append_predictions_only_persists_future_rows(self, temp_storage_dir):
        store = PredictionSnapshotStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        generated_at = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        known_until = generated_at + timedelta(minutes=15)
        index = pd.DatetimeIndex(
            [
                generated_at,
                generated_at + timedelta(minutes=15),
                generated_at + timedelta(minutes=30),
                generated_at + timedelta(minutes=45),
            ]
        )
        predictions = pd.DataFrame({"price": [10.0, 11.0, 12.0, 13.0]}, index=index)

        stored = store.append_predictions(
            predictions,
            generated_at,
            known_until,
            "20260401T120000Z",
            generated_at - timedelta(days=120),
            generated_at + timedelta(days=7),
        )

        loaded = store.load()
        assert stored == 2
        assert len(loaded) == 2
        assert loaded["target_time_utc"].tolist() == [
            pd.Timestamp("2026-04-01T12:30:00Z"),
            pd.Timestamp("2026-04-01T12:45:00Z"),
        ]
        assert loaded["model_version"].tolist() == ["20260401T120000Z", "20260401T120000Z"]

    def test_append_predictions_preserves_multiple_generations(self, temp_storage_dir):
        store = PredictionSnapshotStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        base_generated_at = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        target_index = pd.DatetimeIndex([pd.Timestamp("2026-04-02T12:00:00Z")])

        first = pd.DataFrame({"price": [15.0]}, index=target_index)
        second = pd.DataFrame({"price": [16.0]}, index=target_index)

        store.append_predictions(
            first,
            base_generated_at,
            base_generated_at,
            "20260401T120000Z",
            base_generated_at - timedelta(days=120),
            base_generated_at + timedelta(days=7),
        )
        store.append_predictions(
            second,
            base_generated_at + timedelta(hours=6),
            base_generated_at + timedelta(hours=6),
            "20260401T180000Z",
            base_generated_at - timedelta(days=120),
            base_generated_at + timedelta(days=7),
        )

        loaded = store.load()
        assert len(loaded) == 2
        assert loaded["target_time_utc"].nunique() == 1
        assert loaded["generated_at_utc"].nunique() == 2

    def test_append_predictions_prunes_rows_older_than_retention(self, temp_storage_dir):
        store = PredictionSnapshotStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        old_generated_at = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        recent_generated_at = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        target_index = pd.DatetimeIndex([pd.Timestamp("2026-04-02T12:00:00Z")])

        store.append_predictions(
            pd.DataFrame({"price": [15.0]}, index=target_index),
            old_generated_at,
            old_generated_at,
            "20240101T120000Z",
            old_generated_at - timedelta(days=120),
            old_generated_at + timedelta(days=7),
        )
        store.append_predictions(
            pd.DataFrame({"price": [16.0]}, index=target_index),
            recent_generated_at,
            recent_generated_at,
            "20260401T120000Z",
            recent_generated_at - timedelta(days=120),
            recent_generated_at + timedelta(days=7),
        )

        loaded = store.load()
        assert len(loaded) == 1
        assert loaded.iloc[0]["generated_at_utc"] == pd.Timestamp(recent_generated_at)

    def test_load_old_snapshot_file_without_model_version(self, temp_storage_dir):
        store = PredictionSnapshotStore(PriceRegionName.FI.to_region(), temp_storage_dir)
        path = store.get_storage_file()
        assert path is not None

        legacy = pd.DataFrame(
            {
                "generated_at_utc": ["2026-04-01T12:00:00Z"],
                "target_time_utc": ["2026-04-01T12:30:00Z"],
                "predicted_price": [12.0],
                "lead_minutes": [30],
                "region": ["FI"],
                "train_start_utc": ["2025-12-01T12:00:00Z"],
                "train_end_utc": ["2026-04-08T12:00:00Z"],
            }
        )
        legacy.to_csv(path, index=False, compression="gzip")

        loaded = store.load()

        assert "model_version" in loaded.columns
        assert pd.isna(loaded.iloc[0]["model_version"])
