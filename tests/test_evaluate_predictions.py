"""Tests for predictor.evaluate_predictions module."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import predictor.evaluate_predictions as evaluate_predictions_module
from predictor.evaluate_predictions import (
    build_evaluation_frame,
    generate_backtest_predictions,
    load_snapshot_predictions,
    render_text_report,
    summarize_evaluation,
)
from predictor.model.predictionsnapshotstore import PredictionSnapshotStore
from predictor.model.priceregion import PriceRegionName


class TestEvaluatePredictions:
    """Tests for snapshot-backed prediction evaluation."""

    @pytest.mark.asyncio
    async def test_generate_backtest_predictions_includes_model_version_and_reports_progress(self, monkeypatch):
        region = PriceRegionName.FI.to_region()

        class DummyDataStore:
            def __init__(self):
                self.horizon_cutoff = None

            async def get_data(self, *_args, **_kwargs):
                return pd.DataFrame()

        class DummyPredictor:
            def __init__(self, predictor_region, storage_dir):
                self.region = predictor_region
                self.storage_dir = storage_dir
                self.weatherstore = DummyDataStore()
                self.pricestore = DummyDataStore()
                self.entsoestore = DummyDataStore()
                self.marketstore = DummyDataStore()
                self.auxstore = DummyDataStore()
                self.gasstore = DummyDataStore()
                self.model_version = None

            async def load_from_persistence(self):
                return self

            async def train(self, _start, end):
                generated_at = end + timedelta(minutes=15)
                self.model_version = f"version-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"

            async def predict(self, start, _end, fill_known=True, generated_at=None):
                assert fill_known is False
                assert generated_at == start
                index = pd.DatetimeIndex([pd.Timestamp(start) + timedelta(days=1)])
                return pd.DataFrame({"price": [42.0]}, index=index)

        monkeypatch.setattr(evaluate_predictions_module, "PricePredictor", DummyPredictor)
        monkeypatch.setattr(evaluate_predictions_module, "PREDICTION_HORIZON_DAYS", 1)

        progress_messages: list[str] = []
        predictions = await generate_backtest_predictions(
            region,
            datetime(2026, 4, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 3, 0, 0, tzinfo=timezone.utc),
            training_window_days=120,
            storage_dir=None,
            progress=progress_messages.append,
        )

        assert predictions["model_version"].tolist() == [
            "version-20260401T000000Z",
            "version-20260402T000000Z",
        ]
        assert any("Preloading backtest inputs" in message for message in progress_messages)
        assert any("[1/3] Training backtest model" in message for message in progress_messages)
        assert progress_messages[-1] == "Backtest generation complete: 2 rows in evaluation window"

    def test_snapshot_history_report_contains_baselines_and_buckets(self, temp_storage_dir):
        region = PriceRegionName.FI.to_region()
        store = PredictionSnapshotStore(region, temp_storage_dir)
        generated_at = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        target_index = pd.date_range(
            start="2026-04-01T00:15:00Z",
            periods=4,
            freq="24h",
            tz="UTC",
        )
        predictions = pd.DataFrame({"price": [10.5, 20.5, 30.5, 40.5]}, index=target_index)
        store.append_predictions(
            predictions,
            generated_at,
            generated_at,
            "20260401T000000Z",
            generated_at - timedelta(days=120),
            generated_at + timedelta(days=7),
        )

        actual_index = pd.date_range(
            start="2026-03-25T00:15:00Z",
            end="2026-04-04T00:15:00Z",
            freq="24h",
            tz="UTC",
        )
        actual_prices = pd.DataFrame(
            {"price": [8.0, 9.0, 10.0, 11.0, 12.0, 20.0, 31.0, 45.0, 55.0, 65.0, 75.0]},
            index=actual_index,
        )

        loaded = load_snapshot_predictions(
            region,
            temp_storage_dir,
            datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 4, 0, 15, tzinfo=timezone.utc),
        )
        frame = build_evaluation_frame(loaded, actual_prices, region)
        report = summarize_evaluation(frame)
        text = render_text_report(
            {
                **report,
                "region": region.bidding_zone_entsoe,
                "source": "snapshots",
                "eval_start_utc": "2026-04-01T00:00:00+00:00",
                "eval_end_utc": "2026-04-04T00:15:00+00:00",
                "training_window_days": 120,
            }
        )

        assert report["overall"]["model"] is not None
        assert report["overall"]["yesterday_baseline"] is not None
        assert report["overall"]["last_week_baseline"] is not None
        assert "0-24h" in report["lead_buckets"]
        assert "24-48h" in report["lead_buckets"]
        assert "48-72h" in report["lead_buckets"]
        assert "72h+" in report["lead_buckets"]
        assert "off_window" in report["generated_at_windows"]
        assert report["daily_worst_dates"]
        assert "Overall metrics:" in text
        assert "model" in text
        assert "yesterday_baseline" in text
        assert "last_week_baseline" in text
        assert "Lead buckets:" in text
        assert "Generated-at windows:" in text
