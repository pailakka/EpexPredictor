"""Tests for predictor.evaluate_predictions module."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from predictor.evaluate_predictions import (
    build_evaluation_frame,
    load_snapshot_predictions,
    render_text_report,
    summarize_evaluation,
)
from predictor.model.predictionsnapshotstore import PredictionSnapshotStore
from predictor.model.priceregion import PriceRegionName


class TestEvaluatePredictions:
    """Tests for snapshot-backed prediction evaluation."""

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
