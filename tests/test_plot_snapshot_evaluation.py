"""Tests for predictor.plot_snapshot_evaluation module."""

import os

import pandas as pd

from predictor.model.priceregion import PriceRegionName
from predictor.plot_snapshot_evaluation import render_snapshot_plot, select_plot_frame


class TestPlotSnapshotEvaluation:
    def test_select_plot_frame_uses_latest_generation_per_target(self):
        frame = pd.DataFrame(
            {
                "generated_at_utc": pd.to_datetime(
                    [
                        "2026-04-01T00:00:00Z",
                        "2026-04-01T06:00:00Z",
                        "2026-04-01T00:00:00Z",
                    ],
                    utc=True,
                ),
                "target_time_utc": pd.to_datetime(
                    [
                        "2026-04-02T00:00:00Z",
                        "2026-04-02T00:00:00Z",
                        "2026-04-02T00:15:00Z",
                    ],
                    utc=True,
                ),
                "lead_minutes": [1440, 1080, 1425],
                "lead_bucket": ["24-48h", "0-24h", "24-48h"],
                "generated_at_window": ["off_window", "primary_window", "off_window"],
                "actual_price": [20.0, 20.0, 21.0],
                "predicted_price": [18.0, 19.0, 22.0],
                "yesterday_baseline": [17.0, 17.0, 18.0],
                "last_week_baseline": [16.0, 16.0, 17.0],
            }
        )

        selected = select_plot_frame(frame, "latest", "all")

        assert len(selected) == 2
        assert selected.iloc[0]["generated_at_utc"] == pd.Timestamp("2026-04-01T06:00:00Z")
        assert selected.iloc[0]["predicted_price"] == 19.0

    def test_select_plot_frame_respects_bucket_and_window_filters(self):
        frame = pd.DataFrame(
            {
                "generated_at_utc": pd.to_datetime(
                    [
                        "2026-04-01T00:00:00Z",
                        "2026-04-01T07:00:00Z",
                        "2026-04-01T08:00:00Z",
                    ],
                    utc=True,
                ),
                "target_time_utc": pd.to_datetime(
                    [
                        "2026-04-02T00:00:00Z",
                        "2026-04-02T00:00:00Z",
                        "2026-04-02T00:15:00Z",
                    ],
                    utc=True,
                ),
                "lead_minutes": [1440, 1080, 1095],
                "lead_bucket": ["24-48h", "0-24h", "0-24h"],
                "generated_at_window": ["off_window", "primary_window", "primary_window"],
                "actual_price": [20.0, 20.0, 21.0],
                "predicted_price": [18.0, 19.0, 22.0],
                "yesterday_baseline": [17.0, 17.0, 18.0],
                "last_week_baseline": [16.0, 16.0, 17.0],
            }
        )

        selected = select_plot_frame(frame, "0-24h", "primary_window")

        assert len(selected) == 2
        assert set(selected["lead_bucket"]) == {"0-24h"}
        assert set(selected["generated_at_window"]) == {"primary_window"}

    def test_render_snapshot_plot_writes_png(self, temp_storage_dir):
        region = PriceRegionName.FI.to_region()
        frame = pd.DataFrame(
            {
                "target_time_utc": pd.to_datetime(
                    [
                        "2026-04-02T00:00:00Z",
                        "2026-04-02T00:15:00Z",
                        "2026-04-02T00:30:00Z",
                        "2026-04-02T00:45:00Z",
                    ],
                    utc=True,
                ),
                "actual_price": [20.0, 21.0, 19.5, 18.0],
                "predicted_price": [19.0, 20.5, 20.0, 17.5],
                "yesterday_baseline": [18.0, 19.0, 18.5, 17.0],
                "last_week_baseline": [17.0, 18.0, 17.5, 16.0],
            }
        )
        output_file = f"{temp_storage_dir}/snapshot_plot.png"

        metrics = render_snapshot_plot(
            frame,
            region,
            "snapshots",
            "latest",
            "all",
            output_file,
            width=800,
            height=400,
            transparent=False,
        )

        assert metrics["model"] is not None
        assert metrics["model"]["count"] == 4
        assert os.path.exists(output_file)
        with open(output_file, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
