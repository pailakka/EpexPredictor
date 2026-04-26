#!/usr/bin/python3

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone

from predictor.evaluate_predictions import (
    ProgressReporter,
    build_evaluation_frame,
    emit_progress,
    generate_backtest_predictions,
    load_actual_prices,
    load_snapshot_predictions,
    make_stderr_progress_reporter,
    metric_summary,
)
from predictor.model.priceregion import PriceRegion, PriceRegionName

DEFAULT_STORAGE_DIR = os.getenv("EPEXPREDICTOR_DATADIR", "./data")


def select_plot_frame(
    evaluation_frame: pd.DataFrame,
    selection: str,
    generated_at_window: str,
) -> pd.DataFrame:
    selected = evaluation_frame.copy()
    if selected.empty:
        return selected

    if generated_at_window != "all":
        selected = selected[selected["generated_at_window"] == generated_at_window].copy()

    if selection != "latest":
        selected = selected[selected["lead_bucket"] == selection].copy()

    if selected.empty:
        return selected

    selected = selected.sort_values(
        ["target_time_utc", "generated_at_utc", "lead_minutes"],
        ascending=[True, False, True],
    )
    selected = selected.drop_duplicates(subset=["target_time_utc"], keep="first")
    return selected.sort_values("target_time_utc").reset_index(drop=True)


def summarize_plot_frame(frame: pd.DataFrame) -> dict[str, dict[str, float | int] | None]:
    return {
        "model": metric_summary(frame["actual_price"], frame["predicted_price"]),
        "yesterday_baseline": metric_summary(frame["actual_price"], frame["yesterday_baseline"]),
        "last_week_baseline": metric_summary(frame["actual_price"], frame["last_week_baseline"]),
    }


def selection_label(selection: str) -> str:
    if selection == "latest":
        return "latest available forecast per target"
    return f"latest forecast within lead bucket {selection}"


def source_label(source: str) -> str:
    if source == "backtest":
        return "rolling backtest"
    return "live snapshot"


def format_metric(metric: dict[str, float | int] | None) -> str:
    if metric is None:
        return "n/a"
    return f"MAE {metric['mae']:.2f}, RMSE {metric['rmse']:.2f}, n={metric['count']}"


def render_snapshot_plot(
    frame: pd.DataFrame,
    region: PriceRegion,
    source: str,
    selection: str,
    generated_at_window: str,
    output_file: str,
    width: int,
    height: int,
    transparent: bool,
    timezone_name: str | None = None,
) -> dict[str, dict[str, float | int] | None]:
    if frame.empty:
        raise ValueError("No snapshot rows available for the requested filters")

    metrics = summarize_plot_frame(frame)
    target_timezone = timezone_name or region.timezone
    plot_times = frame["target_time_utc"].dt.tz_convert(target_timezone)

    model_abs_error = (frame["predicted_price"] - frame["actual_price"]).abs()
    yesterday_abs_error = (frame["yesterday_baseline"] - frame["actual_price"]).abs()
    week_abs_error = (frame["last_week_baseline"] - frame["actual_price"]).abs()

    fig, (ax_price, ax_error) = plt.subplots(
        2,
        1,
        figsize=(width / 100, height / 100),
        dpi=100,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    ax_price.plot(plot_times, frame["actual_price"], color="black", linewidth=2.2, label="Actual")
    ax_price.plot(plot_times, frame["predicted_price"], color="#1565c0", linewidth=1.6, label="Model")
    if frame["yesterday_baseline"].notna().any():
        ax_price.plot(
            plot_times,
            frame["yesterday_baseline"],
            color="#ef6c00",
            linewidth=1.2,
            linestyle="--",
            label="Yesterday baseline",
        )
    if frame["last_week_baseline"].notna().any():
        ax_price.plot(
            plot_times,
            frame["last_week_baseline"],
            color="#6d6d6d",
            linewidth=1.1,
            linestyle=":",
            label="Last-week baseline",
        )
    ax_price.set_ylabel("ct/kWh")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    ax_error.plot(plot_times, model_abs_error, color="#1565c0", linewidth=1.5, label="Model |error|")
    if frame["yesterday_baseline"].notna().any():
        ax_error.plot(
            plot_times,
            yesterday_abs_error,
            color="#ef6c00",
            linewidth=1.1,
            linestyle="--",
            label="Yesterday |error|",
        )
    if frame["last_week_baseline"].notna().any():
        ax_error.plot(
            plot_times,
            week_abs_error,
            color="#6d6d6d",
            linewidth=1.0,
            linestyle=":",
            label="Last-week |error|",
        )
    ax_error.set_ylabel("|error|")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(loc="upper left")

    generated_window_label = "all generated-at windows" if generated_at_window == "all" else generated_at_window
    fig.suptitle(f"{region.bidding_zone_entsoe} {source_label(source)} comparison", fontsize=14, fontweight="bold")
    ax_price.set_title(
        f"{selection_label(selection)}; {generated_window_label}; "
        f"model {format_metric(metrics['model'])}; "
        f"yesterday {format_metric(metrics['yesterday_baseline'])}; "
        f"last-week {format_metric(metrics['last_week_baseline'])}",
        fontsize=9.5,
    )

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_error.xaxis.set_major_locator(locator)
    ax_error.xaxis.set_major_formatter(formatter)
    ax_error.set_xlabel(f"Target time ({target_timezone})")

    output_path = Path(output_file)
    if output_path.parent != Path():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", transparent=transparent)
    plt.close(fig)
    return metrics


def describe_time_coverage(label: str, times: pd.Series | pd.Index) -> str:
    if len(times) == 0:
        return f"{label}: empty"
    values = pd.to_datetime(times, utc=True)
    return f"{label}: {values.min().isoformat()} -> {values.max().isoformat()}"


async def load_predictions(
    source: str,
    region: PriceRegion,
    eval_start: datetime,
    eval_end: datetime,
    storage_dir: str | None,
    training_window_days: int,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    if source == "backtest":
        return await generate_backtest_predictions(
            region,
            eval_start,
            eval_end,
            training_window_days,
            storage_dir,
            progress=progress,
        )
    emit_progress(progress, "Loading stored snapshot predictions")
    predictions = load_snapshot_predictions(region, storage_dir, eval_start, eval_end)
    emit_progress(progress, f"Loaded {len(predictions)} snapshot prediction rows")
    return predictions


async def generate_snapshot_plot(
    region: PriceRegion,
    eval_start: datetime,
    eval_end: datetime,
    storage_dir: str | None,
    source: str,
    training_window_days: int,
    selection: str,
    generated_at_window: str,
    output_file: str,
    width: int,
    height: int,
    transparent: bool,
    timezone_name: str | None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    emit_progress(progress, f"Preparing {source} plot for {region.bidding_zone_entsoe}")
    predictions = await load_predictions(
        source,
        region,
        eval_start,
        eval_end,
        storage_dir,
        training_window_days,
        progress=progress,
    )
    if predictions.empty:
        if source == "backtest":
            raise ValueError("No backtest rows were generated for the requested target window.")
        raise ValueError(
            "No snapshot rows found for the requested target window. "
            "Wait until the worker has stored snapshots for that range, or choose a window that exists in "
            f"{Path(storage_dir or DEFAULT_STORAGE_DIR) / f'prediction_snapshots_v1_{region.bidding_zone_entsoe}.csv.gz'}."
        )

    emit_progress(progress, "Loading actual prices for plotting")
    actual_prices = await load_actual_prices(region, storage_dir, eval_start, eval_end)
    emit_progress(progress, "Building evaluation frame for plotting")
    evaluation_frame = build_evaluation_frame(predictions, actual_prices, region)
    if evaluation_frame.empty:
        if source == "backtest":
            raise ValueError(
                "Backtest rows were generated, but none overlap with actual prices. "
                f"{describe_time_coverage('backtest targets', predictions['target_time_utc'])}; "
                f"{describe_time_coverage('actual prices', actual_prices.index)}."
            )
        raise ValueError(
            "Snapshot rows exist, but none overlap with actual prices yet. "
            f"{describe_time_coverage('snapshot targets', predictions['target_time_utc'])}; "
            f"{describe_time_coverage('actual prices', actual_prices.index)}."
        )

    selected_frame = select_plot_frame(evaluation_frame, selection, generated_at_window)
    if selected_frame.empty:
        raise ValueError(
            "Snapshots and actual prices overlap, but the selected filters removed all rows. "
            f"Selection={selection}, generated_at_window={generated_at_window}."
        )

    emit_progress(progress, f"Rendering plot with {len(selected_frame)} rows")
    metrics = render_snapshot_plot(
        selected_frame,
        region,
        source,
        selection,
        generated_at_window,
        output_file,
        width,
        height,
        transparent,
        timezone_name,
    )
    emit_progress(progress, f"Plot written to {Path(output_file).resolve()}")
    return {
        "output_file": str(Path(output_file).resolve()),
        "rows_plotted": int(len(selected_frame)),
        "source": source,
        "training_window_days": training_window_days,
        "selection": selection,
        "generated_at_window": generated_at_window,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a prediction-vs-actual PNG plot from snapshots or backtest output.")
    parser.add_argument("--region", required=True, choices=[region.value for region in PriceRegionName])
    parser.add_argument("--eval-start", required=True, help="Start of target evaluation window in ISO-8601 format.")
    parser.add_argument("--eval-end", required=True, help="End of target evaluation window in ISO-8601 format.")
    parser.add_argument("--output-file", required=True, help="PNG file to write.")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--source", choices=["snapshots", "backtest"], default="snapshots")
    parser.add_argument("--training-window-days", type=int, default=120)
    parser.add_argument(
        "--selection",
        choices=["latest", "0-24h", "24-48h", "48-72h", "72h+"],
        default="latest",
        help="How to choose one snapshot per target timestamp.",
    )
    parser.add_argument(
        "--generated-at-window",
        choices=["all", "primary_window", "off_window"],
        default="all",
        help="Optional filter on generated-at time window.",
    )
    parser.add_argument("--timezone", default=None, help="Timezone used on the x-axis. Defaults to the region timezone.")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--transparent", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    eval_start = datetime.fromisoformat(args.eval_start.replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(args.eval_end.replace("Z", "+00:00"))
    region = PriceRegionName(args.region).to_region()
    progress = make_stderr_progress_reporter()

    try:
        result = await generate_snapshot_plot(
            region,
            eval_start,
            eval_end,
            args.storage_dir,
            args.source,
            args.training_window_days,
            args.selection,
            args.generated_at_window,
            args.output_file,
            args.width,
            args.height,
            args.transparent,
            args.timezone,
            progress=progress,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    metrics = result["metrics"]
    print(f"Wrote {result['output_file']}")
    print(f"Rows plotted: {result['rows_plotted']}")
    print(f"Source: {result['source']}")
    print(f"Training window days: {result['training_window_days']}")
    print(f"Selection: {result['selection']}")
    print(f"Generated-at window: {result['generated_at_window']}")
    print(f"Model: {format_metric(metrics['model'])}")
    print(f"Yesterday baseline: {format_metric(metrics['yesterday_baseline'])}")
    print(f"Last-week baseline: {format_metric(metrics['last_week_baseline'])}")


if __name__ == "__main__":
    asyncio.run(main())
