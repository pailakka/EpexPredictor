#!/usr/bin/python3

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from predictor.model.predictionsnapshotstore import PredictionSnapshotStore
from predictor.model.pricepredictor import PricePredictor
from predictor.model.priceregion import PriceRegion, PriceRegionName

PREDICTION_HORIZON_DAYS = 7
DEFAULT_STORAGE_DIR = os.getenv("EPEXPREDICTOR_DATADIR", "./data")


def to_utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def metric_summary(actual: pd.Series, prediction: pd.Series) -> dict[str, float | int] | None:
    frame = pd.DataFrame({"actual": actual, "prediction": prediction}).dropna()
    if frame.empty:
        return None

    diff = frame["prediction"] - frame["actual"]
    return {
        "count": int(len(frame)),
        "mae": float(diff.abs().mean()),
        "rmse": float(math.sqrt((diff.pow(2)).mean())),
    }


def assign_lead_bucket(lead_minutes: pd.Series) -> pd.Series:
    def _bucket(value: Any) -> str:
        if pd.isna(value):
            return "unknown"
        lead_hours = float(value) / 60.0
        if lead_hours < 24.0:
            return "0-24h"
        if lead_hours < 48.0:
            return "24-48h"
        if lead_hours < 72.0:
            return "48-72h"
        return "72h+"

    return lead_minutes.map(_bucket)


def assign_volatility_buckets(frame: pd.DataFrame) -> pd.Series:
    target_dates = frame["target_time_utc"].dt.floor("D")
    daily_volatility = frame.groupby(target_dates)["actual_price"].std().fillna(0.0)
    thresholds = daily_volatility.quantile([0.25, 0.5, 0.75]).to_dict()

    def _bucket(day: pd.Timestamp) -> str:
        value = float(daily_volatility.loc[day])
        q25 = float(thresholds.get(0.25, value))
        q50 = float(thresholds.get(0.5, value))
        q75 = float(thresholds.get(0.75, value))
        if value <= q25:
            return "low"
        if value <= q50:
            return "mid"
        if value <= q75:
            return "high"
        return "very_high"

    return target_dates.map(_bucket)


def assign_generated_at_window(frame: pd.DataFrame, region: PriceRegion) -> pd.Series:
    timezone = region.get_timezone_info()
    local_hours = frame["generated_at_utc"].dt.tz_convert(timezone).dt.hour
    start_hour, end_hour = region.primary_generated_at_window_local
    return local_hours.map(lambda hour: "primary_window" if start_hour <= hour < end_hour else "off_window")


def build_evaluation_frame(predictions: pd.DataFrame, actual_prices: pd.DataFrame, region: PriceRegion) -> pd.DataFrame:
    frame = predictions.copy()
    if frame.empty:
        return frame

    actual_series = actual_prices["price"].sort_index()
    frame["generated_at_utc"] = pd.to_datetime(frame["generated_at_utc"], utc=True)
    frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True)
    frame["predicted_price"] = pd.to_numeric(frame["predicted_price"], errors="coerce")

    frame["actual_price"] = actual_series.reindex(frame["target_time_utc"]).to_numpy()
    frame["yesterday_baseline"] = actual_series.reindex(
        frame["target_time_utc"] - pd.Timedelta(days=1)
    ).to_numpy()
    frame["last_week_baseline"] = actual_series.reindex(
        frame["target_time_utc"] - pd.Timedelta(days=7)
    ).to_numpy()

    frame = frame.dropna(subset=["actual_price", "predicted_price"]).copy()
    if frame.empty:
        return frame

    frame["lead_bucket"] = assign_lead_bucket(frame["lead_minutes"])
    frame["target_date"] = frame["target_time_utc"].dt.floor("D")
    frame["volatility_bucket"] = assign_volatility_buckets(frame)
    frame["generated_at_window"] = assign_generated_at_window(frame, region)
    frame["abs_error"] = (frame["predicted_price"] - frame["actual_price"]).abs()
    return frame


def summarize_evaluation(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows_evaluated": int(len(frame)),
        "overall": {},
        "lead_buckets": {},
        "volatility_buckets": {},
        "generated_at_windows": {},
        "daily_worst_dates": [],
    }
    if frame.empty:
        return report

    series_map = {
        "model": frame["predicted_price"],
        "yesterday_baseline": frame["yesterday_baseline"],
        "last_week_baseline": frame["last_week_baseline"],
    }

    for name, series in series_map.items():
        report["overall"][name] = metric_summary(frame["actual_price"], series)

    for bucket, bucket_df in frame.groupby("lead_bucket"):
        report["lead_buckets"][bucket] = {}
        for name, series in series_map.items():
            report["lead_buckets"][bucket][name] = metric_summary(bucket_df["actual_price"], series.loc[bucket_df.index])

    for bucket, bucket_df in frame.groupby("volatility_bucket"):
        report["volatility_buckets"][bucket] = {}
        for name, series in series_map.items():
            report["volatility_buckets"][bucket][name] = metric_summary(bucket_df["actual_price"], series.loc[bucket_df.index])

    for bucket, bucket_df in frame.groupby("generated_at_window"):
        report["generated_at_windows"][bucket] = {}
        for name, series in series_map.items():
            report["generated_at_windows"][bucket][name] = metric_summary(bucket_df["actual_price"], series.loc[bucket_df.index])

    daily = frame.groupby("target_date").agg(
        model_mae=("abs_error", "mean"),
        model_rmse=("abs_error", lambda s: float(math.sqrt((s.pow(2)).mean()))),
        max_abs_error=("abs_error", "max"),
        observations=("abs_error", "size"),
        daily_volatility=("actual_price", "std"),
    )
    daily["daily_volatility"] = daily["daily_volatility"].fillna(0.0)
    daily = daily.sort_values(["model_mae", "max_abs_error"], ascending=False).head(5)
    for day, row in daily.iterrows():
        report["daily_worst_dates"].append(
            {
                "date": day.date().isoformat(),
                "model_mae": float(row["model_mae"]),
                "model_rmse": float(row["model_rmse"]),
                "max_abs_error": float(row["max_abs_error"]),
                "daily_volatility": float(row["daily_volatility"]),
                "observations": int(row["observations"]),
            }
        )

    return report


async def preload_backtest_data(
    predictor: PricePredictor,
    generation_start: datetime,
    eval_end: datetime,
    training_window_days: int,
) -> None:
    data_start = generation_start - timedelta(days=training_window_days)
    data_end = eval_end + timedelta(days=PREDICTION_HORIZON_DAYS)
    await asyncio.gather(
        predictor.weatherstore.get_data(data_start, data_end),
        predictor.pricestore.get_data(data_start, data_end),
        predictor.entsoestore.get_data(data_start, data_end),
        predictor.marketstore.get_data(data_start, data_end),
        predictor.auxstore.get_data(data_start, data_end),
        predictor.gasstore.get_data(data_start, data_end),
    )


async def generate_backtest_predictions(
    region: PriceRegion,
    eval_start: datetime,
    eval_end: datetime,
    training_window_days: int,
    storage_dir: str | None,
) -> pd.DataFrame:
    generation_start = eval_start - timedelta(days=PREDICTION_HORIZON_DAYS)
    predictor = await PricePredictor(region, storage_dir).load_from_persistence()
    await preload_backtest_data(predictor, generation_start, eval_end, training_window_days)

    all_predictions: list[pd.DataFrame] = []
    generated_at = generation_start
    while generated_at <= eval_end:
        train_start = generated_at - timedelta(days=training_window_days)
        train_end = generated_at - timedelta(minutes=15)
        prediction_end = generated_at + timedelta(days=PREDICTION_HORIZON_DAYS)

        predictor.pricestore.horizon_cutoff = generated_at
        predictor.gasstore.horizon_cutoff = generated_at
        await predictor.train(train_start, train_end)
        prediction = await predictor.predict(generated_at, prediction_end, fill_known=False, generated_at=generated_at)
        predictor.pricestore.horizon_cutoff = None
        predictor.gasstore.horizon_cutoff = None

        if not prediction.empty:
            snapshot = prediction.reset_index()
            snapshot.rename(
                columns={
                    snapshot.columns[0]: "target_time_utc",
                    "price": "predicted_price",
                },
                inplace=True,
            )
            generated_at_ts = to_utc_timestamp(generated_at)
            snapshot["generated_at_utc"] = generated_at
            snapshot["lead_minutes"] = (
                (snapshot["target_time_utc"] - generated_at_ts).dt.total_seconds() / 60.0
            ).round().astype(int)
            snapshot["region"] = region.bidding_zone_entsoe
            snapshot["train_start_utc"] = train_start
            snapshot["train_end_utc"] = train_end
            all_predictions.append(snapshot)

        generated_at += timedelta(days=1)

    if not all_predictions:
        return pd.DataFrame(columns=PredictionSnapshotStore.required_columns)

    combined = pd.concat(all_predictions, ignore_index=True)
    eval_start_ts = to_utc_timestamp(eval_start)
    eval_end_ts = to_utc_timestamp(eval_end)
    combined = combined[
        (combined["target_time_utc"] >= eval_start_ts)
        & (combined["target_time_utc"] <= eval_end_ts)
    ]
    return combined[PredictionSnapshotStore.required_columns]


def load_snapshot_predictions(
    region: PriceRegion,
    storage_dir: str | None,
    eval_start: datetime,
    eval_end: datetime,
) -> pd.DataFrame:
    snapshots = PredictionSnapshotStore(region, storage_dir).load()
    if snapshots.empty:
        return snapshots
    eval_start_ts = to_utc_timestamp(eval_start)
    eval_end_ts = to_utc_timestamp(eval_end)
    return snapshots[
        (snapshots["target_time_utc"] >= eval_start_ts)
        & (snapshots["target_time_utc"] <= eval_end_ts)
    ].copy()


async def load_actual_prices(
    region: PriceRegion,
    storage_dir: str | None,
    eval_start: datetime,
    eval_end: datetime,
) -> pd.DataFrame:
    predictor = await PricePredictor(region, storage_dir).load_from_persistence()
    return await predictor.pricestore.get_data(eval_start - timedelta(days=7), eval_end)


async def evaluate_predictions(
    region: PriceRegion,
    eval_start: datetime,
    eval_end: datetime,
    training_window_days: int,
    storage_dir: str | None,
    source: str,
    primary_window_only: bool,
) -> dict[str, Any]:
    actual_prices = await load_actual_prices(region, storage_dir, eval_start, eval_end)
    if source == "snapshots":
        predictions = load_snapshot_predictions(region, storage_dir, eval_start, eval_end)
    else:
        predictions = await generate_backtest_predictions(
            region,
            eval_start,
            eval_end,
            training_window_days,
            storage_dir,
        )

    frame = build_evaluation_frame(predictions, actual_prices, region)
    if primary_window_only:
        frame = frame[frame["generated_at_window"] == "primary_window"].copy()
    report = summarize_evaluation(frame)
    report.update(
        {
            "region": region.bidding_zone_entsoe,
            "source": source,
            "eval_start_utc": eval_start.astimezone(timezone.utc).isoformat(),
            "eval_end_utc": eval_end.astimezone(timezone.utc).isoformat(),
            "training_window_days": training_window_days,
            "primary_window_only": primary_window_only,
        }
    )
    return report


def format_metric_line(name: str, metric: dict[str, float | int] | None) -> str:
    if metric is None:
        return f"- {name}: n/a"
    return f"- {name}: MAE={metric['mae']:.3f}, RMSE={metric['rmse']:.3f}, count={metric['count']}"


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Region: {report['region']}",
        f"Source: {report['source']}",
        f"Target window: {report['eval_start_utc']} -> {report['eval_end_utc']}",
        f"Training window days: {report['training_window_days']}",
        f"Rows evaluated: {report['rows_evaluated']}",
        "",
        "Overall metrics:",
    ]
    for name, metric in report["overall"].items():
        lines.append(format_metric_line(name, metric))

    lines.append("")
    lines.append("Lead buckets:")
    for bucket in ["0-24h", "24-48h", "48-72h", "72h+", "unknown"]:
        if bucket not in report["lead_buckets"]:
            continue
        lines.append(f"{bucket}:")
        for name, metric in report["lead_buckets"][bucket].items():
            lines.append(f"  {format_metric_line(name, metric)[2:]}")

    lines.append("")
    lines.append("Volatility buckets:")
    for bucket in ["low", "mid", "high", "very_high"]:
        if bucket not in report["volatility_buckets"]:
            continue
        lines.append(f"{bucket}:")
        for name, metric in report["volatility_buckets"][bucket].items():
            lines.append(f"  {format_metric_line(name, metric)[2:]}")

    lines.append("")
    lines.append("Generated-at windows:")
    for bucket in ["primary_window", "off_window"]:
        if bucket not in report["generated_at_windows"]:
            continue
        lines.append(f"{bucket}:")
        for name, metric in report["generated_at_windows"][bucket].items():
            lines.append(f"  {format_metric_line(name, metric)[2:]}")

    lines.append("")
    lines.append("Worst dates:")
    for day in report["daily_worst_dates"]:
        lines.append(
            f"- {day['date']}: MAE={day['model_mae']:.3f}, RMSE={day['model_rmse']:.3f}, "
            f"max_abs_error={day['max_abs_error']:.3f}, daily_volatility={day['daily_volatility']:.3f}, observations={day['observations']}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate electricity price predictions against actual prices.")
    parser.add_argument("--region", required=True, choices=[region.value for region in PriceRegionName])
    parser.add_argument("--eval-start", required=True, help="Start of target evaluation window in ISO-8601 format.")
    parser.add_argument("--eval-end", required=True, help="End of target evaluation window in ISO-8601 format.")
    parser.add_argument("--training-window-days", type=int, default=120)
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    parser.add_argument("--source", choices=["backtest", "snapshots"], default="backtest")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--primary-window-only", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    eval_start = datetime.fromisoformat(args.eval_start.replace("Z", "+00:00"))
    eval_end = datetime.fromisoformat(args.eval_end.replace("Z", "+00:00"))
    region = PriceRegionName(args.region).to_region()

    report = await evaluate_predictions(
        region,
        eval_start,
        eval_end,
        args.training_window_days,
        args.storage_dir,
        args.source,
        args.primary_window_only,
    )

    if args.output_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))


if __name__ == "__main__":
    asyncio.run(main())
