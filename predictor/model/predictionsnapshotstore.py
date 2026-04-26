from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .priceregion import PriceRegion


class PredictionSnapshotStore:
    """
    Persist generated price forecasts with generation timestamps for later evaluation.
    Repeated forecasts for the same target timestamp are intentionally preserved.
    """

    required_columns = [
        "generated_at_utc",
        "target_time_utc",
        "predicted_price",
        "lead_minutes",
        "region",
        "model_version",
        "train_start_utc",
        "train_end_utc",
    ]

    def __init__(self, region: PriceRegion, storage_dir: str | None):
        self.region = region
        self.storage_dir = storage_dir

    def get_storage_file(self) -> str | None:
        if self.storage_dir is None:
            return None
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
        return f"{self.storage_dir}/prediction_snapshots_v1_{self.region.bidding_zone_entsoe}.csv.gz"

    def load(self) -> pd.DataFrame:
        fn = self.get_storage_file()
        if fn is None or not os.path.exists(fn):
            return pd.DataFrame(columns=self.required_columns)

        df = pd.read_csv(fn, compression="gzip", parse_dates=[
            "generated_at_utc",
            "target_time_utc",
            "train_start_utc",
            "train_end_utc",
        ])
        return self._normalize_dataframe(df)

    def append_predictions(
        self,
        predictions: pd.DataFrame,
        generated_at: datetime,
        known_until: datetime | None,
        model_version: str | None,
        train_start: datetime,
        train_end: datetime,
    ) -> int:
        fn = self.get_storage_file()
        if fn is None or predictions.empty:
            return 0

        generated_at_ts = self._to_utc_timestamp(generated_at)
        known_until_ts = self._to_utc_timestamp(known_until) if known_until is not None else generated_at_ts
        future_predictions = predictions[predictions.index > known_until_ts]
        if future_predictions.empty:
            return 0

        snapshot = pd.DataFrame(
            {
                "generated_at_utc": generated_at_ts,
                "target_time_utc": future_predictions.index.tz_convert("UTC"),
                "predicted_price": future_predictions["price"].astype(float).to_numpy(),
                "lead_minutes": (
                    (future_predictions.index - generated_at_ts).total_seconds() / 60.0
                ).round().astype(int),
                "region": self.region.bidding_zone_entsoe,
                "model_version": model_version,
                "train_start_utc": self._to_utc_timestamp(train_start),
                "train_end_utc": self._to_utc_timestamp(train_end),
            }
        )

        combined = pd.concat([self.load(), snapshot], ignore_index=True)
        retention_cutoff = generated_at_ts - timedelta(days=self.region.retention_days)
        combined = combined[combined["generated_at_utc"] >= retention_cutoff]
        combined = self._normalize_dataframe(combined)
        combined.sort_values(["generated_at_utc", "target_time_utc"], inplace=True)
        combined.to_csv(fn, index=False, compression="gzip")
        return len(snapshot)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.reindex(columns=self.required_columns)

        normalized = df.copy()
        for col in ["generated_at_utc", "target_time_utc", "train_start_utc", "train_end_utc"]:
            normalized[col] = pd.to_datetime(normalized[col], utc=True)

        normalized["predicted_price"] = pd.to_numeric(normalized["predicted_price"], errors="coerce")
        normalized["lead_minutes"] = pd.to_numeric(normalized["lead_minutes"], errors="coerce").astype("Int64")
        normalized["region"] = normalized["region"].astype(str)
        if "model_version" not in normalized.columns:
            normalized["model_version"] = pd.Series(pd.NA, index=normalized.index, dtype="string")
        normalized["model_version"] = normalized["model_version"].astype("string")
        return normalized[self.required_columns]

    def _to_utc_timestamp(self, value: datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
