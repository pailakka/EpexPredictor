from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .priceregion import PriceRegion


class DataFrameSnapshotStore:
    """
    Append-only issue-time storage for wide data frames indexed by target time.
    """

    def __init__(
        self,
        region: PriceRegion,
        storage_dir: str | None,
        prefix: str,
        extra_columns: Iterable[str] | None = None,
    ):
        self.region = region
        self.storage_dir = storage_dir
        self.prefix = prefix
        self.base_columns = ["generated_at_utc", "target_time_utc", "region"]
        self.extra_columns = list(extra_columns or [])

    @property
    def required_columns(self) -> list[str]:
        return self.base_columns + self.extra_columns

    def get_storage_file(self) -> str | None:
        if self.storage_dir is None:
            return None
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)
        return f"{self.storage_dir}/{self.prefix}_{self.region.bidding_zone_entsoe}.csv.gz"

    def load(self) -> pd.DataFrame:
        fn = self.get_storage_file()
        if fn is None or not os.path.exists(fn):
            return pd.DataFrame(columns=self.required_columns)

        df = pd.read_csv(
            fn,
            compression="gzip",
            parse_dates=["generated_at_utc", "target_time_utc"],
        )
        return self._normalize_dataframe(df)

    def append_frame(
        self,
        frame: pd.DataFrame,
        generated_at: datetime,
        known_until: datetime | None = None,
        extra_values: dict[str, object] | None = None,
    ) -> int:
        fn = self.get_storage_file()
        if fn is None or frame.empty:
            return 0

        generated_at_ts = self._to_utc_timestamp(generated_at)
        known_until_ts = self._to_utc_timestamp(known_until) if known_until is not None else generated_at_ts

        future_frame = frame[frame.index > known_until_ts].copy()
        if future_frame.empty:
            return 0

        snapshot = future_frame.reset_index()
        snapshot.rename(columns={snapshot.columns[0]: "target_time_utc"}, inplace=True)
        snapshot["generated_at_utc"] = generated_at_ts
        snapshot["region"] = self.region.bidding_zone_entsoe
        for key in self.extra_columns:
            snapshot[key] = (extra_values or {}).get(key)

        combined = pd.concat([self.load(), snapshot], ignore_index=True)
        retention_cutoff = generated_at_ts - pd.Timedelta(days=self.region.retention_days)
        combined = combined[combined["generated_at_utc"] >= retention_cutoff]
        combined = self._normalize_dataframe(combined)
        combined.sort_values(["generated_at_utc", "target_time_utc"], inplace=True)
        combined.to_csv(fn, index=False, compression="gzip")
        return len(snapshot)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.reindex(columns=self.required_columns)

        normalized = df.copy()
        for column in self.required_columns:
            if column not in normalized.columns:
                normalized[column] = pd.NA
        normalized["generated_at_utc"] = pd.to_datetime(normalized["generated_at_utc"], utc=True)
        normalized["target_time_utc"] = pd.to_datetime(normalized["target_time_utc"], utc=True)
        normalized["region"] = normalized["region"].astype(str)
        for column in self.extra_columns:
            if column.endswith("_utc") and column in normalized.columns:
                normalized[column] = pd.to_datetime(normalized[column], utc=True)
            elif column == "model_version":
                normalized[column] = normalized[column].astype("string")
        return normalized[self.required_columns + [column for column in normalized.columns if column not in self.required_columns]]

    def _to_utc_timestamp(self, value: datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
