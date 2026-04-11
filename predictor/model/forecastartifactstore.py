from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .priceregion import PriceRegion


class ForecastArtifactStore:
    """
    Persist the latest served forecasts and versioned model metadata for a region.
    """

    def __init__(self, region: PriceRegion, storage_dir: str | None):
        self.region = region
        self.storage_dir = storage_dir

    def _ensure_storage_dir(self) -> str | None:
        if self.storage_dir is None:
            return None
        normalized = self.storage_dir.rstrip("/") or self.storage_dir
        os.makedirs(normalized, exist_ok=True)
        return normalized

    def _frame_path(self, kind: str) -> str | None:
        base_dir = self._ensure_storage_dir()
        if base_dir is None:
            return None
        return f"{base_dir}/forecast_latest_{kind}_v1_{self.region.bidding_zone_entsoe}.csv.gz"

    def _metadata_path(self) -> str | None:
        base_dir = self._ensure_storage_dir()
        if base_dir is None:
            return None
        return f"{base_dir}/forecast_latest_metadata_v1_{self.region.bidding_zone_entsoe}.json"

    def _model_dir(self, version: str) -> str | None:
        base_dir = self._ensure_storage_dir()
        if base_dir is None:
            return None
        model_dir = f"{base_dir}/models/{self.region.bidding_zone_entsoe}/{version}"
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def get_model_files(self, version: str) -> dict[str, str]:
        model_dir = self._model_dir(version)
        if model_dir is None or not os.path.isdir(model_dir):
            return {}

        files: dict[str, str] = {}
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".txt"):
                continue
            files[filename[:-4]] = os.path.join(model_dir, filename)
        return files

    def load_latest(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        return (
            self._load_frame("point"),
            self._load_frame("eval"),
            self._load_frame("quantiles"),
            self._load_metadata(),
        )

    def save_latest(
        self,
        point_forecast: pd.DataFrame,
        evaluation_forecast: pd.DataFrame,
        quantiles: pd.DataFrame | None,
        metadata: dict[str, Any],
        model_files: dict[str, str] | None = None,
    ) -> None:
        point_path = self._frame_path("point")
        eval_path = self._frame_path("eval")
        quantile_path = self._frame_path("quantiles")
        metadata_path = self._metadata_path()
        if point_path is None or eval_path is None or metadata_path is None:
            return

        self._write_frame(point_forecast, point_path)
        self._write_frame(evaluation_forecast, eval_path)
        if quantiles is not None and not quantiles.empty and quantile_path is not None:
            self._write_frame(quantiles, quantile_path)
        elif quantile_path is not None and os.path.exists(quantile_path):
            os.remove(quantile_path)

        payload = dict(metadata)
        if model_files:
            payload["model_files"] = model_files
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def save_model_bundle(self, version: str, models: dict[str, Any]) -> dict[str, str]:
        model_dir = self._model_dir(version)
        if model_dir is None:
            return {}

        files: dict[str, str] = {}
        for name, model in models.items():
            if model is None:
                continue
            path = f"{model_dir}/{name}.txt"
            model.save_model(path)
            files[name] = path
        return files

    def prune_model_bundles(self, referenced_versions: set[str]) -> list[str]:
        base_dir = self._ensure_storage_dir()
        if base_dir is None:
            return []

        models_dir = f"{base_dir}/models/{self.region.bidding_zone_entsoe}"
        if not os.path.isdir(models_dir):
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.region.retention_days)
        removed: list[str] = []
        for version in os.listdir(models_dir):
            version_dir = os.path.join(models_dir, version)
            if not os.path.isdir(version_dir) or version in referenced_versions:
                continue
            version_ts = self._parse_version_timestamp(version)
            if version_ts is None or version_ts >= cutoff:
                continue
            for root, dirs, files in os.walk(version_dir, topdown=False):
                for filename in files:
                    os.remove(os.path.join(root, filename))
                for dirname in dirs:
                    os.rmdir(os.path.join(root, dirname))
            os.rmdir(version_dir)
            removed.append(version)
        return sorted(removed)

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._metadata_path()
        if metadata_path is None or not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_frame(self, kind: str) -> pd.DataFrame:
        path = self._frame_path(kind)
        if path is None or not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, compression="gzip", parse_dates=["target_time_utc"])
        df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True)
        df.set_index("target_time_utc", inplace=True)
        df.index.name = "time"
        return df

    def _write_frame(self, frame: pd.DataFrame, path: str) -> None:
        if frame.empty:
            return
        out = frame.copy()
        out = out.reset_index()
        out.rename(columns={out.columns[0]: "target_time_utc"}, inplace=True)
        out.to_csv(path, index=False, compression="gzip")

    def _parse_version_timestamp(self, version: str) -> datetime | None:
        try:
            return datetime.strptime(version, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
