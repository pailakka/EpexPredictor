from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, override

import aiohttp
import pandas as pd
from entsoe import entsoe

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)


class MarketFeatureStore(DataStore):
    """
    Issue-time market features for FI-like coupled markets.

    ENTSO-E is the default source because the project already depends on it.
    Fingrid is used opportunistically when an API key is configured.
    """

    FINGRID_DATASETS = {
        "fingrid_load_forecast": 166,
        "fingrid_wind_forecast": 246,
        "fingrid_solar_forecast": 247,
        "fingrid_capacity_fi_ee": 115,
        "fingrid_capacity_ee_fi": 112,
        "fingrid_imbalance": 397,
        "fingrid_commercial_flow_fi_ee": 140,
        "fingrid_electric_boiler": 371,
    }
    FINGRID_MIN_SECONDS_BETWEEN_REQUESTS = 6.2
    FINGRID_HISTORY_LOOKBACK_DAYS = 2
    FINGRID_FUTURE_LOOKAHEAD_DAYS = 10
    FINGRID_REFRESH_COOLDOWN = timedelta(minutes=15)
    FINGRID_METADATA_KEYS = {
        "created",
        "createdat",
        "datasetid",
        "datasetname",
        "endtime",
        "id",
        "starttime",
        "time",
        "unit",
        "updated",
        "updatedat",
        "value",
        "valuefloat",
        "valuenumber",
    }

    SHADOW_PRICE_AREAS = ["SE_1", "SE_3", "EE", "NO_4"]

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        super().__init__(region, storage_dir, "market_v1")
        self.update_lock = asyncio.Lock()
        self.entsoe_api_key = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY", None) or None
        self.fingrid_api_key = os.getenv("EPEXPREDICTOR_FINGRID_API_KEY", None) or None
        self._logged_fingrid_status = False
        self.last_fingrid_refresh: datetime | None = None

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=3)

    @override
    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        if not self.region.use_market_features:
            return False

        updated = False
        async with self.update_lock:
            for rstart, rend in self.gen_missing_date_ranges(start, end):
                updated = await self.refresh_range(rstart, rend) or updated
        return updated

    @override
    def gen_missing_date_ranges(self, start: datetime, end: datetime) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        range_start: datetime | None = None
        current = start
        while current <= end:
            next_day = current + timedelta(days=1)
            too_long = range_start is not None and (current - range_start).days >= 14
            if range_start is not None and (next_day in self.data.index or next_day > end or too_long):
                result.append((pd.Timestamp(range_start), pd.Timestamp(current)))
                range_start = None

            if range_start is None and current not in self.data.index:
                range_start = current

            current = next_day
        return result

    async def refresh_range(self, rstart: datetime, rend: datetime) -> bool:
        if not self.region.use_market_features:
            return False

        self._log_source_configuration()
        frames: list[pd.DataFrame] = []

        entsoe_frame = await asyncio.to_thread(self._fetch_entsoe_range, rstart, rend)
        if not entsoe_frame.empty:
            frames.append(entsoe_frame)

        fingrid_frame = await self._fetch_fingrid_range(rstart, rend)
        if not fingrid_frame.empty:
            frames.append(fingrid_frame)

        if not frames:
            return False

        df = pd.concat(frames, axis=1).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        updated = self._update_data(df)
        if updated:
            log.info("%s: market features updated", self.region.bidding_zone_entsoe)
            self.data.sort_index(inplace=True)
            await self.serialize()
        return updated

    def _fetch_entsoe_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        if self.entsoe_api_key is None:
            return pd.DataFrame()

        client = entsoe.EntsoePandasClient(api_key=self.entsoe_api_key)
        qstart = pd.Timestamp(rstart - timedelta(days=2))
        qend = pd.Timestamp(rend + timedelta(days=2))

        frames: list[pd.DataFrame] = []
        calls: list[tuple[str, Any]] = [
            (
                "entsoe_load_forecast",
                lambda: client.query_load_forecast(
                    self.region.bidding_zone_entsoe,
                    start=qstart,
                    end=qend,
                    process_type="A01",
                ),
            ),
            (
                "entsoe_generation_forecast",
                lambda: client.query_generation_forecast(
                    self.region.bidding_zone_entsoe,
                    start=qstart,
                    end=qend,
                ),
            ),
            (
                "entsoe_wind_solar",
                lambda: client.query_wind_and_solar_forecast(
                    self.region.bidding_zone_entsoe,
                    start=qstart,
                    end=qend,
                ),
            ),
            ("flow_se3_to_fi", lambda: client.query_crossborder_flows("SE_3", "FI", start=qstart, end=qend)),
            ("flow_fi_to_se3", lambda: client.query_crossborder_flows("FI", "SE_3", start=qstart, end=qend)),
            ("flow_ee_to_fi", lambda: client.query_crossborder_flows("EE", "FI", start=qstart, end=qend)),
            ("flow_fi_to_ee", lambda: client.query_crossborder_flows("FI", "EE", start=qstart, end=qend)),
            ("imbalance_prices", lambda: client.query_imbalance_prices("FI", start=qstart, end=qend)),
        ]

        for shadow_area in self.region.shadow_regions:
            calls.append(
                (
                    f"shadow_price_{shadow_area.lower()}",
                    lambda sa=shadow_area: client.query_day_ahead_prices(
                        sa,
                        start=qstart,
                        end=qend,
                    ),
                )
            )
            calls.append(
                (
                    f"shadow_wind_{shadow_area.lower()}",
                    lambda sa=shadow_area: client.query_wind_and_solar_forecast(
                        sa,
                        start=qstart,
                        end=qend,
                    ),
                )
            )
            calls.append(
                (
                    f"shadow_load_{shadow_area.lower()}",
                    lambda sa=shadow_area: client.query_load_forecast(
                        sa,
                        start=qstart,
                        end=qend,
                        process_type="A01",
                    ),
                )
            )
            calls.append(
                (
                    f"capacity_fi_to_{shadow_area.lower()}",
                    lambda sa=shadow_area: client.query_net_transfer_capacity_dayahead(
                        "FI",
                        sa,
                        start=qstart,
                        end=qend,
                    ),
                )
            )
            calls.append(
                (
                    f"capacity_{shadow_area.lower()}_to_fi",
                    lambda sa=shadow_area: client.query_net_transfer_capacity_dayahead(
                        sa,
                        "FI",
                        start=qstart,
                        end=qend,
                    ),
                )
            )

        for name, call in calls:
            try:
                result = call()
            except Exception as exc:
                log.debug("%s: market feature %s unavailable: %s", self.region.bidding_zone_entsoe, name, exc)
                continue

            frame = self._normalize_entsoe_frame(result, prefix=name)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.loc[pd.Timestamp(rstart):pd.Timestamp(rend)]

    async def _fetch_fingrid_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        if self.region.bidding_zone_entsoe != "FI" or self.fingrid_api_key is None:
            return pd.DataFrame()

        now = datetime.now(timezone.utc)

        # Use file mtime as a persistent cooldown so restarts don't bypass the limit
        effective_last_refresh = self.last_fingrid_refresh or self.get_storage_mtime()
        if effective_last_refresh is not None and now - effective_last_refresh < self.FINGRID_REFRESH_COOLDOWN:
            return pd.DataFrame()

        query_window_start = now - timedelta(days=self.FINGRID_HISTORY_LOOKBACK_DAYS)
        query_window_end = now + timedelta(days=self.FINGRID_FUTURE_LOOKAHEAD_DAYS)
        if rend < query_window_start or rstart > query_window_end:
            return pd.DataFrame()

        query_start = max(rstart - timedelta(days=2), query_window_start)
        query_end = min(rend + timedelta(days=2), query_window_end)

        base_url = "https://data.fingrid.fi/api/data"
        headers = {"x-api-key": self.fingrid_api_key, "accept": "application/json"}
        params_common = {
            "format": "json",
            "locale": "en",
            "page": 1,
            "pageSize": 20000,
            "sortBy": "startTime",
            "sortOrder": "asc",
            "oneRowPerTimePeriod": "true",
            "startTime": query_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "endTime": query_end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            frames: list[pd.DataFrame] = []
            for column_name, dataset_id in self.FINGRID_DATASETS.items():
                params = dict(params_common)
                params["datasets"] = dataset_id
                try:
                    payload = await self._fetch_fingrid_payload(session, base_url, params, dataset_id)
                except Exception as exc:
                    log.warning(
                        "%s: Fingrid dataset %s (%s) unavailable: %s",
                        self.region.bidding_zone_entsoe,
                        dataset_id,
                        column_name,
                        exc,
                    )
                    continue

                frame = self._normalize_fingrid_payload(payload, column_name)
                if not frame.empty:
                    log.info(
                        "%s: Fingrid dataset %s (%s) returned %d rows",
                        self.region.bidding_zone_entsoe,
                        dataset_id,
                        column_name,
                        len(frame),
                    )
                    frames.append(frame)
                await asyncio.sleep(self.FINGRID_MIN_SECONDS_BETWEEN_REQUESTS)

            if not frames:
                return pd.DataFrame()

            combined = pd.concat(frames, axis=1).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            self.last_fingrid_refresh = now
            return combined.loc[pd.Timestamp(rstart):pd.Timestamp(rend)]

    async def _fetch_fingrid_payload(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        params: dict[str, Any],
        dataset_id: int,
    ) -> Any:
        retries = 3
        for attempt in range(1, retries + 1):
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                text = await response.text()
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after is not None else 2.0
                    wait_seconds = max(wait_seconds, self.FINGRID_MIN_SECONDS_BETWEEN_REQUESTS)
                    log.warning(
                        "%s: Fingrid dataset %s rate-limited on attempt %d/%d, retrying in %.1fs",
                        self.region.bidding_zone_entsoe,
                        dataset_id,
                        attempt,
                        retries,
                        wait_seconds,
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {text[:300]}")
                return await response.json()
        raise RuntimeError("repeated Fingrid rate limits")

    def _log_source_configuration(self):
        if self._logged_fingrid_status:
            return
        if self.fingrid_api_key is None and self.region.bidding_zone_entsoe == "FI":
            log.info("%s: Fingrid API key not configured, using ENTSO-E-only market features", self.region.bidding_zone_entsoe)
        elif self.fingrid_api_key is not None and self.region.bidding_zone_entsoe == "FI":
            log.info("%s: Fingrid API key configured, enriching FI market features with Fingrid datasets", self.region.bidding_zone_entsoe)
        self._logged_fingrid_status = True

    def _normalize_entsoe_frame(self, data: Any, prefix: str) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()

        if isinstance(data, pd.Series):
            frame = data.to_frame(name=data.name or prefix)
        else:
            frame = pd.DataFrame(data).copy()

        if frame.empty:
            return frame

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        elif frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")

        frame = frame.sort_index().resample("15min").ffill().bfill()
        frame.columns = [self._normalize_column_name(prefix, column) for column in frame.columns]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame.index.name = "time"
        return frame.dropna(how="all")

    def _normalize_fingrid_payload(self, payload: Any, column_name: str) -> pd.DataFrame:
        items = payload
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("result") or payload.get("items") or []

        rows: list[tuple[pd.Timestamp, float]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            raw_time = item.get("startTime") or item.get("start_time") or item.get("time")
            raw_value = self._extract_fingrid_value(item)
            if raw_time is None or raw_value is None:
                continue
            try:
                timestamp = pd.Timestamp(raw_time)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                else:
                    timestamp = timestamp.tz_convert("UTC")
                rows.append((timestamp, float(raw_value)))
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=["time", column_name]).drop_duplicates(subset=["time"], keep="last")
        frame.set_index("time", inplace=True)
        frame = frame.sort_index().resample("15min").ffill().bfill()
        frame.index.name = "time"
        return frame

    def _extract_fingrid_value(self, item: dict[str, Any]) -> float | None:
        for key in ("value", "valueFloat", "valueNumber"):
            raw_value = item.get(key)
            if raw_value is None or isinstance(raw_value, bool):
                continue
            numeric_value = pd.to_numeric(raw_value, errors="coerce")
            if pd.notna(numeric_value):
                return float(numeric_value)

        candidates: list[float] = []
        for key, raw_value in item.items():
            normalized_key = key.replace("_", "").lower()
            if normalized_key in self.FINGRID_METADATA_KEYS or normalized_key.endswith("id"):
                continue
            if raw_value is None or isinstance(raw_value, bool):
                continue
            numeric_value = pd.to_numeric(raw_value, errors="coerce")
            if pd.notna(numeric_value):
                candidates.append(float(numeric_value))

        if len(candidates) == 1:
            return candidates[0]
        return None

    def _normalize_column_name(self, prefix: str, column: Any) -> str:
        if isinstance(column, tuple):
            raw = "_".join(str(part) for part in column if part not in (None, "", "nan"))
        else:
            raw = str(column)
        normalized = raw.strip().lower()
        normalized = normalized.replace(" ", "_").replace("-", "_").replace("/", "_")
        normalized = normalized.replace("(", "").replace(")", "").replace(".", "")
        normalized = normalized.replace("__", "_")
        if normalized == prefix:
            return prefix
        return f"{prefix}_{normalized}".strip("_")
