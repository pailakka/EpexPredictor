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
        "fingrid_wind_power_realtime": 181,
        "fingrid_wind_power_forecast": 245,
        "fingrid_wind_forecast": 246,
        "fingrid_solar_forecast": 247,
        "fingrid_wind_capacity": 268,
        "fingrid_capacity_fi_ee": 115,
        "fingrid_capacity_ee_fi": 112,
        "fingrid_nuclear_production": 188,
        "fingrid_imbalance": 397,
        "fingrid_commercial_flow_fi_ee": 140,
        "fingrid_electric_boiler": 371,
    }
    FINGRID_MIN_SECONDS_BETWEEN_REQUESTS = 6.2
    FINGRID_HISTORY_LOOKBACK_DAYS = 150
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
    FINGRID_HISTORICAL_REQUIRED_COLUMNS = [
        "fingrid_load_forecast",
        "fingrid_wind_power_realtime",
        "fingrid_wind_power_forecast",
        "fingrid_wind_forecast",
        "fingrid_solar_forecast",
        "fingrid_wind_capacity",
        "fingrid_nuclear_production",
    ]

    SHADOW_PRICE_AREAS = ["SE_1", "SE_3", "EE", "NO_4"]
    JAO_IMPORT_COLUMNS = {
        "border_SE1_FI": "jao_import_capacity_se1_fi",
        "border_SE3_FI": "jao_import_capacity_se3_fi",
        "border_EE_FI": "jao_import_capacity_ee_fi",
    }
    JAO_ENDPOINT = "https://publicationtool.jao.eu/nordic/api/data/maxBorderFlow"
    JAO_WINDOW_HOURS = 48

    BALTIC_WIND_LOCATIONS = [
        ("eu_ws_EE01", 58.8960, 22.5605),
        ("eu_ws_EE02", 58.6347, 25.1230),
        ("eu_ws_DK01", 56.4260, 8.1281),
        ("eu_ws_DK02", 56.6013, 11.1047),
        ("eu_ws_DE01", 54.2194, 9.6961),
        ("eu_ws_DE02", 52.6367, 9.8451),
        ("eu_ws_SE01", 65.2536, 21.6020),
        ("eu_ws_SE02", 64.5000, 17.0000),
        ("eu_ws_SE03", 59.7852, 13.0042),
    ]

    SYKE_ODATA_BASE = "https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1/odata"
    SYKE_CARRY_DAYS = 60
    SYKE_HYDRO_METRICS = [
        {
            "name": "HydroPrecip_5d",
            "entity": "SadantaAlue",
            "places": [848, 852, 810, 811, 834, 837, 879, 881, 885, 886],
            "extra_filter": "Jakso_Id eq 2",
        },
        {
            "name": "HydroSWE",
            "entity": "LumiAlue",
            "places": [196, 200, 159, 160, 183, 185, 226, 228, 232, 233],
            "extra_filter": "",
        },
    ]
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
            day_needs_refresh = self._market_day_needs_refresh(current)
            too_long = range_start is not None and (current - range_start).days >= 14

            if range_start is not None and (not day_needs_refresh or too_long):
                result.append((pd.Timestamp(range_start), pd.Timestamp(current)))
                range_start = None

            if range_start is None and day_needs_refresh:
                range_start = current

            current = next_day

        if range_start is not None:
            result.append((pd.Timestamp(range_start), pd.Timestamp(end)))
        return result

    def _market_day_needs_refresh(self, day: datetime) -> bool:
        day_ts = pd.Timestamp(day)
        if day_ts.tzinfo is None:
            day_ts = day_ts.tz_localize("UTC")
        else:
            day_ts = day_ts.tz_convert("UTC")
        if day_ts not in self.data.index:
            return True

        required_columns = self._historical_required_columns()
        if not required_columns:
            return False

        today_utc = pd.Timestamp.now(tz=timezone.utc).floor("D")
        if day_ts >= today_utc:
            return False

        day_end = day_ts + pd.Timedelta(days=1)
        day_frame = self.data.loc[(self.data.index >= day_ts) & (self.data.index < day_end)]
        if day_frame.empty:
            return True

        for column in required_columns:
            if column not in day_frame.columns or day_frame[column].notna().sum() == 0:
                return True
        return False

    def _historical_required_columns(self) -> list[str]:
        if self.region.bidding_zone_entsoe != "FI":
            return []

        columns: list[str] = []
        if self.fingrid_api_key is not None:
            columns.extend(self.FINGRID_HISTORICAL_REQUIRED_COLUMNS)
        columns.extend(self.JAO_IMPORT_COLUMNS.values())
        columns.append("jao_import_capacity_total")
        for metric in self.SYKE_HYDRO_METRICS:
            columns.append(f"{metric['name']}_median")
            columns.append(f"{metric['name']}_p10")
        columns.extend(location[0] for location in self.BALTIC_WIND_LOCATIONS)
        return list(dict.fromkeys(columns))

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

        jao_frame = await self._fetch_jao_range(rstart, rend)
        if not jao_frame.empty:
            frames.append(jao_frame)

        baltic_wind_frame = await self._fetch_baltic_wind_range(rstart, rend)
        if not baltic_wind_frame.empty:
            frames.append(baltic_wind_frame)

        hydrology_frame = await self._fetch_hydrology_range(rstart, rend)
        if not hydrology_frame.empty:
            frames.append(hydrology_frame)

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
        historical_backfill = rend < now - timedelta(days=1)
        if (
            not historical_backfill
            and effective_last_refresh is not None
            and now - effective_last_refresh < self.FINGRID_REFRESH_COOLDOWN
        ):
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

    async def _fetch_jao_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        if self.region.bidding_zone_entsoe != "FI":
            return pd.DataFrame()

        qstart = self._utc_timestamp(rstart)
        qend = self._utc_timestamp(rend)
        frames: list[pd.DataFrame] = []

        async with aiohttp.ClientSession() as session:
            cursor = qstart
            while cursor < qend:
                window_end = min(cursor + pd.Timedelta(hours=self.JAO_WINDOW_HOURS), qend)
                params = {
                    "FromUtc": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "ToUtc": window_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                }
                try:
                    async with session.get(
                        self.JAO_ENDPOINT,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        text = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"HTTP {response.status}: {text[:300]}")
                        payload = await response.json()
                except Exception as exc:
                    log.warning("%s: JAO import capacity unavailable: %s", self.region.bidding_zone_entsoe, exc)
                    cursor = window_end
                    continue

                frame = self._normalize_jao_payload(payload)
                if not frame.empty:
                    frames.append(frame)
                cursor = window_end

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.loc[qstart:qend]

    def _normalize_jao_payload(self, payload: Any) -> pd.DataFrame:
        items = payload
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("result") or payload.get("items") or []

        rows: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            raw_time = item.get("dateTimeUtc") or item.get("startTime") or item.get("time")
            if raw_time is None:
                continue
            row: dict[str, Any] = {"time": self._utc_timestamp(raw_time)}
            for source_column, target_column in self.JAO_IMPORT_COLUMNS.items():
                row[target_column] = pd.to_numeric(item.get(source_column), errors="coerce")
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows).drop_duplicates(subset=["time"], keep="last")
        frame.set_index("time", inplace=True)
        border_columns = list(self.JAO_IMPORT_COLUMNS.values())
        frame[border_columns] = frame[border_columns].apply(pd.to_numeric, errors="coerce")
        total = frame[border_columns].sum(axis=1, min_count=1)
        if total.eq(0.0).any():
            total = total.mask(total.eq(0.0)).ffill().fillna(0.0)
        frame["jao_import_capacity_total"] = total
        frame = frame.sort_index().resample("15min").ffill().bfill()
        frame.index.name = "time"
        return frame.dropna(how="all")

    async def _fetch_baltic_wind_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        if self.region.bidding_zone_entsoe != "FI":
            return pd.DataFrame()

        qstart = self._utc_timestamp(rstart)
        qend = self._utc_timestamp(rend)
        now = pd.Timestamp.now(tz=timezone.utc)
        frames: list[pd.DataFrame] = []
        lats = ",".join(str(location[1]) for location in self.BALTIC_WIND_LOCATIONS)
        lons = ",".join(str(location[2]) for location in self.BALTIC_WIND_LOCATIONS)

        async with aiohttp.ClientSession() as session:
            archive_end = min(qend, now - pd.Timedelta(days=3))
            if qstart <= archive_end:
                params = {
                    "latitude": lats,
                    "longitude": lons,
                    "start_date": qstart.date().isoformat(),
                    "end_date": archive_end.date().isoformat(),
                    "hourly": "wind_speed_100m",
                    "wind_speed_unit": "ms",
                    "timezone": "UTC",
                }
                try:
                    payload = await self._fetch_openmeteo_wind_payload(
                        session,
                        "https://archive-api.open-meteo.com/v1/archive",
                        params,
                    )
                    frame = self._normalize_openmeteo_wind_payload(payload, "wind_speed_100m")
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    log.warning("%s: Baltic historical wind unavailable: %s", self.region.bidding_zone_entsoe, exc)

            if qend >= now - pd.Timedelta(days=5):
                params = {
                    "latitude": lats,
                    "longitude": lons,
                    "hourly": "wind_speed_120m",
                    "wind_speed_unit": "ms",
                    "past_days": 4,
                    "forecast_days": 10,
                    "timezone": "UTC",
                }
                try:
                    payload = await self._fetch_openmeteo_wind_payload(
                        session,
                        "https://api.open-meteo.com/v1/forecast",
                        params,
                    )
                    frame = self._normalize_openmeteo_wind_payload(payload, "wind_speed_120m")
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    log.warning("%s: Baltic forecast wind unavailable: %s", self.region.bidding_zone_entsoe, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.resample("15min").interpolate("time").ffill().bfill()
        combined.index.name = "time"
        return combined.loc[qstart:qend]

    async def _fetch_openmeteo_wind_payload(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=45)) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {text[:300]}")
            return await response.json()

    def _normalize_openmeteo_wind_payload(self, payload: Any, value_key: str) -> pd.DataFrame:
        items = payload if isinstance(payload, list) else [payload]
        frames: list[pd.DataFrame] = []
        for i, item in enumerate(items[: len(self.BALTIC_WIND_LOCATIONS)]):
            if not isinstance(item, dict):
                continue
            hourly = item.get("hourly") or {}
            raw_times = hourly.get("time")
            raw_values = hourly.get(value_key)
            if raw_times is None or raw_values is None:
                continue
            code = self.BALTIC_WIND_LOCATIONS[i][0]
            frame = pd.DataFrame(
                {
                    "time": pd.to_datetime(raw_times, utc=True, errors="coerce"),
                    code: pd.to_numeric(pd.Series(raw_values), errors="coerce"),
                }
            ).dropna(subset=["time"])
            if frame.empty:
                continue
            frame.set_index("time", inplace=True)
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.index.name = "time"
        return combined.dropna(how="all")

    async def _fetch_hydrology_range(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        if self.region.bidding_zone_entsoe != "FI":
            return pd.DataFrame()

        qstart = self._utc_timestamp(rstart)
        qend = self._utc_timestamp(rend)
        if qend < qstart:
            return pd.DataFrame()

        target_index = pd.date_range(qstart.floor("15min"), qend.ceil("15min"), freq="15min", tz=timezone.utc)
        if target_index.empty:
            return pd.DataFrame()

        now = pd.Timestamp.now(tz=timezone.utc)
        last_complete_day = now.floor("D") - pd.Timedelta(days=1)
        fetch_start_day = min(target_index.min().floor("D"), last_complete_day) - pd.Timedelta(days=self.SYKE_CARRY_DAYS)
        fetch_end_day = last_complete_day
        rows_by_metric: dict[str, pd.DataFrame] = {}

        async with aiohttp.ClientSession() as session:
            for metric in self.SYKE_HYDRO_METRICS:
                try:
                    rows_by_metric[metric["name"]] = await self._fetch_syke_metric_rows(
                        session,
                        metric,
                        fetch_start_day.to_pydatetime(),
                        fetch_end_day.to_pydatetime(),
                    )
                except Exception as exc:
                    log.warning(
                        "%s: SYKE hydrology %s unavailable: %s",
                        self.region.bidding_zone_entsoe,
                        metric["name"],
                        exc,
                    )
                    rows_by_metric[metric["name"]] = pd.DataFrame()

        frame = self._compute_hydrology_frame(rows_by_metric, target_index)
        return frame.loc[qstart:qend]

    async def _fetch_syke_metric_rows(
        self,
        session: aiohttp.ClientSession,
        metric: dict[str, Any],
        start: datetime,
        end: datetime,
        chunk_days: int = 31,
        top: int = 500,
    ) -> pd.DataFrame:
        if start > end:
            return pd.DataFrame(columns=["Aika", "Paikka_Id", "Arvo"])

        frames: list[pd.DataFrame] = []
        cursor = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        places = metric.get("places", [])
        place_filter = " or ".join(f"Paikka_Id eq {place_id}" for place_id in places)

        while cursor <= end_ts:
            chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
            query_filter = (
                f"Aika ge datetime'{cursor.strftime('%Y-%m-%dT%H:%M:%S')}' and "
                f"Aika le datetime'{chunk_end.strftime('%Y-%m-%dT%H:%M:%S')}' and "
                f"({place_filter})"
            )
            if metric.get("extra_filter"):
                query_filter = f"{query_filter} and {metric['extra_filter']}"

            skip = 0
            while True:
                params = {
                    "$top": str(top),
                    "$skip": str(skip),
                    "$select": "Paikka_Id,Aika,Arvo",
                    "$filter": query_filter,
                    "$orderby": "Aika asc",
                }
                url = f"{self.SYKE_ODATA_BASE}/{metric['entity']}"
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=45)) as response:
                    text = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status}: {text[:300]}")
                    payload = await response.json()

                rows = payload.get("value", []) if isinstance(payload, dict) else []
                if rows:
                    frames.append(self._normalize_syke_rows(rows))
                if len(rows) < top:
                    break
                skip += top

            cursor = chunk_end + pd.Timedelta(seconds=1)

        if not frames:
            return pd.DataFrame(columns=["Aika", "Paikka_Id", "Arvo"])

        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(subset=["Paikka_Id", "Aika"], keep="last")
        return frame.sort_values(["Aika", "Paikka_Id"]).reset_index(drop=True)

    def _normalize_syke_rows(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["Aika", "Paikka_Id", "Arvo"])
        frame["Aika"] = pd.to_datetime(frame.get("Aika"), utc=True, errors="coerce")
        frame["Paikka_Id"] = pd.to_numeric(frame.get("Paikka_Id"), errors="coerce")
        frame["Arvo"] = pd.to_numeric(frame.get("Arvo"), errors="coerce")
        frame = frame.dropna(subset=["Aika", "Paikka_Id", "Arvo"])
        if frame.empty:
            return pd.DataFrame(columns=["Aika", "Paikka_Id", "Arvo"])
        frame["Paikka_Id"] = frame["Paikka_Id"].astype(int)
        return frame[["Aika", "Paikka_Id", "Arvo"]]

    def _compute_hydrology_frame(
        self,
        rows_by_metric: dict[str, pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        index = self._utc_index(index)
        frame = pd.DataFrame(index=index)
        if index.empty:
            return frame

        day_index = pd.date_range(index.min().floor("D"), index.max().floor("D"), freq="D", tz=timezone.utc)
        target_days = pd.Series(index.floor("D"), index=index)

        for metric in self.SYKE_HYDRO_METRICS:
            name = metric["name"]
            median_col = f"{name}_median"
            p10_col = f"{name}_p10"
            rows = rows_by_metric.get(name, pd.DataFrame())
            if rows.empty:
                frame[median_col] = pd.NA
                frame[p10_col] = pd.NA
                continue

            source = rows.copy()
            source["Aika"] = pd.to_datetime(source["Aika"], utc=True, errors="coerce").dt.floor("D")
            source["Paikka_Id"] = pd.to_numeric(source["Paikka_Id"], errors="coerce")
            source["Arvo"] = pd.to_numeric(source["Arvo"], errors="coerce")
            source = source.dropna(subset=["Aika", "Paikka_Id", "Arvo"])
            if source.empty:
                frame[median_col] = pd.NA
                frame[p10_col] = pd.NA
                continue

            source["Paikka_Id"] = source["Paikka_Id"].astype(int)
            pivot = source.pivot_table(index="Aika", columns="Paikka_Id", values="Arvo", aggfunc="last")
            pivot = pivot.reindex(columns=metric.get("places", []))
            ext_start = min(pivot.index.min(), day_index.min())
            ext_index = pd.date_range(ext_start, day_index.max(), freq="D", tz=timezone.utc)
            pivot = pivot.reindex(ext_index).ffill().reindex(day_index)

            median = pivot.median(axis=1, skipna=True)
            p10 = pivot.quantile(0.10, axis=1, interpolation="linear")
            frame[median_col] = target_days.map(median)
            frame[p10_col] = target_days.map(p10)

        frame.index.name = "time"
        return frame

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

    def _utc_timestamp(self, value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _utc_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        utc_index = pd.DatetimeIndex(index)
        if utc_index.tz is None:
            return utc_index.tz_localize("UTC")
        return utc_index.tz_convert("UTC")

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
