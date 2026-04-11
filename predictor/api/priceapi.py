import asyncio
from io import BytesIO
import logging
import os
from pathlib import Path
from matplotlib.figure import Figure
import pandas as pd
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Self
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from predictor.model.dataframesnapshotstore import DataFrameSnapshotStore
from predictor.model.forecastartifactstore import ForecastArtifactStore
from predictor.model.forecastexplainer import ForecastExplainer
from predictor.model.priceregion import PriceRegion, PriceRegionName
from predictor.model.predictionsnapshotstore import PredictionSnapshotStore
import predictor.model.pricepredictor as pp


import warnings

# Used internally inside the standard library by asyncio.to_thread.. annoying
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="asyncio"
)



app = FastAPI(title="EPEX day-ahead prediction API", description="""
API can be used free of charge on a fair use premise.
There are no guarantees on availability or correctnes of the data.
This is an open source project, feel free to host it yourself. [Source code and docs](https://github.com/b3nn0/EpexPredictor)

### Attribution
Electricity prices provided under CC-BY-4.0 by [energy-charts.info](https://api.energy-charts.info/) and [ENTSO-E](https://www.entsoe.eu/)

[Weather data by Open-Meteo.com](https://open-meteo.com/)
""")

UI_DIR = Path(__file__).with_name("ui")


##### Logging Setup

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now(timezone.utc)

    response = await call_next(request)

    process_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000.0

    client = request.client.host if request.client else "-"
    user_agent = request.headers.get("user-agent", "-")

    log.info(
        '%s "%s %s" %d %.2fms "%s"',
        client,
        request.method,
        request.url,
        response.status_code,
        process_time,
        user_agent,
    )

    return response



logging.getLogger("uvicorn.error").handlers.clear()
logging.getLogger("uvicorn.error").handlers.extend(logging.getLogger().handlers)
logging.getLogger("uvicorn.access").disabled = True # we handle this ourself in middleware above



@app.get("/",  include_in_schema=False)
def api_docs():
    return RedirectResponse("/docs")


USE_PERSISTENT_TESTDATA = os.getenv("USE_PERSISTENT_TEST_DATA", "false").lower() in ("yes", "true", "t", "1")
EPEXPREDICTOR_DATADIR = os.getenv("EPEXPREDICTOR_DATADIR")
TRAINING_DAYS = 120
DEFAULT_TIMEZONE = "Europe/Berlin"
ENABLE_REQUEST_TRAINING = os.getenv("EPEXPREDICTOR_ENABLE_REQUEST_TRAINING", "true").lower() in ("yes", "true", "t", "1")


class PriceUnit(str, Enum):
    CT_PER_KWH = "CT_PER_KWH" #1.0
    EUR_PER_KWH = "EUR_PER_KWH"# 1 / 100.0
    EUR_PER_MWH = "EUR_PER_MWH"# 1 / 100.0 * 1000

    def convert(self, ct_per_kwh) -> float:
        if self.value == self.EUR_PER_KWH:
            return ct_per_kwh / 100.0
        elif self.value == self.EUR_PER_MWH:
            return ct_per_kwh / 100.0 * 1000
        return ct_per_kwh

class OutputFormat(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PriceModel(BaseModel):
    """Price at a specific time. Output-only model, uses camelCase for API compatibility."""

    starts_at: datetime = Field(serialization_alias="startsAt")
    total: float

class PricesModelShort(BaseModel):
    s: list[int]
    t: list[float]

class PricesModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prices: list[PriceModel]
    known_until: datetime = Field(serialization_alias="knownUntil")


class SnapshotSelectionStrategy(str, Enum):
    LATEST = "latest"
    EARLIEST = "earliest"


class SnapshotScenarioRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region: PriceRegionName = PriceRegionName.FI
    generated_at_utc: datetime = Field(alias="generatedAtUtc")
    target_time_utc: datetime = Field(alias="targetTimeUtc")
    load_forecast_pct: float = Field(0.0, alias="loadForecastPct")
    wind_forecast_pct: float = Field(0.0, alias="windForecastPct")
    solar_forecast_pct: float = Field(0.0, alias="solarForecastPct")
    import_headroom_mw: float = Field(0.0, alias="importHeadroomMw")
    cross_border_flow_mw: float = Field(0.0, alias="crossBorderFlowMw")
    coupled_market_spread_delta: float = Field(0.0, alias="coupledMarketSpreadDelta")

    def scenario_inputs(self) -> dict[str, float]:
        return {
            "load_forecast_pct": self.load_forecast_pct,
            "wind_forecast_pct": self.wind_forecast_pct,
            "solar_forecast_pct": self.solar_forecast_pct,
            "import_headroom_mw": self.import_headroom_mw,
            "cross_border_flow_mw": self.cross_border_flow_mw,
            "coupled_market_spread_delta": self.coupled_market_spread_delta,
        }


    
class RegionPriceManager:
    predictor : pp.PricePredictor
    prediction_snapshots: PredictionSnapshotStore
    forecast_artifacts: ForecastArtifactStore
    predictor_snapshots: DataFrameSnapshotStore
    distribution_snapshots: DataFrameSnapshotStore
    explainer: ForecastExplainer

    last_retrain : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_weather_update : datetime = datetime(1980, 1, 1, tzinfo=timezone.utc)
    last_known_price : datetime
    last_generated_forecast: datetime


    cachedprices : pd.DataFrame
    cachedeval : pd.DataFrame
    cachedquantiles: pd.DataFrame
    latest_metadata: dict[str, Any]

    update_lock: asyncio.Lock

    init_lock: asyncio.Lock
    is_loaded: bool = False

    def __init__(self, region: PriceRegion):
        self.init_lock = asyncio.Lock() # ensures only one aio worker will load persistent data on first access
        self.update_lock = asyncio.Lock() # ensures only one aio worker will trigger model update

        self.cachedprices = pd.DataFrame()
        self.cachedeval = pd.DataFrame()
        self.cachedquantiles = pd.DataFrame()
        self.latest_metadata = {}
        self.last_known_price = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.last_generated_forecast = datetime(1970, 1, 1, tzinfo=timezone.utc)

        self.predictor = pp.PricePredictor(region, storage_dir=EPEXPREDICTOR_DATADIR)
        self.prediction_snapshots = PredictionSnapshotStore(region, EPEXPREDICTOR_DATADIR)
        self.forecast_artifacts = ForecastArtifactStore(region, EPEXPREDICTOR_DATADIR)
        self.explainer = ForecastExplainer(region, self.forecast_artifacts)
        self.predictor_snapshots = DataFrameSnapshotStore(
            region,
            EPEXPREDICTOR_DATADIR,
            "predictor_snapshots_v1",
            extra_columns=["model_version", "train_start_utc", "train_end_utc"],
        )
        self.distribution_snapshots = DataFrameSnapshotStore(
            region,
            EPEXPREDICTOR_DATADIR,
            "forecast_distributions_v1",
            extra_columns=["model_version", "train_start_utc", "train_end_utc"],
        )

    async def ensure_loaded(self) -> Self:
        async with self.init_lock:
            if self.is_loaded:
                return self
            log.info(f"{self.predictor.region.bidding_zone_entsoe}: Loading persistent data")
            await self.predictor.load_from_persistence()
            self.load_cached_artifacts()
            self.is_loaded = True
        return self

    def load_cached_artifacts(self):
        cachedprices, cachedeval, cachedquantiles, metadata = self.forecast_artifacts.load_latest()
        self.latest_metadata = metadata
        if not cachedprices.empty:
            self.cachedprices = cachedprices
        if not cachedeval.empty:
            self.cachedeval = cachedeval
        if not cachedquantiles.empty:
            self.cachedquantiles = cachedquantiles

        generated_at = metadata.get("generated_at_utc")
        if generated_at:
            self.last_generated_forecast = pd.Timestamp(generated_at).to_pydatetime()
            self.last_retrain = self.last_generated_forecast

        known_until = metadata.get("known_until_utc")
        if known_until:
            self.last_known_price = pd.Timestamp(known_until).to_pydatetime()

    def _ensure_fi_explainability(self):
        if self.predictor.region.bidding_zone_entsoe != "FI":
            raise HTTPException(status_code=400, detail="Snapshot explainability is currently available only for FI")

    def _to_utc_timestamp(self, value: datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _slice_snapshot_frame(
        self,
        frame: pd.DataFrame,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        sliced = frame.copy()
        sliced["target_time_utc"] = pd.to_datetime(sliced["target_time_utc"], utc=True)
        if start_ts is not None:
            sliced = sliced[sliced["target_time_utc"] >= self._to_utc_timestamp(start_ts)]
        if end_ts is not None:
            sliced = sliced[sliced["target_time_utc"] <= self._to_utc_timestamp(end_ts)]
        return sliced

    def _load_snapshot_frame(
        self,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> pd.DataFrame:
        predictions = self._slice_snapshot_frame(self.prediction_snapshots.load(), start_ts, end_ts)
        features = self._slice_snapshot_frame(self.predictor_snapshots.load(), start_ts, end_ts)
        return self.explainer.merge_snapshot_frames(
            predictions,
            features,
            self.predictor.pricestore.data,
        )

    def list_snapshot_runs(
        self,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> dict[str, Any]:
        self._ensure_fi_explainability()
        predictions = self._slice_snapshot_frame(self.prediction_snapshots.load(), start_ts, end_ts)
        features = self._slice_snapshot_frame(self.predictor_snapshots.load(), start_ts, end_ts)
        return {
            "region": self.predictor.region.bidding_zone_entsoe,
            "runs": self.explainer.build_snapshot_catalog(predictions, features, start_ts, end_ts),
        }

    def get_snapshot_summary(
        self,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        selection: SnapshotSelectionStrategy = SnapshotSelectionStrategy.LATEST,
        generated_at: datetime | None = None,
    ) -> dict[str, Any]:
        self._ensure_fi_explainability()
        merged = self._load_snapshot_frame(start_ts, end_ts)
        selected = self.explainer.select_rows(merged, selection=selection.value, generated_at=generated_at)
        summary = self.explainer.summarize_rows(selected)
        rows = []
        for _, row in selected.iterrows():
            rows.append(
                {
                    "time_utc": pd.Timestamp(row["target_time_utc"]).isoformat(),
                    "generated_at_utc": pd.Timestamp(row["generated_at_utc"]).isoformat(),
                    "predicted_price": None if pd.isna(row.get("predicted_price")) else float(row["predicted_price"]),
                    "actual_price": None if pd.isna(row.get("actual_price")) else float(row["actual_price"]),
                    "abs_error": None
                    if pd.isna(row.get("actual_price")) or pd.isna(row.get("predicted_price"))
                    else float(abs(float(row["predicted_price"]) - float(row["actual_price"]))),
                    "model_version": None if pd.isna(row.get("model_version")) else str(row["model_version"]),
                    "explainable": bool(row.get("explainable")),
                    "explainable_reason": row.get("explainable_reason"),
                }
            )

        runs = self.list_snapshot_runs(start_ts, end_ts)["runs"]
        return {
            "region": self.predictor.region.bidding_zone_entsoe,
            "selection_strategy": selection.value,
            "selected_generated_at_utc": None if generated_at is None else self._to_utc_timestamp(generated_at).isoformat(),
            "rows": rows,
            "runs": runs,
            **summary,
        }

    def _find_snapshot_row(self, generated_at: datetime, target_time: datetime) -> pd.Series:
        merged = self._load_snapshot_frame(target_time, target_time)
        generated_at_ts = self._to_utc_timestamp(generated_at)
        target_time_ts = self._to_utc_timestamp(target_time)
        match = merged[
            (merged["generated_at_utc"] == generated_at_ts)
            & (merged["target_time_utc"] == target_time_ts)
        ]
        if match.empty:
            raise HTTPException(status_code=404, detail="No snapshot row found for the selected generated-at and target time")
        return match.iloc[0]

    def get_snapshot_explanation(self, generated_at: datetime, target_time: datetime) -> dict[str, Any]:
        self._ensure_fi_explainability()
        row = self._find_snapshot_row(generated_at, target_time)
        try:
            return self.explainer.explain_row(row)
        except FileNotFoundError as exc:
            return {
                "explainable": False,
                "reason": str(exc),
                "generated_at_utc": self._to_utc_timestamp(generated_at).isoformat(),
                "target_time_utc": self._to_utc_timestamp(target_time).isoformat(),
            }

    def evaluate_snapshot_scenario(self, request: SnapshotScenarioRequest) -> dict[str, Any]:
        self._ensure_fi_explainability()
        row = self._find_snapshot_row(request.generated_at_utc, request.target_time_utc)
        try:
            return self.explainer.evaluate_scenario(row, request.scenario_inputs())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    def prune_model_bundles(self) -> list[str]:
        referenced_versions: set[str] = set()
        for frame in (
            self.prediction_snapshots.load(),
            self.predictor_snapshots.load(),
            self.distribution_snapshots.load(),
        ):
            if "model_version" not in frame.columns:
                continue
            referenced_versions.update(str(value) for value in frame["model_version"].dropna().unique())
        if self.predictor.model_version is not None:
            referenced_versions.add(self.predictor.model_version)
        return self.forecast_artifacts.prune_model_bundles(referenced_versions)

    def _ui_source_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "weather": self.predictor.weatherstore.data,
            "market": self.predictor.marketstore.data,
            "entsoe": self.predictor.entsoestore.data,
            "gas": self.predictor.gasstore.data,
        }

    def _ui_humanize_column(self, column: str) -> str:
        label = column.replace("_", " ").strip()
        if not label:
            return column
        return label[0].upper() + label[1:]

    def _ui_source_groups(self) -> list[dict[str, object]]:
        labels = {
            "weather": "Weather",
            "market": "Market",
            "entsoe": "ENTSO-E",
            "gas": "Gas",
        }
        groups: list[dict[str, object]] = []
        for group_name, frame in self._ui_source_frames().items():
            if frame.empty:
                continue
            columns = sorted(str(column) for column in frame.columns)
            groups.append(
                {
                    "id": group_name,
                    "label": labels.get(group_name, group_name.title()),
                    "columns": [
                        {
                            "name": column,
                            "label": self._ui_humanize_column(column),
                        }
                        for column in columns
                    ],
                }
            )
        return groups

    def _ui_default_source_columns(self, available_columns: set[str]) -> list[str]:
        preferred = [
            "temp_0",
            "wind_0",
            "irradiance_0",
            "fingrid_load_forecast",
            "fingrid_wind_forecast",
            "fingrid_solar_forecast",
            "entsoe_load_forecast_forecasted_load",
            "load",
            "shadow_price_se_3",
        ]
        selected = [column for column in preferred if column in available_columns]
        if selected:
            return selected[:6]
        return sorted(available_columns)[:6]

    def _ui_time_bounds(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        indices: list[pd.DatetimeIndex] = []
        for frame in [self.cachedeval, self.cachedprices, self.predictor.pricestore.data, *self._ui_source_frames().values()]:
            if frame is None or frame.empty:
                continue
            index = frame.index
            if isinstance(index, pd.DatetimeIndex):
                indices.append(index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC"))
        if not indices:
            return (None, None)
        starts = [index.min() for index in indices]
        ends = [index.max() for index in indices]
        return (min(starts), max(ends))

    def _ui_default_range(self, min_ts: pd.Timestamp | None, max_ts: pd.Timestamp | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if min_ts is None or max_ts is None:
            return (None, None)
        known_until = pd.Timestamp(self.last_known_price)
        if known_until.year > 1971:
            known_until = known_until.tz_localize("UTC") if known_until.tzinfo is None else known_until.tz_convert("UTC")
            default_end = min(max_ts, known_until + timedelta(days=1))
        else:
            default_end = max_ts
        default_start = max(min_ts, default_end - timedelta(days=2))
        return (default_start, default_end)

    def _slice_frame(self, frame: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        sliced = frame.copy()
        if not isinstance(sliced.index, pd.DatetimeIndex):
            sliced.index = pd.to_datetime(sliced.index, utc=True)
        elif sliced.index.tz is None:
            sliced.index = sliced.index.tz_localize("UTC")
        else:
            sliced.index = sliced.index.tz_convert("UTC")
        return sliced.sort_index().loc[start_ts:end_ts]

    def _frame_to_rows(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if frame.empty:
            return rows
        for index, row in frame.iterrows():
            assert isinstance(index, pd.Timestamp)
            item: dict[str, object] = {"time_utc": index.isoformat()}
            for column, value in row.items():
                item[str(column)] = None if pd.isna(value) else float(value)
            rows.append(item)
        return rows

    def get_ui_data(
        self,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        source_columns: list[str] | None = None,
    ) -> dict[str, object]:
        min_ts, max_ts = self._ui_time_bounds()
        default_start, default_end = self._ui_default_range(min_ts, max_ts)
        if default_start is None or default_end is None:
            raise HTTPException(status_code=503, detail="No cached forecast data available yet")

        start = pd.Timestamp(start_ts) if start_ts is not None else default_start
        end = pd.Timestamp(end_ts) if end_ts is not None else default_end
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end < start:
            raise HTTPException(status_code=400, detail="endTs must be after startTs")

        source_groups = self._ui_source_groups()
        available_columns = {
            column["name"]
            for group in source_groups
            for column in group["columns"]  # type: ignore[index]
        }
        selected_columns = [column for column in (source_columns or self._ui_default_source_columns(available_columns)) if column in available_columns]

        price_frame = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="15min", tz="UTC"))
        if not self.cachedeval.empty:
            predicted = self._slice_frame(self.cachedeval[["price"]], start, end).rename(columns={"price": "predicted_price"})
            price_frame = price_frame.join(predicted, how="left")
        if not self.cachedprices.empty:
            served = self._slice_frame(self.cachedprices[["price"]], start, end).rename(columns={"price": "served_price"})
            price_frame = price_frame.join(served, how="left")
        if not self.predictor.pricestore.data.empty:
            actual = self._slice_frame(self.predictor.pricestore.data[["price"]], start, end).rename(columns={"price": "actual_price"})
            price_frame = price_frame.join(actual, how="left")

        source_frame = pd.DataFrame(index=price_frame.index)
        for frame in self._ui_source_frames().values():
            if frame.empty:
                continue
            columns = [column for column in selected_columns if column in frame.columns]
            if not columns:
                continue
            source_frame = source_frame.join(self._slice_frame(frame[columns], start, end), how="outer")
        source_frame = source_frame.sort_index()

        table_frame = price_frame.join(source_frame, how="outer").sort_index()
        actual_overlap = 0
        if {"predicted_price", "actual_price"} <= set(table_frame.columns):
            actual_overlap = int(table_frame[["predicted_price", "actual_price"]].dropna().shape[0])

        return {
            "region": self.predictor.region.bidding_zone_entsoe,
            "timezone": self.predictor.region.timezone,
            "generated_at_utc": self.last_generated_forecast.isoformat(),
            "known_until_utc": self.last_known_price.isoformat(),
            "range": {
                "min_utc": min_ts.isoformat(),
                "max_utc": max_ts.isoformat(),
                "default_start_utc": default_start.isoformat(),
                "default_end_utc": default_end.isoformat(),
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
            },
            "summary": {
                "price_rows": int(len(price_frame)),
                "source_rows": int(len(source_frame.dropna(how="all"))),
                "table_rows": int(len(table_frame.dropna(how="all"))),
                "actual_overlap_rows": actual_overlap,
            },
            "source_groups": source_groups,
            "selected_source_columns": selected_columns,
            "price_rows": self._frame_to_rows(price_frame),
            "source_rows": self._frame_to_rows(source_frame.dropna(how="all")),
            "table_rows": self._frame_to_rows(table_frame.dropna(how="all")),
        }


    def _normalize_start_ts(self, start_ts: datetime | None, tz: ZoneInfo, hourly: bool) -> datetime:
        """Normalize start_ts to the target timezone."""
        if start_ts is None:
            now = datetime.now(tz=tz)
            if hourly:
                return now.replace(second=0, microsecond=0, minute=0, hour=now.hour)
            else:
                return now.replace(second=0, microsecond=0, minute=(now.minute // 15 * 15), hour=now.hour)
        if start_ts.tzinfo is None:
            return start_ts.replace(tzinfo=tz)
        return start_ts.astimezone(tz)


    async def prices(self, hours: int = -1, surcharge: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG) -> PricesModel | PricesModelShort:

        await self.update_in_background()

        try:
            tz = ZoneInfo(timezone)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid timezone {timezone}")
        start_ts = self._normalize_start_ts(start_ts, tz, hourly)
        end_ts = start_ts + timedelta(hours=hours) if hours >= 0 else datetime(2999, 1, 1, tzinfo=tz)

        prediction = self.cachedeval if evaluation else self.cachedprices
        if hourly:
            prediction = prediction.resample("1h").mean()
        
        prediction = prediction.loc[start_ts:end_ts]

        prices = []

        for dt, price in zip(prediction.index, prediction["price"]): # seems to be much faster than .iterrows()..
            assert isinstance(dt, pd.Timestamp)
            total = (price + surcharge) * (1 + tax_percent / 100.0)
            total = unit.convert(total)
            prices.append(PriceModel(starts_at=dt.to_pydatetime().astimezone(tz), total=round(total, 4)))

        if format == OutputFormat.SHORT:
            return self.format_short(prices)
        return PricesModel(prices=prices, known_until=self.last_known_price.astimezone(tz))

        
    def format_short(self, prices: List[PriceModel]) -> PricesModelShort:
        return PricesModelShort(
            s=[round(p.starts_at.timestamp()) for p in prices],
            t=[round(p.total, 4) for p in prices]
        )


    async def update_in_background(self):
        if self.update_lock.locked() and len(self.cachedprices) > 0:
            return # don't queue up multiple updates if we already have a filled cache

        update_future = self.update_data_if_needed()
        if len(self.cachedprices) == 0: # first call, no prices yet -> wait until first update is done
            await update_future
        elif ENABLE_REQUEST_TRAINING:
            asyncio.create_task(update_future)
        else:
            return


    async def update_data_if_needed(self, force: bool = False):
        async with self.update_lock:
            currts = datetime.now(timezone.utc)
            train_start = currts - timedelta(days=TRAINING_DAYS)
            train_end = datetime.now(timezone.utc) + timedelta(days=7) # will ensure all weather data is fetched immediately, not partially for training and then partially for prediction

            weather_age = (currts - self.last_weather_update).total_seconds()
            if (
                not force
                and
                not self._forecast_is_stale(currts)
                and self.predictor.last_data_update() <= self.last_retrain
                and weather_age <= 60 * 60 * 3
                and not self.predictor.pricestore.needs_horizon_revalidation()
                and not self.predictor.gasstore.needs_horizon_revalidation()
            ):
                return


            retrain = False

            # since we cache the prediction result, the price store is never queried and never updates until next retrain/weather update..
            # Ensure we retrain (and re-fetch horizon) more often if needed
            if self.predictor.pricestore.needs_horizon_revalidation() or self.predictor.gasstore.needs_horizon_revalidation():
                await self.predictor.pricestore.get_data(currts, train_end)
                await self.predictor.gasstore.get_data(currts, train_end)

            if weather_age > 60 * 60 * 3:  # update forecasted input data every 3 hours
                start = datetime.now(timezone.utc) - timedelta(days=1)
                end = datetime.now(timezone.utc) + timedelta(days=8)
                await self.predictor.refresh_forecasts(start, end)
                self.last_weather_update = currts
                retrain = True


            if force or self.predictor.last_data_update() > self.last_retrain or retrain:
                log.info(f"{self.predictor.region.bidding_zone_entsoe}: data has been updated - triggering model retrain")
                self.last_retrain = datetime.now(timezone.utc)

                await self.predictor.train(train_start, train_end)
                details = await self.predictor.predict_with_details(
                    train_start,
                    train_end,
                    fill_known=False,
                    generated_at=currts,
                )
                neweval = details["point"]
                known_prices = await self.predictor.pricestore.get_data(train_start, train_end)
                newprices = neweval.copy()
                newprices.update(known_prices)
                self.cachedprices = newprices
                self.cachedeval = neweval
                self.cachedquantiles = details["quantiles"]
                lastknown = self.predictor.pricestore.get_last_known()
                if lastknown is not None:
                    self.last_known_price = lastknown
                self.last_generated_forecast = currts
                stored_rows = await asyncio.to_thread(
                    self.prediction_snapshots.append_predictions,
                    newprices,
                    currts,
                    lastknown,
                    self.predictor.model_version,
                    train_start,
                    train_end,
                )
                if stored_rows > 0:
                    log.info(
                        "%s: stored %d prediction snapshots",
                        self.predictor.region.bidding_zone_entsoe,
                        stored_rows,
                    )

                predictor_snapshot_rows = await asyncio.to_thread(
                    self.predictor_snapshots.append_frame,
                    details["features"],
                    currts,
                    lastknown,
                    {
                        "model_version": self.predictor.model_version,
                        "train_start_utc": train_start,
                        "train_end_utc": train_end,
                    },
                )
                distribution_snapshot_rows = 0
                if not self.cachedquantiles.empty:
                    distribution_snapshot_rows = await asyncio.to_thread(
                        self.distribution_snapshots.append_frame,
                        self.cachedquantiles,
                        currts,
                        lastknown,
                        {
                            "model_version": self.predictor.model_version,
                            "train_start_utc": train_start,
                            "train_end_utc": train_end,
                        },
                    )

                model_files = {}
                if self.predictor.model_version is not None:
                    model_files = self.forecast_artifacts.save_model_bundle(
                        self.predictor.model_version,
                        self.predictor.get_model_artifacts(),
                    )
                self.forecast_artifacts.save_latest(
                    self.cachedprices,
                    self.cachedeval,
                    self.cachedquantiles,
                    {
                        "region": self.predictor.region.bidding_zone_entsoe,
                        "generated_at_utc": currts.isoformat(),
                        "known_until_utc": self.last_known_price.isoformat(),
                        "train_start_utc": train_start.isoformat(),
                        "train_end_utc": train_end.isoformat(),
                        "model_version": self.predictor.model_version,
                        "feature_columns": list(details["features"].columns),
                        "predictor_snapshot_rows": predictor_snapshot_rows,
                        "distribution_snapshot_rows": distribution_snapshot_rows,
                    },
                    model_files=model_files,
                )

                removed_versions = self.prune_model_bundles()
                if removed_versions:
                    log.info(
                        "%s: pruned %d unreferenced model bundles",
                        self.predictor.region.bidding_zone_entsoe,
                        len(removed_versions),
                    )

                self.predictor.cleanup()

    def _forecast_is_stale(self, currts: datetime) -> bool:
        if self.cachedprices.empty:
            return True
        max_age = timedelta(hours=self.predictor.region.worker_interval_hours + 1)
        return currts - self.last_generated_forecast > max_age

 


class Prices:
    region_prices: Dict[PriceRegionName, RegionPriceManager]

    def __init__(self):
        self.region_prices = {}

    async def prices(self, hours: int = -1, surcharge: float = 0.0, tax_percent: float = 0.0, start_ts: datetime | None = None,
                    region: PriceRegionName = PriceRegionName.DE, unit: PriceUnit = PriceUnit.CT_PER_KWH, evaluation: bool = False, hourly: bool = False,
                    timezone: str = DEFAULT_TIMEZONE, format: OutputFormat = OutputFormat.LONG):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region.to_region())
        
        await self.region_prices[region].ensure_loaded()
        return await self.region_prices[region].prices(hours, surcharge, tax_percent, start_ts, unit, evaluation, hourly, timezone, format)
    
    async def get_price_manager(self, region: PriceRegionName):
        if region not in self.region_prices:
            self.region_prices[region] = RegionPriceManager(region.to_region())
        
        await self.region_prices[region].ensure_loaded()
        return self.region_prices[region]


prices_handler = Prices()


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_page():
    return HTMLResponse((UI_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/ui/app.css", include_in_schema=False)
def ui_styles():
    return FileResponse(UI_DIR / "app.css", media_type="text/css")


@app.get("/ui/app.js", include_in_schema=False)
def ui_script():
    return FileResponse(UI_DIR / "app.js", media_type="text/javascript")


@app.get("/ui/data")
async def get_ui_data(
    region: PriceRegionName = Query(PriceRegionName.FI, description="Region/bidding zone"),
    start_ts: datetime | None = Query(None, alias="startTs"),
    end_ts: datetime | None = Query(None, alias="endTs"),
    source_columns: list[str] | None = Query(None, alias="sourceColumns"),
):
    manager = await prices_handler.get_price_manager(region)
    return manager.get_ui_data(start_ts, end_ts, source_columns)


@app.get("/ui/api/snapshots", include_in_schema=False)
async def get_ui_snapshots(
    region: PriceRegionName = Query(PriceRegionName.FI, description="Region/bidding zone"),
    start_ts: datetime | None = Query(None, alias="startTs"),
    end_ts: datetime | None = Query(None, alias="endTs"),
):
    manager = await prices_handler.get_price_manager(region)
    return manager.list_snapshot_runs(start_ts, end_ts)


@app.get("/ui/api/explanation-summary", include_in_schema=False)
async def get_ui_explanation_summary(
    region: PriceRegionName = Query(PriceRegionName.FI, description="Region/bidding zone"),
    start_ts: datetime | None = Query(None, alias="startTs"),
    end_ts: datetime | None = Query(None, alias="endTs"),
    selection: SnapshotSelectionStrategy = Query(SnapshotSelectionStrategy.LATEST),
    generated_at_utc: datetime | None = Query(None, alias="generatedAtUtc"),
):
    manager = await prices_handler.get_price_manager(region)
    return manager.get_snapshot_summary(start_ts, end_ts, selection, generated_at_utc)


@app.get("/ui/api/explanation", include_in_schema=False)
async def get_ui_explanation(
    region: PriceRegionName = Query(PriceRegionName.FI, description="Region/bidding zone"),
    generated_at_utc: datetime = Query(..., alias="generatedAtUtc"),
    target_time_utc: datetime = Query(..., alias="targetTimeUtc"),
):
    manager = await prices_handler.get_price_manager(region)
    return manager.get_snapshot_explanation(generated_at_utc, target_time_utc)


@app.post("/ui/api/scenario", include_in_schema=False)
async def post_ui_scenario(request: SnapshotScenarioRequest):
    manager = await prices_handler.get_price_manager(request.region)
    return manager.evaluate_snapshot_scenario(request)


@app.get("/prices")
async def get_prices(
    hours: int = Query(-1, description="How many hours to predict"),
    surcharge: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}"),

    # Legacy parameters, only here for backwards compatibility
    country: PriceRegionName = Query(None, description="", include_in_schema=False),
    fixed_price: float = Query(None, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice", include_in_schema=False),
    ) -> PricesModel:
    """
    Get price prediction - verbose output format with objects containing full ISO timestamp and price
    """
    if country:
        region = country
    if fixed_price is not None:
        surcharge = fixed_price

    res = await prices_handler.prices(hours, surcharge, tax_percent, start_ts, region, unit, evaluation, hourly, timezone, format=OutputFormat.LONG)
    assert isinstance(res, PricesModel)
    return res


@app.get("/prices_short")
async def get_prices_short(
    hours: int = Query(-1, description="How many hours to predict"),
    surcharge: float = Query(0.0, description="Add this fixed amount to all prices (ct/kWh)"),
    tax_percent: float = Query(0.0, description="Tax % to add to the final price", alias="taxPercent"),
    start_ts: datetime | None = Query(None, description="Start output from this time. At most ~90 days in the past", alias="startTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone", alias="country"),
    evaluation: bool = Query(False, description="Switches to evaluation mode. All values will be generated by the model, instead of only future values. Useful to evaluate model performance."),
    unit: PriceUnit = Query(PriceUnit.CT_PER_KWH, description="Unit of output"),
    hourly: bool = Query(False, description="Output hourly average prices (if your energy provider uses hourly prices)"),
    timezone: str = Query(DEFAULT_TIMEZONE, description=f"Timezone for startTs and output timestamps. Default is {DEFAULT_TIMEZONE}"),
    
    # Legacy parameters, only here for backwards compatibility
    country: PriceRegionName = Query(None, description="", include_in_schema=False),
    fixed_price: float = Query(None, description="Add this fixed amount to all prices (ct/kWh)", alias="fixedPrice", include_in_schema=False),
    ) -> PricesModelShort:
    """
    Get price prediction - short output format with unix timestamp array and price array
    """
    if country:
        region = country
    if fixed_price is not None:
        surcharge = fixed_price

    res = await prices_handler.prices(hours, surcharge, tax_percent, start_ts, region, unit, evaluation, hourly, timezone, format=OutputFormat.SHORT)
    assert isinstance(res, PricesModelShort)
    return res


@app.get("/eval_plot", response_class=Response, response_model=None, responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG plot"
        },
        400: {
            "content": {"application/json": {}}
        }
    })
async def generate_evaluation_plot(
    start_ts: datetime | None = Query(None, description="Plot range start, at most ~1 year in the past. Default today 00:00Z", alias="startTs"),
    end_ts: datetime | None = Query(None, description="Plot range end, Default startTs + 1 week. At most 31 days after startTs and 10 days from now", alias="endTs"),
    region: PriceRegionName = Query(PriceRegionName.DE, description="Region/bidding zone"),
    transparent: bool = Query(False, description="Render with transparent background"),
    width: int = Query(2048, description="image width in pixels", ge=300, le=10000),
    height: int = Query(1024, description="image height in pixels", ge=300, le=10000)):
    """
    Trains a model just for you, training with 120 days before the given time range and providing a forecast for the given range.
    - If there is no cached weather or price data for the given time range, this request can take a while. Be patient.
    - This request is rather CPU intensive. Do not batch-call or you will be banned.
    """
    now = datetime.now(timezone.utc)
    start_ts = start_ts or now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = end_ts or start_ts + timedelta(days=7)
    start_ts = start_ts.astimezone(timezone.utc)
    end_ts = end_ts.astimezone(timezone.utc)
    if (end_ts - start_ts).total_seconds() > 31 * 24 * 60 * 60:
        raise HTTPException(status_code=400, detail="At most 4 weeks can be plotted")
    
    if start_ts < now - timedelta(days=365):
        raise HTTPException(status_code=400, detail="Requested range too far in the past")
    
    if end_ts > now + timedelta(days=10):
        raise HTTPException(status_code=400, detail="Requested range too far in the future")
    
    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="endTs must be after startTs")

    # reuse the same data stores for a unified cache
    pricemanager = await prices_handler.get_price_manager(region)
    await pricemanager.update_data_if_needed()
    orig_predictor = pricemanager.predictor
    
    predictor = pp.PricePredictor(region.to_region())
    predictor.use_datastores_from(orig_predictor)

    learn_start = start_ts - timedelta(days=TRAINING_DAYS)
    learn_end = start_ts
    await predictor.train(learn_start, learn_end)

    predicted = await predictor.predict(start_ts, end_ts, fill_known=False, generated_at=start_ts)
    predicted = predicted.rename(columns={"price": "predicted"})

    actual = await predictor.pricestore.get_data(start_ts, end_ts)
    actual = actual.rename(columns={"price": "actual"})

    merged = pd.concat([predicted, actual])

    img_data = BytesIO()
    plot = merged.plot.line(grid=True)
    assert isinstance(plot.figure, Figure)
    plot.margins(0)
    plot.figure.set_size_inches(width / 100, height / 100)
    plot.figure.savefig(img_data, format="png", transparent=transparent, dpi=100, bbox_inches="tight")
    plt.close(plot.figure)

    img_data.seek(0)

    response = Response(content=img_data.read(), media_type="image/png")
    response.headers.update({
        "Cache-Control": "max-age=60"
    })

    return response
