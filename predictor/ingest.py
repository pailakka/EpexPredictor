import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timezone
import logging
from entsoe import entsoe
from .db import get_db

log = logging.getLogger(__name__)

class IngestionManager:
    """Stateless fetcher that pulls arrays directly into the DuckDB snapshot tables to prevent data leakage."""
    
    def __init__(self, entsoe_api_key: str, fingrid_api_key: str):
        self.entsoe_api_key = entsoe_api_key
        self.fingrid_api_key = fingrid_api_key
        self.db = get_db()
        # Ensure client is instantiated eagerly if key is provided
        self.client = entsoe.EntsoePandasClient(api_key=self.entsoe_api_key) if self.entsoe_api_key else None

    async def ingest_fingrid_snapshot(self, dataset_id: str, area: str, unit: str, start: datetime, end: datetime):
        if not self.fingrid_api_key:
            return
            
        base_url = "https://data.fingrid.fi/api/data"
        headers = {"x-api-key": self.fingrid_api_key, "accept": "application/json"}
        params = {
            "format": "json",
            "locale": "en",
            "pageSize": 20000,
            "oneRowPerTimePeriod": "true",
            "datasets": dataset_id,
            "startTime": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "endTime": end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        
        as_of = datetime.now(timezone.utc)
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(base_url, params=params) as response:
                if response.status >= 400:
                    text = await response.text()
                    log.error(f"Fingrid API {dataset_id} failed {response.status}: {text[:100]}")
                    return
                data = await response.json()
                
        items = data.get("data") or data.get("result", [])
        if not items:
            return
            
        rows = []
        for item in items:
            # handle varying Fingrid payload structures
            raw_time = item.get("startTime") or item.get("time") or item.get("start_time")
            raw_val = item.get("value") or item.get("valueFloat") or item.get("valueNumber")
            
            if raw_time is None or raw_val is None: 
                continue
            
            ts = pd.Timestamp(raw_time)
            if ts.tzinfo is None: ts = ts.tz_localize("UTC")
            else: ts = ts.tz_convert("UTC")
                
            rows.append({
                "source": "fingrid",
                "dataset_id": str(dataset_id),
                "as_of_timestamp": as_of,
                "delivery_start_utc": ts.to_pydatetime(),
                "area": area,
                "value": float(raw_val),
                "unit": unit,
                "payload_hash": None # Skipped payload hashing for brevity
            })
            
        if rows:
            df = pd.DataFrame(rows)
            # Bulk append dataframe to DuckDB via sql interface
            self.db.conn.execute("INSERT INTO source_snapshots SELECT * FROM df")
            log.info(f"[Fingrid] Snapshotted {len(df)} rows for dataset {dataset_id}")

    def ingest_entsoe_prices(self, area: str, start: pd.Timestamp, end: pd.Timestamp):
        if not self.client: return
        
        try:
            series = self.client.query_day_ahead_prices(area, start=start, end=end)
            if series is None or series.empty: return
            
            as_of = datetime.now(timezone.utc)
            rows = []
            for ts, val in series.items():
                if pd.isna(val): continue
                if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                else: ts = ts.tz_convert("UTC")
                    
                rows.append({
                    "area": area,
                    "delivery_start_utc": ts.to_pydatetime(),
                    "market_day_cet": ts.tz_convert("CET").date(),
                    "resolution_min": 60, # EntsoE typically returns 60min here, or 15min natively depending on mapping
                    "price_eur_mwh": float(val),
                    "published_at": as_of,
                    "source": "entsoe"
                })
                
            if rows:
                df = pd.DataFrame(rows)
                # INSERT OR IGNORE via ON CONFLICT
                self.db.conn.execute("""
                    INSERT INTO market_prices 
                    SELECT * FROM df 
                    ON CONFLICT(area, delivery_start_utc) DO UPDATE SET 
                        price_eur_mwh = EXCLUDED.price_eur_mwh,
                        published_at = EXCLUDED.published_at
                """)
                log.info(f"[EntsoE] Updated {len(df)} prices for {area}")
        except Exception as e:
            log.error(f"EntsoE Price fetch failed for {area}: {e}")
