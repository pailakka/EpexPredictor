import asyncio
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import cast, override

import pandas as pd
from astral import Observer, sun

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

class AuxDataStore(DataStore):
    """
    Used as in-memory store for computed data (holidays, slot number, day of week etc)
    Not stored to disk, just used as a cache to make retraining faster
    """

    data : pd.DataFrame
    region : PriceRegion

    update_lock: asyncio.Lock


    def __init__(self, region : PriceRegion, storage_dir: str | None = None):
        super().__init__(region)
        self.update_lock = asyncio.Lock()

    @override
    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        updated = False

        async with self.update_lock:
            for rstart, rend in self.gen_missing_date_ranges(start, end):
                df = await asyncio.to_thread(self._compute_data, rstart, rend)
                if len(df) > 0:
                    self._update_data(df)
                    updated = True

            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: aux data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()
            return updated
    
    @override
    def get_next_horizon_revalidation_time(self) -> datetime|None:
        return None

    def _compute_data(self, rstart: datetime, rend: datetime) -> pd.DataFrame:
        """
        Careful: will be called in separate thread
        """
        tzlocal = self.region.get_timezone_info()

        # make it full day to be sure
        rstart = rstart.replace(hour=0, minute=0, second=0, microsecond=0)
        rend = rend.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        log.info(f"{self.region.bidding_zone_entsoe}: computing aux data from {rstart.isoformat()} to {rend.isoformat()}")

        df = pd.DataFrame(data={"time": [pd.to_datetime(rstart, utc=True), pd.to_datetime(rend, utc=True)]})
        df.set_index("time", inplace=True)
        df = cast(pd.DataFrame, df.resample('15min').ffill())
        df.reset_index(inplace=True)

        df["holiday"] = df["time"].apply(lambda t: self.is_holiday(t.astimezone(tzlocal)))
        # Add specific integer temporal features (these will be treated as categoricals by LightGBM)
        local_times = df["time"].dt.tz_convert(tzlocal)
        df["weekday"] = local_times.dt.weekday
        df["month"] = local_times.dt.month
        df["day_of_year"] = local_times.dt.dayofyear
        df["hour_of_day"] = local_times.dt.hour + (local_times.dt.minute / 60.0) # E.g. 14:15 becomes 14.25
        
        observer = Observer(latitude=statistics.mean(self.region.latitudes), longitude=statistics.mean(self.region.longitudes))

        df["sunelevation"] = df["time"].apply(lambda t: sun.elevation(observer, t))
        df["azimuth"] = df["time"].apply(lambda t: sun.azimuth(observer, t))
        df["sr_influence"] = df["time"].apply(lambda t: (t - sun.sunrise(observer, date=t)).total_seconds())
        df["ss_influence"] = df["time"].apply(lambda t: (t - sun.sunset(observer, date=t)).total_seconds())

        def _dist_to_local_hour_sec(t: pd.Timestamp, hour: int) -> float:
            local_t = t.astimezone(tzlocal)
            local_peak = local_t.replace(hour=hour, minute=0, second=0, microsecond=0)
            return (local_t - local_peak).total_seconds()

        df["morningpeak"] = df["time"].apply(lambda t: _dist_to_local_hour_sec(t, 8))
        df["eveningpeak"]  = df["time"].apply(lambda t: _dist_to_local_hour_sec(t, 19))

        # hour_of_week: a continuous within-week demand cycle feature for LEAR.
        # Mon 00:00=0 ... Sun 23:45=167.75. Lets LEAR learn that Mon morning is
        # structurally different from Sat night without needing separate dummies.
        df["hour_of_week"] = df["weekday"] * 24.0 + local_times.dt.hour + local_times.dt.minute / 60.0

        df.set_index("time", inplace=True)
        return df

    def is_holiday(self, t : pd.Timestamp) -> float:
        if t.weekday() == 6:
            return 1

        date = t.date()

        cnt_holiday = sum(bool(date in h)
                      for h in self.region.holidays)
        return cnt_holiday / len(self.region.holidays)
