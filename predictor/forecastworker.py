#!/usr/bin/python3

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from predictor.api.priceapi import RegionPriceManager
from predictor.model.priceregion import PriceRegionName

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background worker that refreshes cached forecasts.")
    parser.add_argument(
        "--regions",
        default="FI",
        help="Comma separated list of bidding zones to refresh. Defaults to FI.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh cycle and exit.",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=3,
        help="Loop interval in hours when not using --once.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if cached artifacts look fresh.",
    )
    return parser.parse_args()


async def refresh_regions(
    managers: list[tuple[PriceRegionName, RegionPriceManager]] | list[PriceRegionName],
    force: bool,
) -> None:
    for item in managers:
        if isinstance(item, tuple):
            region_name, manager = item
        else:
            region_name = item
            manager = RegionPriceManager(region_name.to_region())
            await manager.ensure_loaded()
        log.info("%s: worker refresh started", region_name.value)
        await manager.update_data_if_needed(force=force)
        log.info("%s: worker refresh finished", region_name.value)


async def main() -> None:
    args = parse_args()
    region_names = [PriceRegionName(name.strip()) for name in args.regions.split(",") if name.strip()]

    # Create managers once so in-memory state (cooldowns, caches) persists across cycles
    managers: list[tuple[PriceRegionName, RegionPriceManager]] = []
    for region_name in region_names:
        manager = RegionPriceManager(region_name.to_region())
        await manager.ensure_loaded()
        managers.append((region_name, manager))

    while True:
        cycle_started = datetime.now(timezone.utc)
        await refresh_regions(managers, force=args.force)
        if args.once:
            return

        next_cycle = cycle_started + timedelta(hours=args.interval_hours)
        sleep_seconds = max(30.0, (next_cycle - datetime.now(timezone.utc)).total_seconds())
        log.info("worker sleeping for %.0f seconds until %s", sleep_seconds, next_cycle.isoformat())
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    asyncio.run(main())
