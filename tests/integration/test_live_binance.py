"""Live checks against public Binance endpoints. Run explicitly: `uv run pytest -m network`.

These re-verify the facts recorded in CLAUDE.md ("Verified exchange facts")."""

from datetime import date, timedelta

import numpy as np
import pytest

from fut.adapters.binance.bulk import BinanceVisionArchive
from fut.adapters.binance.http import BinanceHttp
from fut.adapters.binance.rate_limit import WeightLimiter
from fut.adapters.binance.rest import BinanceRest
from fut.config import DEFAULT_UNIVERSE
from fut.domain.entities import Dataset
from fut.domain.time import DAY_MS, date_to_ms

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def rest():
    r = BinanceRest(
        BinanceHttp("https://fapi.binance.com", limiter=WeightLimiter(2400, 60, safety=0.25))
    )
    yield r
    r.http.close()


def test_exchange_info_rate_limits_and_universe(rest):
    info = rest.exchange_info()
    limits = {
        (x["rateLimitType"], x["interval"], x["intervalNum"]): x["limit"]
        for x in info.raw["rateLimits"]
    }
    assert limits[("REQUEST_WEIGHT", "MINUTE", 1)] == 2400
    specs = {s.symbol: s for s in info.specs}
    for sym in DEFAULT_UNIVERSE:
        assert specs[sym].is_perpetual and specs[sym].is_trading


def test_archive_matches_rest_for_one_day(rest):
    day = date.today() - timedelta(days=3)
    arch = BinanceVisionArchive()
    try:
        bf = arch.daily(Dataset.KLINES, "BTCUSDT", "1m", day)
    finally:
        arch.close()
    assert bf is not None and len(bf.frame) == 1440
    start = date_to_ms(day)
    fresh = rest.klines(Dataset.KLINES, "BTCUSDT", "1m", start, start + DAY_MS - 1)
    assert np.array_equal(fresh.open_time, bf.frame.open_time)
    assert np.array_equal(fresh.close, bf.frame.close)
    assert np.array_equal(fresh.volume, bf.frame.volume)


def test_funding_history_fields(rest):
    rates = rest.funding_rates(
        "BTCUSDT", date_to_ms(date(2020, 1, 1)), date_to_ms(date(2020, 1, 3)) - 1
    )
    assert len(rates) == 6  # 8h interval in Jan 2020; endTime is inclusive
    assert rates[0].mark_price is None  # empty string in old records
