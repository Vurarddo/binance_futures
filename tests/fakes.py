"""In-memory fakes for the data ports + synthetic data helpers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np

from fut.domain.entities import Dataset, FundingRate, Source, SymbolSpec
from fut.domain.klines import KlineFrame
from fut.domain.time import MINUTE_MS, YearMonth, date_to_ms
from fut.ports.data import BulkFile, ExchangeInfo, daily_file_name, monthly_file_name


def make_frame(
    start_ms: int,
    n: int,
    *,
    step_ms: int = MINUTE_MS,
    source: Source = Source.MONTHLY,
    price: float = 100.0,
) -> KlineFrame:
    """Values are a pure function of open_time, so any two fetches of the same bar agree."""
    ot = start_ms + np.arange(n, dtype=np.int64) * step_ms
    k = ot // step_ms
    close = price * (1 + 0.01 * np.sin(k / 97.0))
    open_ = price * (1 + 0.01 * np.sin((k - 1) / 97.0))
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    vol = 1.0 + (k % 7)
    return KlineFrame.from_columns(
        {
            "open_time": ot,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "close_time": ot + step_ms - 1,
            "quote_volume": vol * close,
            "trades": 1 + (k % 50),
            "taker_buy_volume": vol / 2,
            "taker_buy_quote_volume": vol * close / 2,
            "source": np.full(n, int(source), dtype=np.int8),
        }
    )


def spec(symbol: str, onboard: date = date(2019, 9, 8)) -> SymbolSpec:
    d = Decimal
    return SymbolSpec(
        symbol=symbol,
        status="TRADING",
        contract_type="PERPETUAL",
        onboard_ms=date_to_ms(onboard),
        delivery_ms=4133404800000,
        base_asset=symbol[:-4],
        quote_asset="USDT",
        margin_asset="USDT",
        tick_size=d("0.1"),
        min_price=d("1"),
        max_price=d("1000000"),
        step_size=d("0.001"),
        min_qty=d("0.001"),
        max_qty=d("1000"),
        market_step_size=d("0.001"),
        market_min_qty=d("0.001"),
        market_max_qty=d("120"),
        min_notional=d("50"),
        max_num_orders=200,
        percent_price_up=d("1.05"),
        percent_price_down=d("0.95"),
        maint_margin_percent=d("2.5"),
        required_margin_percent=d("5"),
        liquidation_fee=d("0.0125"),
        market_take_bound=d("0.05"),
        order_types=("LIMIT", "MARKET"),
        time_in_force=("GTC",),
    )


class FakeArchive:
    """Publishes monthly files for months before `monthly_until` (exclusive) and daily files
    for days before `daily_until` (exclusive), except days in `missing_days`."""

    def __init__(
        self,
        monthly_until: YearMonth,
        daily_until: date,
        missing_days: set[date] | None = None,
        monthly_holes: set[date] | None = None,
    ) -> None:
        self.monthly_until = monthly_until
        self.daily_until = daily_until
        self.missing_days = missing_days or set()
        self.monthly_holes = monthly_holes or set()  # days a monthly archive omits
        self.calls: list[str] = []

    def monthly(
        self, dataset: Dataset, symbol: str, interval: str, month: YearMonth
    ) -> BulkFile | None:
        self.calls.append(f"M {dataset.value} {symbol} {month}")
        if month >= self.monthly_until:
            return None
        n = (month.end_ms - month.start_ms) // MINUTE_MS
        frame = make_frame(month.start_ms, n, source=Source.MONTHLY)
        keep = np.ones(n, dtype=bool)
        for d in self.monthly_holes:
            keep &= ~(
                (frame.open_time >= date_to_ms(d)) & (frame.open_time < date_to_ms(d) + 86_400_000)
            )
        return BulkFile(monthly_file_name(symbol, interval, month), "0" * 64, frame.take(keep))

    def daily(self, dataset: Dataset, symbol: str, interval: str, day: date) -> BulkFile | None:
        self.calls.append(f"D {dataset.value} {symbol} {day}")
        if day >= self.daily_until or day in self.missing_days:
            return None
        return BulkFile(
            daily_file_name(symbol, interval, day),
            "0" * 64,
            make_frame(date_to_ms(day), 1440, source=Source.DAILY),
        )


class FakeMarket:
    def __init__(self, symbols: list[str], funding: dict[str, list[FundingRate]] | None = None):
        self.specs = tuple(spec(s) for s in symbols)
        self.funding = funding or {}
        self.kline_calls: list[tuple[str, int, int]] = []
        self.funding_calls: list[tuple[str, int, int]] = []

    def exchange_info(self) -> ExchangeInfo:
        return ExchangeInfo(0, self.specs, {"symbols": [s.symbol for s in self.specs]})

    def klines(
        self, dataset: Dataset, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> KlineFrame:
        self.kline_calls.append((symbol, start_ms, end_ms))
        n = (end_ms - start_ms) // MINUTE_MS + 1
        # Same generator as the archive -> identical values for the same open_time.
        return make_frame(start_ms, n, source=Source.REST)

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> list[FundingRate]:
        self.funding_calls.append((symbol, start_ms, end_ms))
        return [f for f in self.funding.get(symbol, []) if start_ms <= f.funding_time_ms <= end_ms]
