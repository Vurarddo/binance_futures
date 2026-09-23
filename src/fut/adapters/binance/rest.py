"""Binance USDⓈ-M REST adapter implementing the MarketData / AccountData ports."""

from __future__ import annotations

from typing import Any

from fut.adapters.binance.http import BinanceHttp
from fut.adapters.binance.parse import (
    parse_exchange_info,
    parse_funding_rows,
    parse_kline_rows,
    parse_leverage_brackets,
)
from fut.adapters.binance.rate_limit import WeightLimiter
from fut.domain.entities import Dataset, FundingRate, Source, SymbolBrackets
from fut.domain.klines import KlineFrame
from fut.domain.time import interval_ms
from fut.ports.data import ExchangeInfo

KLINE_PATHS = {
    Dataset.KLINES: "/fapi/v1/klines",
    Dataset.MARK_PRICE_KLINES: "/fapi/v1/markPriceKlines",
}
# 499 rows cost weight 2 (verified via X-MBX-USED-WEIGHT-1M, 2026-09-23): the best rows/weight.
KLINE_PAGE = 499
FUNDING_PAGE = 1000


def kline_weight(limit: int) -> int:
    """Weight of GET /fapi/v1/klines by `limit` (measured 2026-09-23)."""
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


def funding_limiter() -> WeightLimiter:
    """fundingRate/fundingInfo carry no weight header; docs state a separate
    500 requests / 5 min / IP limit. Keep well under it."""
    return WeightLimiter(500, 300.0, safety=0.5)


class BinanceRest:
    def __init__(self, http: BinanceHttp, funding_limit: WeightLimiter | None = None) -> None:
        self.http = http
        self._funding_limit = funding_limit or funding_limiter()

    # --- MarketData -------------------------------------------------------------------
    def exchange_info(self) -> ExchangeInfo:
        raw: dict[str, Any] = self.http.get("/fapi/v1/exchangeInfo", weight=1)
        server_time, specs = parse_exchange_info(raw)
        return ExchangeInfo(server_time_ms=server_time, specs=specs, raw=raw)

    def klines(
        self, dataset: Dataset, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> KlineFrame:
        step = interval_ms(interval)
        path = KLINE_PATHS[dataset]
        frames: list[KlineFrame] = []
        cur = start_ms
        while cur <= end_ms:
            rows: list[list[Any]] = self.http.get(
                path,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": KLINE_PAGE,
                },
                weight=kline_weight(KLINE_PAGE),
            )
            if not rows:
                break
            f = parse_kline_rows(rows, Source.REST)
            frames.append(f)
            last = int(f.open_time[-1])
            if len(rows) < KLINE_PAGE:
                break
            cur = last + step
        out = KlineFrame.concat(frames)
        return out.between(start_ms, end_ms + 1)

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> list[FundingRate]:
        out: list[FundingRate] = []
        cur = start_ms
        while cur <= end_ms:
            rows: list[dict[str, Any]] = self.http.get(
                "/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": FUNDING_PAGE},
                weight=1,
                extra_limiter=self._funding_limit,
            )
            if not rows:
                break
            batch = parse_funding_rows(rows)
            out.extend(batch)
            if len(rows) < FUNDING_PAGE:
                break
            cur = batch[-1].funding_time_ms + 1
        return out

    # --- AccountData ------------------------------------------------------------------
    def leverage_brackets(self) -> list[SymbolBrackets]:
        if not self.http.time_offset_ms:
            self.http.sync_time()
        raw: list[dict[str, Any]] = self.http.get("/fapi/v1/leverageBracket", weight=1, signed=True)
        return parse_leverage_brackets(raw)
