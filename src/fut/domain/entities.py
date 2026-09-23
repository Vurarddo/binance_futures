"""Pure domain entities. Exchange-boundary numbers are `Decimal`."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum, StrEnum


class Dataset(StrEnum):
    """Kline-shaped datasets. Values match the data.binance.vision path segment."""

    KLINES = "klines"
    MARK_PRICE_KLINES = "markPriceKlines"


class Source(IntEnum):
    """Where a stored bar came from. Higher value wins when the same bar is re-ingested:
    the monthly archive is the final published record, REST is the provisional tail."""

    REST = 1
    DAILY = 2
    MONTHLY = 3


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    status: str
    contract_type: str
    onboard_ms: int
    delivery_ms: int
    base_asset: str
    quote_asset: str
    margin_asset: str
    tick_size: Decimal
    min_price: Decimal
    max_price: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    market_step_size: Decimal
    market_min_qty: Decimal
    market_max_qty: Decimal
    min_notional: Decimal
    max_num_orders: int
    percent_price_up: Decimal
    percent_price_down: Decimal
    maint_margin_percent: Decimal
    required_margin_percent: Decimal
    liquidation_fee: Decimal
    market_take_bound: Decimal
    order_types: tuple[str, ...]
    time_in_force: tuple[str, ...]

    @property
    def is_perpetual(self) -> bool:
        return self.contract_type == "PERPETUAL"

    @property
    def is_trading(self) -> bool:
        return self.status == "TRADING"


@dataclass(frozen=True)
class LeverageBracket:
    bracket: int
    initial_leverage: int
    notional_floor: Decimal
    notional_cap: Decimal
    maint_margin_ratio: Decimal
    cum: Decimal  # maintenance amount, used in the liquidation-price formula


@dataclass(frozen=True)
class SymbolBrackets:
    symbol: str
    brackets: tuple[LeverageBracket, ...]


@dataclass(frozen=True)
class FundingRate:
    symbol: str
    funding_time_ms: int  # actual settlement timestamp as reported (can carry ms jitter)
    funding_rate: Decimal
    mark_price: Decimal | None  # empty string in old records -> None
