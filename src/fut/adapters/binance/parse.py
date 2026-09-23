"""Raw Binance JSON -> domain objects. Nothing outside adapters sees raw payloads."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from fut.domain.entities import FundingRate, LeverageBracket, Source, SymbolBrackets, SymbolSpec
from fut.domain.klines import KlineFrame
from fut.domain.time import normalize_epoch_ms


class ExchangeFormatError(ValueError):
    pass


def _filters(sym: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["filterType"]: f for f in sym.get("filters", [])}


def _d(v: Any) -> Decimal:
    return Decimal(str(v))


def parse_symbol_spec(sym: dict[str, Any]) -> SymbolSpec:
    try:
        f = _filters(sym)
        price, lot, mlot = f["PRICE_FILTER"], f["LOT_SIZE"], f["MARKET_LOT_SIZE"]
        pct = f.get("PERCENT_PRICE", {})
        return SymbolSpec(
            symbol=sym["symbol"],
            status=sym["status"],
            contract_type=sym["contractType"],
            onboard_ms=int(sym["onboardDate"]),
            delivery_ms=int(sym["deliveryDate"]),
            base_asset=sym["baseAsset"],
            quote_asset=sym["quoteAsset"],
            margin_asset=sym["marginAsset"],
            tick_size=_d(price["tickSize"]),
            min_price=_d(price["minPrice"]),
            max_price=_d(price["maxPrice"]),
            step_size=_d(lot["stepSize"]),
            min_qty=_d(lot["minQty"]),
            max_qty=_d(lot["maxQty"]),
            market_step_size=_d(mlot["stepSize"]),
            market_min_qty=_d(mlot["minQty"]),
            market_max_qty=_d(mlot["maxQty"]),
            min_notional=_d(f["MIN_NOTIONAL"]["notional"]),
            max_num_orders=int(f.get("MAX_NUM_ORDERS", {}).get("limit", 0)),
            percent_price_up=_d(pct.get("multiplierUp", "0")),
            percent_price_down=_d(pct.get("multiplierDown", "0")),
            maint_margin_percent=_d(sym["maintMarginPercent"]),
            required_margin_percent=_d(sym["requiredMarginPercent"]),
            liquidation_fee=_d(sym["liquidationFee"]),
            market_take_bound=_d(sym["marketTakeBound"]),
            order_types=tuple(sym.get("orderTypes", ())),
            time_in_force=tuple(sym.get("timeInForce", ())),
        )
    except (KeyError, TypeError, ArithmeticError) as e:
        raise ExchangeFormatError(f"cannot parse symbol {sym.get('symbol')!r}: {e!r}") from e


def parse_exchange_info(raw: dict[str, Any]) -> tuple[int, tuple[SymbolSpec, ...]]:
    specs = []
    for sym in raw["symbols"]:
        # Some non-standard contracts lack MARKET_LOT_SIZE etc.; skip with no crash.
        try:
            specs.append(parse_symbol_spec(sym))
        except ExchangeFormatError:
            continue
    return int(raw["serverTime"]), tuple(specs)


def parse_kline_rows(rows: list[list[Any]], source: Source) -> KlineFrame:
    """REST kline arrays: [openTime, o, h, l, c, v, closeTime, quoteVol, trades,
    takerBuyBase, takerBuyQuote, ignore]."""
    if not rows:
        return KlineFrame.empty()
    for r in rows:
        if len(r) < 11:
            raise ExchangeFormatError(f"kline row has {len(r)} fields: {r!r}")
    cols = list(zip(*rows, strict=False))
    return KlineFrame.from_columns(
        {
            "open_time": normalize_epoch_ms(np.array(cols[0], dtype=np.int64)),
            "open": np.array(cols[1], dtype=np.float64),
            "high": np.array(cols[2], dtype=np.float64),
            "low": np.array(cols[3], dtype=np.float64),
            "close": np.array(cols[4], dtype=np.float64),
            "volume": np.array(cols[5], dtype=np.float64),
            "close_time": normalize_epoch_ms(np.array(cols[6], dtype=np.int64)),
            "quote_volume": np.array(cols[7], dtype=np.float64),
            "trades": np.array(cols[8], dtype=np.int64),
            "taker_buy_volume": np.array(cols[9], dtype=np.float64),
            "taker_buy_quote_volume": np.array(cols[10], dtype=np.float64),
            "source": np.full(len(rows), int(source), dtype=np.int8),
        }
    )


def parse_funding_rows(rows: list[dict[str, Any]]) -> list[FundingRate]:
    out = []
    for r in rows:
        mp = r.get("markPrice")
        out.append(
            FundingRate(
                symbol=r["symbol"],
                funding_time_ms=int(r["fundingTime"]),
                funding_rate=_d(r["fundingRate"]),
                mark_price=_d(mp) if mp not in (None, "") else None,
            )
        )
    return out


def parse_leverage_brackets(raw: list[dict[str, Any]]) -> list[SymbolBrackets]:
    out = []
    for item in raw:
        brackets = tuple(
            LeverageBracket(
                bracket=int(b["bracket"]),
                initial_leverage=int(b["initialLeverage"]),
                notional_floor=_d(b["notionalFloor"]),
                notional_cap=_d(b["notionalCap"]),
                maint_margin_ratio=_d(b["maintMarginRatio"]),
                cum=_d(b["cum"]),
            )
            for b in item["brackets"]
        )
        out.append(SymbolBrackets(symbol=item["symbol"], brackets=brackets))
    return out
