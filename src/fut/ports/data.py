"""Ports for historical data. Adapters implement these; use cases depend only on them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from fut.domain.entities import Dataset, FundingRate, Source, SymbolBrackets, SymbolSpec
from fut.domain.klines import KlineFrame, MergeStats
from fut.domain.time import YearMonth


class Clock(Protocol):
    def now_ms(self) -> int: ...


@dataclass(frozen=True)
class BulkFile:
    """A verified archive from the bulk-data site, already parsed."""

    name: str
    sha256: str
    frame: KlineFrame


def monthly_file_name(symbol: str, interval: str, month: YearMonth) -> str:
    return f"{symbol}-{interval}-{month}.zip"


def daily_file_name(symbol: str, interval: str, day: date) -> str:
    return f"{symbol}-{interval}-{day.isoformat()}.zip"


class BulkArchive(Protocol):
    """data.binance.vision-like archive. Returns None when the file does not exist (yet).
    Implementations must verify the published checksum and raise on mismatch."""

    def monthly(
        self, dataset: Dataset, symbol: str, interval: str, month: YearMonth
    ) -> BulkFile | None: ...

    def daily(self, dataset: Dataset, symbol: str, interval: str, day: date) -> BulkFile | None: ...


@dataclass(frozen=True)
class ExchangeInfo:
    server_time_ms: int
    specs: tuple[SymbolSpec, ...]
    raw: dict[str, object]


class MarketData(Protocol):
    """Public REST market data."""

    def exchange_info(self) -> ExchangeInfo: ...

    def klines(
        self, dataset: Dataset, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> KlineFrame:
        """Closed-interval [start_ms, end_ms] on open_time, paginated internally."""
        ...

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> list[FundingRate]: ...


class AccountData(Protocol):
    """Signed, read-only account endpoints."""

    def leverage_brackets(self) -> list[SymbolBrackets]: ...


@dataclass
class IngestRecord:
    name: str  # e.g. BTCUSDT-1m-2025-01.zip or rest:<start>-<end>
    source: Source
    sha256: str | None
    rows: int
    ingested_at_ms: int
    incoming_duplicates: int = 0
    revised: int = 0


@dataclass
class PartitionManifest:
    ingested: dict[str, IngestRecord] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.ingested

    @property
    def total_revised(self) -> int:
        return sum(r.revised for r in self.ingested.values())

    @property
    def total_incoming_duplicates(self) -> int:
        return sum(r.incoming_duplicates for r in self.ingested.values())


@dataclass(frozen=True)
class PartitionKey:
    dataset: Dataset
    symbol: str
    interval: str
    month: YearMonth


@dataclass(frozen=True)
class FundingSeries:
    symbol: str
    funding_time_ms: list[int]
    funding_rate: list[float]
    mark_price: list[float | None]


class HistoricalStore(Protocol):
    def read_klines(
        self,
        dataset: Dataset,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> KlineFrame: ...

    def merge_partition(
        self, key: PartitionKey, frame: KlineFrame, record: IngestRecord
    ) -> MergeStats: ...

    def manifest(self, key: PartitionKey) -> PartitionManifest: ...

    def partitions(
        self, dataset: Dataset | None = None, symbol: str | None = None
    ) -> list[PartitionKey]: ...

    def last_open_time(self, dataset: Dataset, symbol: str, interval: str) -> int | None: ...

    def series(self) -> list[tuple[Dataset, str, str]]:
        """All stored (dataset, symbol, interval) series."""
        ...

    def series_stats(self, dataset: Dataset, symbol: str, interval: str) -> dict[str, Any]:
        """rows, first, last, months, by_source."""
        ...

    def read_funding(self, symbol: str) -> FundingSeries: ...

    def merge_funding(self, symbol: str, rates: list[FundingRate]) -> int:
        """Returns number of new settlements added."""
        ...

    def funding_symbols(self) -> list[str]: ...

    def save_snapshot(self, kind: str, day: date, payload: object) -> str:
        """Persist a raw JSON snapshot (exchangeInfo, brackets). Returns its path."""
        ...
