"""Coverage and data-quality reports over the historical store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from fut.domain.entities import Dataset, Source
from fut.domain.klines import KlineFrame, merge_klines
from fut.domain.quality import FundingQuality, KlineQuality, check_funding, check_klines
from fut.domain.time import DAY_MS, date_to_ms, interval_ms, ms_to_date
from fut.ports.data import HistoricalStore, MarketData, PartitionKey


@dataclass(frozen=True)
class SeriesCoverage:
    dataset: Dataset
    symbol: str
    interval: str
    rows: int
    first_ms: int | None
    last_ms: int | None
    expected_rows: int
    months: int
    by_source: dict[str, int]

    @property
    def completeness(self) -> float:
        return self.rows / self.expected_rows if self.expected_rows else 0.0


@dataclass(frozen=True)
class FundingCoverage:
    symbol: str
    rows: int
    first_ms: int | None
    last_ms: int | None


def coverage(store: HistoricalStore) -> tuple[list[SeriesCoverage], list[FundingCoverage]]:
    out = []
    for ds, sym, iv in store.series():
        st = store.series_stats(ds, sym, iv)
        step = interval_ms(iv)
        first, last = st.get("first"), st.get("last")
        expected = (last - first) // step + 1 if first is not None and last is not None else 0
        out.append(
            SeriesCoverage(
                ds,
                sym,
                iv,
                st["rows"],
                first,
                last,
                expected,
                st.get("months", 0),
                st.get("by_source", {}),
            )
        )
    fund = []
    for sym in store.funding_symbols():
        f = store.read_funding(sym)
        t = f.funding_time_ms
        fund.append(FundingCoverage(sym, len(t), min(t) if t else None, max(t) if t else None))
    return out, fund


@dataclass(frozen=True)
class RecheckResult:
    """A window re-fetched from REST and compared against stored (bulk) history."""

    day: date
    stored_rows: int
    refetched_rows: int
    overlapping: int
    revised: int  # overlapping bars whose values differ
    only_stored: int
    only_refetched: int
    stored_sources: dict[str, int]


@dataclass
class SeriesQuality:
    dataset: Dataset
    symbol: str
    interval: str
    klines: KlineQuality
    ingest_revisions: int  # value changes seen when a bar was re-ingested from another source
    ingest_duplicates: int  # duplicate open_times inside ingested files
    rechecks: list[RecheckResult] = field(default_factory=list)


@dataclass
class QualityReport:
    series: list[SeriesQuality]
    funding: dict[str, FundingQuality]


def _recheck_days(first_ms: int, last_ms: int) -> list[date]:
    """Deterministic sample: oldest full day, a middle day, newest full day before the tail."""
    first_day = ms_to_date(first_ms + DAY_MS)  # skip a possibly partial first day
    last_day = ms_to_date(last_ms - 2 * DAY_MS)
    if last_day < first_day:
        return []
    mid = ms_to_date((date_to_ms(first_day) + date_to_ms(last_day)) // 2)
    return sorted({first_day, mid, last_day})


def recheck_window(
    store: HistoricalStore, market: MarketData, ds: Dataset, sym: str, iv: str, day: date
) -> RecheckResult:
    start = date_to_ms(day)
    end = start + DAY_MS
    stored = store.read_klines(ds, sym, iv, start, end)
    fresh = market.klines(ds, sym, iv, start, end - 1)
    # merge_klines counts differing values on overlapping open_times.
    _, st = merge_klines(stored, fresh)
    only_stored = len(np.setdiff1d(stored.open_time, fresh.open_time))
    only_fresh = len(np.setdiff1d(fresh.open_time, stored.open_time))
    srcs = {
        Source(int(k)).name: int(v)
        for k, v in zip(*np.unique(stored.source, return_counts=True), strict=True)
    }
    return RecheckResult(
        day, len(stored), len(fresh), st.overlapping, st.revised, only_stored, only_fresh, srcs
    )


def quality(
    store: HistoricalStore,
    *,
    symbols: set[str] | None = None,
    market: MarketData | None = None,
) -> QualityReport:
    series: list[SeriesQuality] = []
    for ds, sym, iv in store.series():
        if symbols and sym not in symbols:
            continue
        frame: KlineFrame = store.read_klines(ds, sym, iv)
        q = check_klines(frame, interval_ms(iv))
        manifests = [store.manifest(k) for k in _keys(store, ds, sym, iv)]
        sq = SeriesQuality(
            ds,
            sym,
            iv,
            q,
            ingest_revisions=sum(m.total_revised for m in manifests),
            ingest_duplicates=sum(m.total_incoming_duplicates for m in manifests),
        )
        if market is not None and q.first_open_ms is not None and q.last_open_ms is not None:
            for day in _recheck_days(q.first_open_ms, q.last_open_ms):
                sq.rechecks.append(recheck_window(store, market, ds, sym, iv, day))
        series.append(sq)
    funding = {}
    for sym in store.funding_symbols():
        if symbols and sym not in symbols:
            continue
        f = store.read_funding(sym)
        funding[sym] = check_funding(
            np.array(f.funding_time_ms, dtype=np.int64), np.array(f.funding_rate, dtype=np.float64)
        )
    return QualityReport(series, funding)


def _keys(store: HistoricalStore, ds: Dataset, sym: str, iv: str) -> list[PartitionKey]:
    return [k for k in store.partitions(ds, sym) if k.interval == iv]
