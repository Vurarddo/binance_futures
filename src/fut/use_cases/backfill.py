"""Resumable, idempotent backfill: bulk archives first, REST for the provisional tail.

Per (dataset, symbol, interval, month) partition:
  1. month already has its monthly archive in the manifest -> skip (complete).
  2. month is finished and a monthly archive exists -> ingest it (supersedes dailies/REST).
  3. otherwise ingest each missing daily archive for days before today.
Then REST tops up from the last stored bar to the last *closed* bar, and funding history is
fetched incrementally from the last stored settlement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

from fut.domain.entities import Dataset, Source, SymbolSpec
from fut.domain.time import (
    DAY_MS,
    YearMonth,
    date_to_ms,
    interval_ms,
    last_closed_open_time,
    month_range,
    ms_to_date,
)
from fut.ports.data import (
    BulkArchive,
    Clock,
    HistoricalStore,
    IngestRecord,
    MarketData,
    PartitionKey,
    daily_file_name,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillRequest:
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    datasets: tuple[Dataset, ...]
    start: date
    funding: bool = True
    rest_topup: bool = True
    # REST top-up is for the recent tail only; beyond this, rely on archives.
    max_rest_days: int = 7
    workers: int = 4


@dataclass
class SeriesResult:
    monthly_files: int = 0
    daily_files: int = 0
    skipped_months: int = 0
    rest_rows: int = 0
    rows_added: int = 0
    revised: int = 0
    missing_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BackfillReport:
    series: dict[tuple[Dataset, str, str], SeriesResult] = field(default_factory=dict)
    funding_added: dict[str, int] = field(default_factory=dict)
    unknown_symbols: list[str] = field(default_factory=list)
    exchange_info_snapshot: str | None = None

    @property
    def ok(self) -> bool:
        return not self.unknown_symbols and not any(r.errors for r in self.series.values())


@dataclass
class _MonthOutcome:
    key: PartitionKey
    monthly: int = 0
    daily: int = 0
    skipped: bool = False
    added: int = 0
    revised: int = 0
    missing: list[str] = field(default_factory=list)


def _ingest_month(
    key: PartitionKey,
    today: date,
    onboard_day: date,
    archive: BulkArchive,
    store: HistoricalStore,
    clock: Clock,
) -> _MonthOutcome:
    out = _MonthOutcome(key)
    manifest = store.manifest(key)
    if any(r.source == Source.MONTHLY for r in manifest.ingested.values()):
        out.skipped = True
        return out
    if key.month.last_day < today:
        bf = archive.monthly(key.dataset, key.symbol, key.interval, key.month)
        if bf is not None:
            rec = IngestRecord(bf.name, Source.MONTHLY, bf.sha256, len(bf.frame), clock.now_ms())
            st = store.merge_partition(key, bf.frame, rec)
            out.monthly, out.added, out.revised = 1, st.added, st.revised
            log.info("%s %s %s %s: monthly %d rows", *_k(key), len(bf.frame))
            return out
    for day in key.month.days():
        if day >= today or day < onboard_day:
            continue
        name = daily_file_name(key.symbol, key.interval, day)
        if manifest.has(name):
            continue
        bf = archive.daily(key.dataset, key.symbol, key.interval, day)
        if bf is None:
            out.missing.append(name)
            continue
        rec = IngestRecord(bf.name, Source.DAILY, bf.sha256, len(bf.frame), clock.now_ms())
        st = store.merge_partition(key, bf.frame, rec)
        out.daily += 1
        out.added += st.added
        out.revised += st.revised
    if out.daily:
        log.info("%s %s %s %s: %d daily files", *_k(key), out.daily)
    return out


def _k(key: PartitionKey) -> tuple[str, str, str, str]:
    return key.dataset.value, key.symbol, key.interval, str(key.month)


def run_backfill(
    req: BackfillRequest,
    *,
    archive: BulkArchive,
    market: MarketData,
    store: HistoricalStore,
    clock: Clock,
    progress: Callable[[str], None] = lambda _msg: None,
) -> BackfillReport:
    report = BackfillReport()
    now = clock.now_ms()
    today = ms_to_date(now)

    info = market.exchange_info()
    report.exchange_info_snapshot = store.save_snapshot("exchange_info", today, info.raw)
    specs: dict[str, SymbolSpec] = {s.symbol: s for s in info.specs}
    symbols = [s for s in req.symbols if s in specs]
    report.unknown_symbols = [s for s in req.symbols if s not in specs]
    for s in report.unknown_symbols:
        log.error("%s not in exchangeInfo (delisted or typo) - skipped", s)

    # 1. Bulk archives, parallel over partitions (each partition has exactly one task).
    tasks: list[tuple[PartitionKey, date]] = []
    for sym in symbols:
        onboard = ms_to_date(specs[sym].onboard_ms)
        first = YearMonth.of(max(req.start, onboard))
        for ds in req.datasets:
            for iv in req.intervals:
                report.series[(ds, sym, iv)] = SeriesResult()
                for m in month_range(first, YearMonth.of(today)):
                    tasks.append((PartitionKey(ds, sym, iv, m), max(req.start, onboard)))

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, req.workers)) as pool:
        futs = {
            pool.submit(_ingest_month, key, today, first_day, archive, store, clock): key
            for key, first_day in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            res = report.series[(key.dataset, key.symbol, key.interval)]
            done += 1
            try:
                o = fut.result()
            except Exception as e:  # keep going; report per series
                res.errors.append(f"{key.month}: {e!r}")
                log.exception("bulk ingest failed for %s", key)
                continue
            res.monthly_files += o.monthly
            res.daily_files += o.daily
            res.skipped_months += int(o.skipped)
            res.rows_added += o.added
            res.revised += o.revised
            res.missing_files.extend(o.missing)
            if done % 25 == 0 or done == len(tasks):
                progress(f"bulk: {done}/{len(tasks)} partitions")

    # 2. REST top-up of the provisional tail.
    if req.rest_topup:
        for (ds, sym, iv), res in report.series.items():
            try:
                _topup(ds, sym, iv, req, market, store, clock, res)
            except Exception as e:
                res.errors.append(f"rest top-up: {e!r}")
                log.exception("REST top-up failed for %s %s %s", ds, sym, iv)
        progress("rest top-up done")

    # 3. Funding history (incremental).
    if req.funding:
        for sym in symbols:
            report.funding_added[sym] = _funding(sym, req.start, market, store, now)
        progress("funding done")
    return report


def _topup(
    ds: Dataset,
    sym: str,
    iv: str,
    req: BackfillRequest,
    market: MarketData,
    store: HistoricalStore,
    clock: Clock,
    res: SeriesResult,
) -> None:
    step = interval_ms(iv)
    now = clock.now_ms()
    end = last_closed_open_time(now, step)
    last = store.last_open_time(ds, sym, iv)
    floor = end - req.max_rest_days * DAY_MS
    start = (last + step) if last is not None else max(floor, date_to_ms(req.start))
    if start < floor:
        log.warning(
            "%s %s %s: tail gap starts %s, beyond %dd REST window; topping up last %dd only",
            ds.value,
            sym,
            iv,
            ms_to_date(start),
            req.max_rest_days,
            req.max_rest_days,
        )
        start = floor
    if start > end:
        return
    frame = market.klines(ds, sym, iv, start, end)
    if not len(frame):
        return
    for m in month_range(YearMonth.of_ms(start), YearMonth.of_ms(end)):
        part = frame.between(m.start_ms, m.end_ms)
        if not len(part):
            continue
        name = f"rest:{int(part.open_time[0])}-{int(part.open_time[-1])}"
        rec = IngestRecord(name, Source.REST, None, len(part), clock.now_ms())
        st = store.merge_partition(PartitionKey(ds, sym, iv, m), part, rec)
        res.rest_rows += len(part)
        res.rows_added += st.added
        res.revised += st.revised


def _funding(sym: str, start: date, market: MarketData, store: HistoricalStore, now_ms: int) -> int:
    cur = store.read_funding(sym)
    since = (max(cur.funding_time_ms) + 1) if cur.funding_time_ms else date_to_ms(start)
    # A settlement at exactly `start` would be missed by +1 only if already stored; fine.
    rates = market.funding_rates(sym, since, now_ms)
    added = store.merge_funding(sym, rates) if rates else 0
    log.info("%s funding: +%d settlements", sym, added)
    return added
