"""`fut` command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from fut.adapters.binance.bulk import BinanceVisionArchive
from fut.adapters.binance.http import BinanceHttp
from fut.adapters.binance.rate_limit import WeightLimiter
from fut.adapters.binance.rest import BinanceRest
from fut.adapters.clock import SystemClock
from fut.adapters.storage.parquet_store import ParquetStore
from fut.config import Settings
from fut.domain.entities import Dataset
from fut.domain.time import from_ms, ms_to_date
from fut.use_cases.backfill import BackfillRequest, run_backfill
from fut.use_cases.inspect import coverage, quality

log = logging.getLogger("fut")

# REQUEST_WEIGHT limit from exchangeInfo.rateLimits (2400/min, verified 2026-09-23).
REQUEST_WEIGHT_PER_MIN = 2400


def _ts(ms: int | None) -> str:
    return "-" if ms is None else from_ms(ms).strftime("%Y-%m-%d %H:%M")


def _market(settings: Settings) -> BinanceRest:
    http = BinanceHttp(
        settings.market_data_url, limiter=WeightLimiter(REQUEST_WEIGHT_PER_MIN, 60.0, safety=0.5)
    )
    return BinanceRest(http)


def _account(settings: Settings) -> BinanceRest:
    http = BinanceHttp(
        settings.trading_url,
        limiter=WeightLimiter(REQUEST_WEIGHT_PER_MIN, 60.0, safety=0.5),
        api_key=settings.api_key,
        api_secret=settings.api_secret,
    )
    return BinanceRest(http)


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def cmd_backfill(args: argparse.Namespace, settings: Settings) -> int:
    store = ParquetStore(settings.data_dir)
    req = BackfillRequest(
        symbols=tuple(args.symbols or settings.universe),
        intervals=tuple(args.intervals or settings.intervals),
        datasets=tuple(Dataset(d) for d in args.datasets) if args.datasets else settings.datasets,
        start=date.fromisoformat(args.start) if args.start else settings.history_start,
        funding=not args.no_funding,
        rest_topup=not args.no_topup,
        workers=args.workers or settings.download_workers,
    )
    archive = BinanceVisionArchive(settings.bulk_data_url)
    market = _market(settings)
    clock = SystemClock()
    try:
        rep = run_backfill(
            req, archive=archive, market=market, store=store, clock=clock, progress=print
        )
    finally:
        archive.close()
        market.http.close()
    today = ms_to_date(clock.now_ms())
    print(f"\nexchangeInfo snapshot: {rep.exchange_info_snapshot}")
    print(
        f"{'dataset':16} {'symbol':9} {'iv':3} {'monthly':>7} {'daily':>5} {'skip':>4} "
        f"{'rest':>6} {'added':>9} {'revised':>7} {'repaired':>8} {'missing':>7} errors"
    )
    for (ds, sym, iv), r in sorted(rep.series.items()):
        # Archives for the last ~2 days are routinely not published yet.
        recent = {f"{sym}-{iv}-{(today - timedelta(days=d)).isoformat()}.zip" for d in (1, 2)}
        missing = [m for m in r.missing_files if m not in recent]
        print(
            f"{ds.value:16} {sym:9} {iv:3} {r.monthly_files:>7} {r.daily_files:>5} "
            f"{r.skipped_months:>4} {r.rest_rows:>6} {r.rows_added:>9} {r.revised:>7} "
            f"{r.repaired_days:>8} "
            f"{len(missing):>7} {len(r.errors)}"
        )
        for m in missing[:5]:
            print(f"    missing archive: {m}")
        for d in r.unrepairable_days[:5]:
            print(f"    unrepairable day (no data in archive or REST): {d}")
        for e in r.errors[:5]:
            print(f"    error: {e}")
    if rep.funding_added:
        print(
            "funding settlements added: "
            + ", ".join(f"{k}=+{v}" for k, v in rep.funding_added.items())
        )
    if rep.unknown_symbols:
        print(f"unknown symbols: {rep.unknown_symbols}")
    return 0 if rep.ok else 1


def cmd_coverage(args: argparse.Namespace, settings: Settings) -> int:
    store = ParquetStore(settings.data_dir)
    series, funding = coverage(store)
    print(
        f"{'dataset':16} {'symbol':9} {'iv':3} {'first':16} {'last':16} {'rows':>9} "
        f"{'expected':>9} {'complete':>8} months  sources"
    )
    for c in series:
        src = " ".join(f"{k}={v}" for k, v in sorted(c.by_source.items()))
        print(
            f"{c.dataset.value:16} {c.symbol:9} {c.interval:3} {_ts(c.first_ms):16} "
            f"{_ts(c.last_ms):16} {c.rows:>9} {c.expected_rows:>9} {c.completeness:>8.4%} "
            f"{c.months:>6}  {src}"
        )
    print(f"\n{'funding':9} {'first':16} {'last':16} {'settlements':>11}")
    for f in funding:
        print(f"{f.symbol:9} {_ts(f.first_ms):16} {_ts(f.last_ms):16} {f.rows:>11}")
    return 0


def cmd_quality(args: argparse.Namespace, settings: Settings) -> int:
    store = ParquetStore(settings.data_dir)
    market = _market(settings) if args.recheck else None
    try:
        rep = quality(store, symbols=set(args.symbols) if args.symbols else None, market=market)
    finally:
        if market is not None:
            market.http.close()
    hdr = (
        f"{'dataset':16} {'symbol':9} {'iv':3} {'rows':>9} {'missing':>7} {'gaps':>4} "
        f"{'dups':>4} {'misal':>5} {'ohlc':>4} {'zeroV':>6} {'outl':>4} {'revis':>5} "
        f"{'recheck(diff/overlap)':>22}"
    )
    print(hdr)
    failed = False
    for s in rep.series:
        q = s.klines
        rc_over = sum(r.overlapping for r in s.rechecks)
        rc_diff = sum(r.revised + r.only_stored + r.only_refetched for r in s.rechecks)
        rc = f"{rc_diff}/{rc_over}" if s.rechecks else "-"
        print(
            f"{s.dataset.value:16} {s.symbol:9} {s.interval:3} {q.rows:>9} {q.missing_bars:>7} "
            f"{q.n_gaps:>4} {q.duplicates:>4} {q.misaligned_open + q.bad_close_time:>5} "
            f"{q.ohlc_violations + q.non_positive_prices:>4} {q.zero_volume_bars:>6} "
            f"{q.outliers:>4} {s.ingest_revisions:>5} {rc:>22}"
        )
        for g in q.largest_gaps[: args.top]:
            print(f"    gap {_ts(g.first_missing_ms)} .. {_ts(g.last_missing_ms)} ({g.bars} bars)")
        for o in q.top_outliers[: args.top]:
            print(f"    outlier {_ts(o.open_time_ms)} logret={o.log_return:+.4f}")
        failed |= q.hard_errors > 0 or rc_diff > 0
    print(
        f"\n{'funding':9} {'rows':>6} {'dups':>4} {'offgrid':>7} {'max|rate|':>9}"
        "  intervals(h:count)  changes"
    )
    for sym, fq in rep.funding.items():
        iv = " ".join(f"{h}h:{n}" for h, n in fq.interval_hours.items())
        ch = "; ".join(f"{_ts(t)} {a}h->{b}h" for t, a, b in fq.interval_changes[:5])
        print(
            f"{sym:9} {fq.rows:>6} {fq.duplicates:>4} {fq.off_grid:>7} "
            f"{fq.max_abs_rate:>9.5f}  {iv}  {ch}"
        )
    out = settings.data_dir / "reports" / f"quality-{date.today().isoformat()}.json"  # noqa: DTZ011
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(rep), indent=1, default=str))
    print(f"\nreport: {out}")
    return 1 if failed else 0


def cmd_specs(args: argparse.Namespace, settings: Settings) -> int:
    store = ParquetStore(settings.data_dir)
    market = _market(settings)
    try:
        info = market.exchange_info()
    finally:
        market.http.close()
    today = ms_to_date(info.server_time_ms)
    print("exchangeInfo:", store.save_snapshot("exchange_info", today, info.raw))
    specs = {s.symbol: s for s in info.specs}
    print(
        f"{'symbol':9} {'status':8} {'tick':>10} {'step':>7} {'minNotional':>11} "
        f"{'maintM%':>7} {'liqFee':>8} onboard"
    )
    for sym in args.symbols or settings.universe:
        s = specs.get(sym)
        if s is None:
            print(f"{sym:9} NOT LISTED")
            continue
        print(
            f"{sym:9} {s.status:8} {s.tick_size!s:>10} {s.step_size!s:>7} {s.min_notional!s:>11} "
            f"{s.maint_margin_percent!s:>7} {s.liquidation_fee!s:>8} {_ts(s.onboard_ms)}"
        )
    if not settings.has_credentials:
        env = settings.trading_env
        print(f"\nleverage brackets: skipped (no FUT_API_KEY/FUT_API_SECRET for {env})")
        return 0
    acct = _account(settings)
    try:
        brackets = acct.leverage_brackets()
    finally:
        acct.http.close()
    path = store.save_snapshot(
        f"leverage_brackets_{settings.trading_env}", today, _jsonable(brackets)
    )
    print(f"\nleverage brackets ({settings.trading_env}): {len(brackets)} symbols -> {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fut", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="download/refresh history (resumable, idempotent)")
    b.add_argument("--symbols", nargs="+")
    b.add_argument("--intervals", nargs="+")
    b.add_argument("--datasets", nargs="+", choices=[d.value for d in Dataset])
    b.add_argument("--start", help="YYYY-MM-DD (default: settings.history_start)")
    b.add_argument("--no-funding", action="store_true")
    b.add_argument("--no-topup", action="store_true", help="skip REST top-up of the recent tail")
    b.add_argument("--workers", type=int)
    b.set_defaults(func=cmd_backfill)

    c = sub.add_parser("coverage", help="what is stored")
    c.set_defaults(func=cmd_coverage)

    q = sub.add_parser("quality", help="data-quality report")
    q.add_argument("--symbols", nargs="+")
    q.add_argument(
        "--no-recheck",
        dest="recheck",
        action="store_false",
        help="skip re-fetching sample days via REST for comparison",
    )
    q.add_argument("--top", type=int, default=3)
    q.set_defaults(func=cmd_quality)

    s = sub.add_parser("specs", help="exchangeInfo snapshot (+ leverage brackets if keyed)")
    s.add_argument("--symbols", nargs="+")
    s.set_defaults(func=cmd_specs)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings()
    rc: int = args.func(args, settings)
    return rc


if __name__ == "__main__":
    sys.exit(main())
