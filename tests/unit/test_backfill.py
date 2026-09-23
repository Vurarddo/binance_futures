from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np

from fut.adapters.clock import FixedClock
from fut.adapters.storage.parquet_store import ParquetStore
from fut.domain.entities import Dataset, FundingRate, Source
from fut.domain.time import HOUR_MS, MINUTE_MS, YearMonth, date_to_ms, to_ms
from fut.use_cases.backfill import BackfillRequest, run_backfill
from fut.use_cases.inspect import coverage, quality
from tests.fakes import FakeArchive, FakeMarket, make_frame

# "Now" is 2025-03-10 12:00:30 UTC. Monthly archives exist up to Feb; dailies up to Mar 8.
NOW = to_ms(datetime(2025, 3, 10, 12, 0, 30, tzinfo=UTC))
REQ = BackfillRequest(
    symbols=("BTCUSDT",),
    intervals=("1m",),
    datasets=(Dataset.KLINES,),
    start=date(2025, 1, 1),
    workers=2,
)


def funding() -> dict[str, list[FundingRate]]:
    t0 = date_to_ms(date(2025, 1, 1))
    return {
        "BTCUSDT": [
            FundingRate("BTCUSDT", t0 + i * 8 * HOUR_MS, Decimal("0.0001"), Decimal("1"))
            for i in range(200)
        ]
    }


def run(tmp_path, archive, market, now=NOW, req=REQ):
    store = ParquetStore(tmp_path)
    rep = run_backfill(req, archive=archive, market=market, store=store, clock=FixedClock(now))
    return store, rep


def test_full_backfill_is_contiguous_to_last_closed_bar(tmp_path):
    archive = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9))
    market = FakeMarket(["BTCUSDT"], funding())
    store, rep = run(tmp_path, archive, market)
    assert rep.ok
    r = rep.series[(Dataset.KLINES, "BTCUSDT", "1m")]
    assert (r.monthly_files, r.daily_files) == (2, 8)
    assert r.missing_files == ["BTCUSDT-1m-2025-03-09.zip"]  # not published yet
    # REST tops up from Mar 9 00:00 to the last closed bar (12:00 is still open -> 11:59).
    assert market.kline_calls == [
        ("BTCUSDT", date_to_ms(date(2025, 3, 9)), NOW - 30_000 - MINUTE_MS)
    ]
    f = store.read_klines(Dataset.KLINES, "BTCUSDT", "1m")
    assert f.open_time[0] == date_to_ms(date(2025, 1, 1))
    assert f.open_time[-1] == NOW - 30_000 - MINUTE_MS
    assert np.all(np.diff(f.open_time) == MINUTE_MS)
    q = quality(store)
    assert q.series[0].klines.missing_bars == 0 and q.series[0].klines.hard_errors == 0
    # Funding: only settlements up to now.
    assert store.read_funding("BTCUSDT").funding_time_ms[-1] <= NOW
    assert rep.funding_added["BTCUSDT"] == len(store.read_funding("BTCUSDT").funding_time_ms)


def test_second_run_fetches_only_what_is_missing(tmp_path):
    archive = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9))
    market = FakeMarket(["BTCUSDT"], funding())
    store, _ = run(tmp_path, archive, market)
    last_funding = store.read_funding("BTCUSDT").funding_time_ms[-1]
    archive.calls.clear()
    market.kline_calls.clear()
    market.funding_calls.clear()

    later = NOW + 5 * MINUTE_MS
    store, rep = run(tmp_path, archive, market, now=later)
    r = rep.series[(Dataset.KLINES, "BTCUSDT", "1m")]
    assert r.skipped_months == 2  # Jan, Feb complete
    assert r.revised == 0
    # March: only the still-unpublished day is re-asked; no monthly request for Jan/Feb.
    assert archive.calls == ["D klines BTCUSDT 2025-03-09"]
    # REST only for the 5 new closed bars.
    ((_, start, end),) = market.kline_calls
    assert (end - start) // MINUTE_MS + 1 == 5
    # Funding resumes after the last stored settlement.
    assert market.funding_calls[0][1] == last_funding + 1


def test_monthly_archive_supersedes_dailies_and_rest(tmp_path):
    # First run early in March: Feb has no monthly archive yet -> dailies + REST.
    early = to_ms(datetime(2025, 3, 1, 0, 30, tzinfo=UTC))
    archive = FakeArchive(YearMonth(2025, 2), date(2025, 2, 28))  # Feb 28 daily not yet out
    market = FakeMarket(["BTCUSDT"])
    store, _ = run(tmp_path, archive, market, now=early)
    feb = store.read_klines(
        Dataset.KLINES, "BTCUSDT", "1m", YearMonth(2025, 2).start_ms, YearMonth(2025, 2).end_ms
    )
    assert set(np.unique(feb.source)) == {Source.DAILY, Source.REST}

    # Later: the Feb monthly archive is published and wins.
    archive2 = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9))
    store, rep = run(tmp_path, archive2, market)
    feb = store.read_klines(
        Dataset.KLINES, "BTCUSDT", "1m", YearMonth(2025, 2).start_ms, YearMonth(2025, 2).end_ms
    )
    assert set(np.unique(feb.source)) == {Source.MONTHLY}
    assert len(feb) == 28 * 1440
    assert rep.series[(Dataset.KLINES, "BTCUSDT", "1m")].revised == 0


def test_missing_daily_in_current_month_repaired_via_rest(tmp_path):
    archive = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9), missing_days={date(2025, 3, 4)})
    market = FakeMarket(["BTCUSDT"])
    store, rep = run(tmp_path, archive, market)
    r = rep.series[(Dataset.KLINES, "BTCUSDT", "1m")]
    assert "BTCUSDT-1m-2025-03-04.zip" in r.missing_files
    assert r.repaired_days == 1
    assert quality(store).series[0].klines.missing_bars == 0


def test_gap_reported_when_repair_disabled(tmp_path):
    archive = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9), missing_days={date(2025, 3, 4)})
    req = BackfillRequest(
        symbols=("BTCUSDT",),
        intervals=("1m",),
        datasets=(Dataset.KLINES,),
        start=date(2025, 1, 1),
        repair_gaps=False,
        funding=False,
    )
    store, _ = run(tmp_path, archive, FakeMarket(["BTCUSDT"]), req=req)
    q = quality(store).series[0].klines
    assert q.missing_bars == 1440 and q.n_gaps == 1


def test_unknown_symbol_reported_and_onboard_respected(tmp_path):
    archive = FakeArchive(YearMonth(2025, 3), date(2025, 3, 9))
    market = FakeMarket(["BTCUSDT"])
    req = BackfillRequest(
        symbols=("BTCUSDT", "NOPEUSDT"),
        intervals=("1m",),
        datasets=(Dataset.KLINES,),
        start=date(2024, 12, 1),
        rest_topup=False,
        funding=False,
    )
    store, rep = run(tmp_path, archive, market, req=req)
    assert rep.unknown_symbols == ["NOPEUSDT"] and not rep.ok
    series, _ = coverage(store)
    assert [(c.symbol, c.months) for c in series] == [("BTCUSDT", 4)]


def test_rest_topup_capped_to_recent_window(tmp_path):
    # Archive has nothing at all: REST must not try to fetch years of 1m bars.
    archive = FakeArchive(YearMonth(2000, 1), date(2000, 1, 1))
    market = FakeMarket(["BTCUSDT"])
    run(
        tmp_path,
        archive,
        market,
        req=BackfillRequest(
            symbols=("BTCUSDT",),
            intervals=("1m",),
            datasets=(Dataset.KLINES,),
            start=date(2020, 1, 1),
            max_rest_days=2,
            funding=False,
        ),
    )
    ((_, start, end),) = market.kline_calls
    assert (end - start) <= 2 * 24 * HOUR_MS


def test_monthly_holes_repaired_from_daily_then_rest_and_not_retried(tmp_path):
    hole_daily = date(2025, 1, 10)  # monthly omits it, daily archive has it
    hole_rest = date(2025, 1, 20)  # monthly omits it, daily archive missing -> REST
    archive = FakeArchive(
        YearMonth(2025, 3),
        date(2025, 3, 9),
        missing_days={hole_rest},
        monthly_holes={hole_daily, hole_rest},
    )
    market = FakeMarket(["BTCUSDT"])
    store, rep = run(tmp_path, archive, market)
    r = rep.series[(Dataset.KLINES, "BTCUSDT", "1m")]
    assert r.repaired_days == 2 and r.unrepairable_days == []
    assert (
        "BTCUSDT",
        date_to_ms(hole_rest),
        date_to_ms(hole_rest) + 1440 * MINUTE_MS - 1,
    ) in market.kline_calls
    assert quality(store).series[0].klines.missing_bars == 0

    archive.calls.clear()
    market.kline_calls.clear()
    run(tmp_path, archive, market, now=NOW + MINUTE_MS)
    assert not any("2025-01" in c for c in archive.calls)
    assert all(start >= date_to_ms(date(2025, 3, 10)) for _, start, _ in market.kline_calls)


def test_unrepairable_day_reported_once_tried(tmp_path):
    hole = date(2025, 1, 15)
    archive = FakeArchive(
        YearMonth(2025, 3), date(2025, 3, 9), missing_days={hole}, monthly_holes={hole}
    )

    class EmptyDayMarket(FakeMarket):
        def klines(self, dataset, symbol, interval, start_ms, end_ms):
            if start_ms == date_to_ms(hole):
                self.kline_calls.append((symbol, start_ms, end_ms))
                return make_frame(start_ms, 0)
            return super().klines(dataset, symbol, interval, start_ms, end_ms)

    market = EmptyDayMarket(["BTCUSDT"])
    _, rep = run(tmp_path, archive, market)
    assert rep.series[(Dataset.KLINES, "BTCUSDT", "1m")].unrepairable_days == ["2025-01:2025-01-15"]
    market.kline_calls.clear()
    run(tmp_path, archive, market, now=NOW + MINUTE_MS)
    assert all(start != date_to_ms(hole) for _, start, _ in market.kline_calls)
