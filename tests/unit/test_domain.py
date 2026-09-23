from datetime import date

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from fut.domain.entities import Source
from fut.domain.klines import KlineFrame, merge_klines
from fut.domain.quality import check_funding, check_klines, find_gaps
from fut.domain.time import (
    HOUR_MS,
    MINUTE_MS,
    YearMonth,
    date_to_ms,
    last_closed_open_time,
    month_range,
    normalize_epoch_ms,
)
from tests.fakes import make_frame

T0 = date_to_ms(date(2025, 1, 1))


# --- time ---------------------------------------------------------------------------------
def test_year_month_bounds():
    m = YearMonth(2024, 2)
    assert m.end_ms - m.start_ms == 29 * 24 * HOUR_MS  # leap year
    assert YearMonth(2024, 12).next() == YearMonth(2025, 1)
    assert [str(x) for x in month_range(YearMonth(2024, 11), YearMonth(2025, 2))] == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]
    assert len(YearMonth(2025, 2).days()) == 28


def test_last_closed_open_time():
    # At 12:00:30 the 11:59 bar is the last fully closed one.
    now = T0 + 12 * HOUR_MS + 30_000
    assert last_closed_open_time(now, MINUTE_MS) == T0 + 12 * HOUR_MS - MINUTE_MS
    # Exactly on the boundary the bar that just closed counts.
    assert last_closed_open_time(T0 + HOUR_MS, MINUTE_MS) == T0 + HOUR_MS - MINUTE_MS


def test_normalize_epoch_ms_converts_microseconds_only():
    ms = np.array([1735689600000, 1735689600000 * 1000], dtype=np.int64)
    assert normalize_epoch_ms(ms).tolist() == [1735689600000, 1735689600000]


# --- merge --------------------------------------------------------------------------------
def test_merge_adds_and_dedupes():
    a = make_frame(T0, 10)
    b = make_frame(T0 + 5 * MINUTE_MS, 10)
    merged, st_ = merge_klines(a, b)
    assert len(merged) == 15
    assert st_.overlapping == 5 and st_.added == 5 and st_.revised == 0
    assert np.all(np.diff(merged.open_time) == MINUTE_MS)


def test_merge_is_idempotent():
    a = make_frame(T0, 100)
    m1, _ = merge_klines(KlineFrame.empty(), a)
    m2, st_ = merge_klines(m1, a)
    assert st_.added == 0 and st_.revised == 0
    for c in KlineFrame.column_names():
        assert np.array_equal(getattr(m1, c), getattr(m2, c))


def test_merge_higher_source_wins_and_revisions_counted():
    rest = make_frame(T0, 10, source=Source.REST)
    cols = make_frame(T0, 10, source=Source.MONTHLY).columns()
    cols["close"] = cols["close"].copy()
    cols["close"][3] += 1.0  # the archive disagrees on one bar
    monthly = KlineFrame.from_columns(cols)
    merged, st_ = merge_klines(rest, monthly)
    assert st_.revised == 1
    assert np.all(merged.source == Source.MONTHLY)
    assert merged.close[3] == monthly.close[3]
    # REST arriving after the archive must not overwrite it.
    merged2, st2 = merge_klines(merged, rest)
    assert np.all(merged2.source == Source.MONTHLY)
    assert st2.revised == 1


def test_merge_counts_incoming_duplicates():
    a = make_frame(T0, 5)
    dup = KlineFrame.concat([a, a.take(np.array([0, 1]))])
    merged, st_ = merge_klines(KlineFrame.empty(), dup)
    assert st_.incoming_duplicates == 2 and len(merged) == 5


@given(
    st.lists(st.integers(0, 500), min_size=0, max_size=60),
    st.lists(st.integers(0, 500), min_size=0, max_size=60),
)
def test_merge_property_union_sorted_unique(xs, ys):
    def frame(idx):
        idx_arr = np.array(sorted(idx), dtype=np.int64)
        f = make_frame(T0, 501)
        return f.take(idx_arr) if len(idx_arr) else KlineFrame.empty()

    merged, st_ = merge_klines(frame(xs), frame(ys))
    expected = sorted(set(xs) | set(ys))
    assert merged.open_time.tolist() == [T0 + i * MINUTE_MS for i in expected]
    assert st_.revised == 0


# --- quality ------------------------------------------------------------------------------
def test_find_gaps():
    ot = np.array([0, 1, 2, 5, 6, 10], dtype=np.int64) * MINUTE_MS
    gaps = find_gaps(ot, MINUTE_MS)
    assert [
        (g.first_missing_ms // MINUTE_MS, g.last_missing_ms // MINUTE_MS, g.bars) for g in gaps
    ] == [
        (3, 4, 2),
        (7, 9, 3),
    ]


@given(st.sets(st.integers(0, 2000), min_size=2, max_size=300))
def test_gap_bars_equal_missing(present):
    ot = np.array(sorted(present), dtype=np.int64) * MINUTE_MS
    gaps = find_gaps(ot, MINUTE_MS)
    span = (max(present) - min(present)) + 1
    assert sum(g.bars for g in gaps) == span - len(present)


def test_check_klines_clean_frame():
    q = check_klines(make_frame(T0, 1000), MINUTE_MS)
    assert q.rows == 1000 and q.expected_rows == 1000 and q.missing_bars == 0
    assert q.hard_errors == 0 and q.outliers == 0 and q.completeness == 1.0


def test_check_klines_detects_problems():
    f = make_frame(T0, 1000)
    cols = {k: v.copy() for k, v in f.columns().items()}
    cols["high"][10] = cols["low"][10] - 1  # OHLC violation
    cols["open_time"][20] += 1  # misaligned
    cols["close_time"][30] += 5  # bad close time
    cols["volume"][40] = 0  # zero volume
    cols["close"][500] *= 1.5  # outlier jump (and back)
    cols["high"][500] = cols["close"][500]
    g = KlineFrame.from_columns(cols)
    keep = np.ones(1000, dtype=bool)
    keep[600:610] = False  # 10-bar gap
    g = KlineFrame.concat([g.take(keep), g.take(np.array([100]))])  # a duplicate
    q = check_klines(g, MINUTE_MS)
    assert q.ohlc_violations >= 1
    assert q.misaligned_open == 1
    assert q.bad_close_time >= 2  # the shifted bar + the misaligned one
    assert q.zero_volume_bars == 1
    assert q.duplicates == 1
    assert q.unsorted == 1
    assert q.outliers >= 1
    assert any(gp.bars == 10 for gp in q.largest_gaps)


def test_check_funding_intervals_and_changes():
    t8 = [i * 8 * HOUR_MS for i in range(10)]
    t4 = [t8[-1] + (i + 1) * 4 * HOUR_MS for i in range(5)]
    times = np.array([*t8, *t4], dtype=np.int64)
    times[3] += 4  # ms jitter like the real API (fundingTime ...004)
    q = check_funding(times, np.full(len(times), 1e-4))
    assert q.interval_hours == {4: 5, 8: 9}
    assert q.interval_changes[0][1:] == (8, 4)
    assert q.off_grid == 0
    assert q.duplicates == 0
