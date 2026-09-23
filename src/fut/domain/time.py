"""Time helpers. Everything is UTC; timestamps are integer epoch milliseconds."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import numpy.typing as npt

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

# Fixed-length kline intervals supported by Binance USDⓈ-M futures. "1w"/"1M" are
# deliberately excluded: "1M" is not fixed-length and neither is needed for research.
INTERVAL_MS: dict[str, int] = {
    "1m": MINUTE_MS,
    "3m": 3 * MINUTE_MS,
    "5m": 5 * MINUTE_MS,
    "15m": 15 * MINUTE_MS,
    "30m": 30 * MINUTE_MS,
    "1h": HOUR_MS,
    "2h": 2 * HOUR_MS,
    "4h": 4 * HOUR_MS,
    "6h": 6 * HOUR_MS,
    "8h": 8 * HOUR_MS,
    "12h": 12 * HOUR_MS,
    "1d": DAY_MS,
}

# Anything above this is not epoch-ms (it would be year ~5138); treat as microseconds.
_MAX_PLAUSIBLE_MS = 100_000_000_000_000


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(f"unsupported interval {interval!r}") from None


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("naive datetime; use timezone-aware UTC datetimes")
    return int(dt.timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def date_to_ms(d: date) -> int:
    return calendar.timegm(d.timetuple()) * 1000


def ms_to_date(ms: int) -> date:
    return from_ms(ms).date()


def normalize_epoch_ms(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Convert microsecond timestamps to milliseconds; leave ms untouched.

    data.binance.vision switched *spot* files to microseconds in 2025. USDⓈ-M files are ms
    (verified 2026-09-23), but guard anyway so a silent format change cannot corrupt data.
    """
    out = values.copy()
    mask = out > _MAX_PLAUSIBLE_MS
    out[mask] //= 1000
    return out


@dataclass(frozen=True, order=True)
class YearMonth:
    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"bad month {self.month}")

    @classmethod
    def parse(cls, s: str) -> YearMonth:
        y, m = s.split("-")
        return cls(int(y), int(m))

    @classmethod
    def of(cls, d: date) -> YearMonth:
        return cls(d.year, d.month)

    @classmethod
    def of_ms(cls, ms: int) -> YearMonth:
        return cls.of(ms_to_date(ms))

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def first_day(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def last_day(self) -> date:
        return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])

    def next(self) -> YearMonth:
        return YearMonth(self.year + self.month // 12, self.month % 12 + 1)

    @property
    def start_ms(self) -> int:
        return date_to_ms(self.first_day)

    @property
    def end_ms(self) -> int:
        """Exclusive end."""
        return self.next().start_ms

    def days(self) -> list[date]:
        d, out = self.first_day, []
        while d <= self.last_day:
            out.append(d)
            d += timedelta(days=1)
        return out


def month_range(first: YearMonth, last: YearMonth) -> list[YearMonth]:
    """Inclusive range of months."""
    out: list[YearMonth] = []
    m = first
    while m <= last:
        out.append(m)
        m = m.next()
    return out


def last_closed_open_time(now_ms: int, step_ms: int) -> int:
    """Open time of the most recent *fully closed* bar at `now_ms`."""
    return (now_ms // step_ms) * step_ms - step_ms
