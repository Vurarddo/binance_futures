"""Pure data-quality checks over kline and funding series."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from fut.domain.klines import KlineFrame
from fut.domain.time import HOUR_MS

# A 1m close-to-close move beyond both thresholds is flagged as an outlier.
DEFAULT_OUTLIER_SIGMAS = 20.0
DEFAULT_OUTLIER_MIN_ABS_LOGRET = 0.05


@dataclass(frozen=True)
class Gap:
    first_missing_ms: int
    last_missing_ms: int
    bars: int


@dataclass(frozen=True)
class Outlier:
    open_time_ms: int
    log_return: float


@dataclass(frozen=True)
class KlineQuality:
    rows: int
    first_open_ms: int | None
    last_open_ms: int | None
    expected_rows: int
    missing_bars: int
    n_gaps: int
    largest_gaps: tuple[Gap, ...]
    duplicates: int
    unsorted: int
    misaligned_open: int
    bad_close_time: int
    ohlc_violations: int
    non_positive_prices: int
    negative_volume: int
    zero_volume_bars: int
    outliers: int
    top_outliers: tuple[Outlier, ...]

    @property
    def completeness(self) -> float:
        return self.rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def hard_errors(self) -> int:
        """Problems that make a bar unusable (as opposed to gaps/zero volume which are real)."""
        return (
            self.duplicates
            + self.unsorted
            + self.misaligned_open
            + self.bad_close_time
            + self.ohlc_violations
            + self.non_positive_prices
            + self.negative_volume
        )


def find_gaps(open_time: npt.NDArray[np.int64], step_ms: int) -> list[Gap]:
    """Gaps between consecutive (sorted, unique) open times."""
    if len(open_time) < 2:
        return []
    d = np.diff(open_time)
    idx = np.nonzero(d > step_ms)[0]
    return [
        Gap(
            first_missing_ms=int(open_time[i] + step_ms),
            last_missing_ms=int(open_time[i + 1] - step_ms),
            bars=int(d[i] // step_ms - 1),
        )
        for i in idx
    ]


def check_klines(
    frame: KlineFrame,
    step_ms: int,
    *,
    top_n: int = 5,
    outlier_sigmas: float = DEFAULT_OUTLIER_SIGMAS,
    outlier_min_abs: float = DEFAULT_OUTLIER_MIN_ABS_LOGRET,
) -> KlineQuality:
    n = len(frame)
    if n == 0:
        return KlineQuality(0, None, None, 0, 0, 0, (), 0, 0, 0, 0, 0, 0, 0, 0, 0, ())

    ot_raw = frame.open_time
    unsorted = int((np.diff(ot_raw) < 0).sum())
    order = np.argsort(ot_raw, kind="stable")
    f = frame.take(order)
    ot = f.open_time
    dup_mask = np.zeros(n, dtype=bool)
    dup_mask[1:] = ot[1:] == ot[:-1]
    duplicates = int(dup_mask.sum())
    u = f.take(~dup_mask)
    uot = u.open_time

    first, last = int(uot[0]), int(uot[-1])
    expected = (last - first) // step_ms + 1
    gaps = find_gaps(uot, step_ms)
    missing = sum(g.bars for g in gaps)

    o, h, lo, c, v = u.open, u.high, u.low, u.close, u.volume
    ohlc_bad = (h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (lo > h)
    non_pos = (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)

    # Outliers: close-to-close log returns between *adjacent* bars only.
    adjacent = np.diff(uot) == step_ms
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(c[1:] / c[:-1])
    lr_adj = np.where(adjacent & np.isfinite(lr), lr, np.nan)
    finite = lr_adj[np.isfinite(lr_adj)]
    outlier_idx: npt.NDArray[np.intp] = np.empty(0, dtype=np.intp)
    if len(finite) > 10:
        mad = float(np.median(np.abs(finite - np.median(finite))))
        sigma = 1.4826 * mad
        thresh = max(outlier_min_abs, outlier_sigmas * sigma)
        with np.errstate(invalid="ignore"):
            outlier_idx = np.nonzero(np.abs(lr_adj) > thresh)[0]
    top = sorted(outlier_idx, key=lambda i: -abs(float(lr_adj[i])))[:top_n]

    return KlineQuality(
        rows=len(u),
        first_open_ms=first,
        last_open_ms=last,
        expected_rows=int(expected),
        missing_bars=int(missing),
        n_gaps=len(gaps),
        largest_gaps=tuple(sorted(gaps, key=lambda g: -g.bars)[:top_n]),
        duplicates=duplicates,
        unsorted=unsorted,
        misaligned_open=int((uot % step_ms != 0).sum()),
        bad_close_time=int((u.close_time != uot + step_ms - 1).sum()),
        ohlc_violations=int(ohlc_bad.sum()),
        non_positive_prices=int(non_pos.sum()),
        negative_volume=int((v < 0).sum()),
        zero_volume_bars=int((v == 0).sum()),
        outliers=len(outlier_idx),
        top_outliers=tuple(Outlier(int(uot[i + 1]), float(lr_adj[i])) for i in top),
    )


@dataclass(frozen=True)
class FundingQuality:
    rows: int
    first_ms: int | None
    last_ms: int | None
    duplicates: int
    # Interval between consecutive settlements, rounded to whole hours -> count.
    interval_hours: dict[int, int] = field(default_factory=dict)
    # Timestamps whose interval to the previous settlement changed (interval regime switch).
    interval_changes: tuple[tuple[int, int, int], ...] = ()  # (time_ms, old_h, new_h)
    off_grid: int = 0  # settlement not within 1s of a whole hour
    max_abs_rate: float = 0.0


def check_funding(
    funding_time_ms: npt.NDArray[np.int64],
    funding_rate: npt.NDArray[np.float64],
    *,
    max_changes: int = 20,
) -> FundingQuality:
    n = len(funding_time_ms)
    if n == 0:
        return FundingQuality(0, None, None, 0)
    order = np.argsort(funding_time_ms, kind="stable")
    t = funding_time_ms[order]
    r = funding_rate[order]
    duplicates = int((np.diff(t) == 0).sum())
    t = np.unique(t)
    hours = np.rint(np.diff(t) / HOUR_MS).astype(np.int64)
    counts = Counter(int(x) for x in hours)
    changes: list[tuple[int, int, int]] = []
    for i in np.nonzero(np.diff(hours) != 0)[0][:max_changes]:
        changes.append((int(t[i + 2]), int(hours[i]), int(hours[i + 1])))
    offset = t % HOUR_MS
    off_grid = int(((offset > 1000) & (offset < HOUR_MS - 1000)).sum())
    return FundingQuality(
        rows=len(t),
        first_ms=int(t[0]),
        last_ms=int(t[-1]),
        duplicates=duplicates,
        interval_hours=dict(sorted(counts.items())),
        interval_changes=tuple(changes),
        off_grid=off_grid,
        max_abs_rate=float(np.nanmax(np.abs(r))) if len(r) else 0.0,
    )
