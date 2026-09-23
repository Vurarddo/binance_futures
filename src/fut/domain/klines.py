"""Columnar kline container used between layers (numpy arrays, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

from fut.domain.entities import Source

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]
I8 = npt.NDArray[np.int8]

# Columns compared when checking whether history changed after the fact.
VALUE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)


@dataclass(frozen=True)
class KlineFrame:
    open_time: I64
    open: F64
    high: F64
    low: F64
    close: F64
    volume: F64
    close_time: I64
    quote_volume: F64
    trades: I64
    taker_buy_volume: F64
    taker_buy_quote_volume: F64
    source: I8

    def __post_init__(self) -> None:
        n = len(self.open_time)
        for f in fields(self):
            if len(getattr(self, f.name)) != n:
                raise ValueError(f"column {f.name} length mismatch")

    def __len__(self) -> int:
        return len(self.open_time)

    @classmethod
    def column_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def empty(cls) -> KlineFrame:
        return cls.from_columns({n: np.empty(0) for n in cls.column_names()})

    @classmethod
    def from_columns(cls, cols: dict[str, npt.NDArray[np.generic]]) -> KlineFrame:
        i64 = {"open_time", "close_time", "trades"}
        conv: dict[str, npt.NDArray[np.generic]] = {}
        for name in cls.column_names():
            arr = np.asarray(cols[name])
            if name in i64:
                conv[name] = arr.astype(np.int64, copy=False)
            elif name == "source":
                conv[name] = arr.astype(np.int8, copy=False)
            else:
                conv[name] = arr.astype(np.float64, copy=False)
        return cls(**conv)  # type: ignore[arg-type]

    def columns(self) -> dict[str, npt.NDArray[np.generic]]:
        return {n: getattr(self, n) for n in self.column_names()}

    def take(self, idx: npt.NDArray[np.intp] | npt.NDArray[np.bool_]) -> KlineFrame:
        return KlineFrame.from_columns({n: a[idx] for n, a in self.columns().items()})

    def with_source(self, source: Source) -> KlineFrame:
        cols = self.columns()
        cols["source"] = np.full(len(self), int(source), dtype=np.int8)
        return KlineFrame.from_columns(cols)

    def between(self, start_ms: int, end_ms: int) -> KlineFrame:
        """Rows with start_ms <= open_time < end_ms."""
        return self.take((self.open_time >= start_ms) & (self.open_time < end_ms))

    @staticmethod
    def concat(frames: list[KlineFrame]) -> KlineFrame:
        if not frames:
            return KlineFrame.empty()
        names = KlineFrame.column_names()
        return KlineFrame.from_columns(
            {n: np.concatenate([getattr(f, n) for f in frames]) for n in names}
        )


@dataclass(frozen=True)
class MergeStats:
    incoming_rows: int
    incoming_duplicates: int  # duplicate open_time inside the incoming batch
    added: int  # open_times not present before
    overlapping: int  # open_times present in both existing and incoming
    revised: int  # overlapping rows whose values differ (history changed after the fact)
    result_rows: int


def _dedupe_keep_last(frame: KlineFrame, priority: npt.NDArray[np.int64]) -> KlineFrame:
    """Sort by open_time, keep the row with the highest priority per open_time."""
    order = np.lexsort((priority, frame.open_time))
    f = frame.take(order)
    ot = f.open_time
    keep = np.ones(len(ot), dtype=bool)
    if len(ot) > 1:
        keep[:-1] = ot[:-1] != ot[1:]
    return f.take(keep)


def merge_klines(existing: KlineFrame, incoming: KlineFrame) -> tuple[KlineFrame, MergeStats]:
    """Merge `incoming` into `existing`, deduplicating on open_time.

    Winner per open_time: higher `Source` wins; on a tie the incoming row wins (a re-fetch
    of the same source replaces the old copy). Differences in value columns between the
    two copies are counted as revisions regardless of the winner.
    """
    n_in = len(incoming)
    # Within incoming: keep the last occurrence (stable order index as priority).
    inc = _dedupe_keep_last(incoming, np.arange(n_in, dtype=np.int64))
    inc_dups = n_in - len(inc)

    common, ie, ii = np.intersect1d(existing.open_time, inc.open_time, return_indices=True)
    revised = 0
    if len(common):
        diff = np.zeros(len(common), dtype=bool)
        for c in VALUE_COLUMNS:
            a, b = getattr(existing, c)[ie], getattr(inc, c)[ii]
            same = a == b
            if a.dtype.kind == "f":
                same |= np.isnan(a) & np.isnan(b)
            diff |= ~same
        revised = int(diff.sum())

    both = KlineFrame.concat([existing, inc])
    # priority = source * 2 + (1 if incoming else 0)  -> source first, then incoming wins ties
    tag = np.concatenate([np.zeros(len(existing), np.int64), np.ones(len(inc), np.int64)])
    priority = both.source.astype(np.int64) * 2 + tag
    merged = _dedupe_keep_last(both, priority)
    stats = MergeStats(
        incoming_rows=n_in,
        incoming_duplicates=inc_dups,
        added=len(inc) - len(common),
        overlapping=len(common),
        revised=revised,
        result_rows=len(merged),
    )
    return merged, stats
