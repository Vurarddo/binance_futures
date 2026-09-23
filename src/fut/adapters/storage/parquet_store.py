"""Hive-partitioned Parquet store queried with DuckDB.

Layout under `root`:
  klines/dataset=<ds>/symbol=<SYM>/interval=<iv>/month=<YYYY-MM>/data.parquet
  klines/.../month=<YYYY-MM>/_manifest.json        (which sources were ingested)
  funding/symbol=<SYM>/data.parquet
  snapshots/<kind>/<YYYY-MM-DD>.json

One writer per partition at a time (the backfill schedules work per partition). Writes are
atomic: temp file in the same directory + os.replace.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fut.domain.entities import Dataset, FundingRate, Source
from fut.domain.klines import KlineFrame, MergeStats, merge_klines
from fut.domain.time import YearMonth
from fut.ports.data import FundingSeries, IngestRecord, PartitionKey, PartitionManifest

_KLINE_SCHEMA = pa.schema(
    [
        ("open_time", pa.int64()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("close_time", pa.int64()),
        ("quote_volume", pa.float64()),
        ("trades", pa.int64()),
        ("taker_buy_volume", pa.float64()),
        ("taker_buy_quote_volume", pa.float64()),
        ("source", pa.int8()),
    ]
)
_FUNDING_SCHEMA = pa.schema(
    [
        ("funding_time", pa.int64()),
        ("funding_rate", pa.float64()),
        ("mark_price", pa.float64()),
    ]
)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # --- paths ------------------------------------------------------------------------
    def _series_dir(self, dataset: Dataset, symbol: str, interval: str) -> Path:
        return (
            self.root
            / "klines"
            / f"dataset={dataset.value}"
            / f"symbol={symbol}"
            / f"interval={interval}"
        )

    def partition_dir(self, key: PartitionKey) -> Path:
        return self._series_dir(key.dataset, key.symbol, key.interval) / f"month={key.month}"

    def _funding_path(self, symbol: str) -> Path:
        return self.root / "funding" / f"symbol={symbol}" / "data.parquet"

    # --- klines -----------------------------------------------------------------------
    def _read_partition(self, key: PartitionKey) -> KlineFrame:
        path = self.partition_dir(key) / "data.parquet"
        if not path.exists():
            return KlineFrame.empty()
        t = pq.read_table(path)
        return KlineFrame.from_columns(
            {n: t.column(n).to_numpy() for n in KlineFrame.column_names()}
        )

    def manifest(self, key: PartitionKey) -> PartitionManifest:
        path = self.partition_dir(key) / "_manifest.json"
        if not path.exists():
            return PartitionManifest()
        raw = json.loads(path.read_text())
        return PartitionManifest(
            ingested={
                k: IngestRecord(**{**v, "source": Source(v["source"])})
                for k, v in raw["ingested"].items()
            }
        )

    def merge_partition(
        self, key: PartitionKey, frame: KlineFrame, record: IngestRecord
    ) -> MergeStats:
        bad = (frame.open_time < key.month.start_ms) | (frame.open_time >= key.month.end_ms)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows outside partition {key.month}")
        merged, stats = merge_klines(self._read_partition(key), frame)
        table = pa.table(merged.columns(), schema=_KLINE_SCHEMA)
        _atomic_write_table(self.partition_dir(key) / "data.parquet", table)
        record.incoming_duplicates = stats.incoming_duplicates
        record.revised = stats.revised
        m = self.manifest(key)
        m.ingested[record.name] = record
        payload = {
            "ingested": {k: {**asdict(v), "source": int(v.source)} for k, v in m.ingested.items()}
        }
        _atomic_write_bytes(
            self.partition_dir(key) / "_manifest.json", json.dumps(payload, indent=1).encode()
        )
        return stats

    def partitions(
        self, dataset: Dataset | None = None, symbol: str | None = None
    ) -> list[PartitionKey]:
        base = self.root / "klines"
        out: list[PartitionKey] = []
        for p in sorted(base.glob("dataset=*/symbol=*/interval=*/month=*/data.parquet")):
            ds, sym, iv, mo = (part.split("=", 1)[1] for part in p.parts[-5:-1])
            key = PartitionKey(Dataset(ds), sym, iv, YearMonth.parse(mo))
            if (dataset is None or key.dataset == dataset) and (
                symbol is None or key.symbol == symbol
            ):
                out.append(key)
        return out

    def _files(
        self, dataset: Dataset, symbol: str, interval: str, start_ms: int | None, end_ms: int | None
    ) -> list[str]:
        out = []
        for key in self.partitions(dataset, symbol):
            if key.interval != interval:
                continue
            if start_ms is not None and key.month.end_ms <= start_ms:
                continue
            if end_ms is not None and key.month.start_ms >= end_ms:
                continue
            out.append(str(self.partition_dir(key) / "data.parquet"))
        return out

    def read_klines(
        self,
        dataset: Dataset,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> KlineFrame:
        """Rows with start_ms <= open_time < end_ms, sorted by open_time."""
        files = self._files(dataset, symbol, interval, start_ms, end_ms)
        if not files:
            return KlineFrame.empty()
        lo = start_ms if start_ms is not None else -(2**62)
        hi = end_ms if end_ms is not None else 2**62
        cols = ", ".join(KlineFrame.column_names())
        with duckdb.connect() as con:
            res = con.execute(
                f"SELECT {cols} FROM read_parquet(?) "  # noqa: S608 - column list is static
                "WHERE open_time >= ? AND open_time < ? ORDER BY open_time",
                [files, lo, hi],
            ).fetchnumpy()
        return KlineFrame.from_columns({k: np.asarray(v) for k, v in res.items()})

    def last_open_time(self, dataset: Dataset, symbol: str, interval: str) -> int | None:
        files = self._files(dataset, symbol, interval, None, None)
        if not files:
            return None
        # Only the latest partition matters.
        with duckdb.connect() as con:
            row = con.execute("SELECT max(open_time) FROM read_parquet(?)", [files[-1]]).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def series_stats(self, dataset: Dataset, symbol: str, interval: str) -> dict[str, Any]:
        files = self._files(dataset, symbol, interval, None, None)
        if not files:
            return {"rows": 0}
        with duckdb.connect() as con:
            rows, first, last = con.execute(  # type: ignore[misc]
                "SELECT count(*), min(open_time), max(open_time) FROM read_parquet(?)", [files]
            ).fetchone()
            by_source = dict(
                con.execute(
                    "SELECT source, count(*) FROM read_parquet(?) GROUP BY source", [files]
                ).fetchall()
            )
        return {
            "rows": int(rows),
            "first": int(first),
            "last": int(last),
            "months": len(files),
            "by_source": {Source(int(k)).name: int(v) for k, v in by_source.items()},
        }

    def series(self) -> list[tuple[Dataset, str, str]]:
        return sorted({(k.dataset, k.symbol, k.interval) for k in self.partitions()})

    # --- funding ----------------------------------------------------------------------
    def read_funding(self, symbol: str) -> FundingSeries:
        path = self._funding_path(symbol)
        if not path.exists():
            return FundingSeries(symbol, [], [], [])
        t = pq.read_table(path)
        mp = t.column("mark_price").to_pylist()
        return FundingSeries(
            symbol,
            t.column("funding_time").to_pylist(),
            t.column("funding_rate").to_pylist(),
            [None if v is None else float(v) for v in mp],
        )

    def merge_funding(self, symbol: str, rates: list[FundingRate]) -> int:
        cur = self.read_funding(symbol)
        rows: dict[int, tuple[float, float | None]] = {
            t: (r, m)
            for t, r, m in zip(cur.funding_time_ms, cur.funding_rate, cur.mark_price, strict=True)
        }
        before = len(rows)
        for fr in rates:
            if fr.symbol != symbol:
                raise ValueError(f"funding for {fr.symbol} passed as {symbol}")
            mp = float(fr.mark_price) if isinstance(fr.mark_price, Decimal) else None
            rows[fr.funding_time_ms] = (float(fr.funding_rate), mp)
        keys = sorted(rows)
        table = pa.table(
            {
                "funding_time": keys,
                "funding_rate": [rows[k][0] for k in keys],
                "mark_price": [rows[k][1] for k in keys],
            },
            schema=_FUNDING_SCHEMA,
        )
        _atomic_write_table(self._funding_path(symbol), table)
        return len(rows) - before

    def funding_symbols(self) -> list[str]:
        return sorted(
            p.parent.name.split("=", 1)[1]
            for p in (self.root / "funding").glob("symbol=*/data.parquet")
        )

    # --- snapshots --------------------------------------------------------------------
    def save_snapshot(self, kind: str, day: date, payload: object) -> str:
        path = self.root / "snapshots" / kind / f"{day.isoformat()}.json"
        _atomic_write_bytes(path, json.dumps(payload, indent=1, default=str).encode())
        return str(path)

    def latest_snapshot(self, kind: str) -> Any:
        files = sorted((self.root / "snapshots" / kind).glob("*.json"))
        return json.loads(files[-1].read_text()) if files else None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """DuckDB connection with `klines` and `funding` views for ad-hoc research."""
        con = duckdb.connect()
        kl = str(self.root / "klines" / "*" / "*" / "*" / "*" / "data.parquet")
        fu = str(self.root / "funding" / "*" / "data.parquet")
        if list((self.root / "klines").glob("*/*/*/*/data.parquet")):
            con.execute(
                # Path comes from our own data root, not user input.
                f"CREATE VIEW klines AS SELECT * FROM read_parquet('{kl}', hive_partitioning=true)"  # noqa: S608
            )
        if list((self.root / "funding").glob("*/data.parquet")):
            con.execute(
                f"CREATE VIEW funding AS SELECT * FROM read_parquet('{fu}', hive_partitioning=true)"  # noqa: S608
            )
        return con
