from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from fut.adapters.storage.parquet_store import ParquetStore
from fut.domain.entities import Dataset, FundingRate, Source
from fut.domain.time import MINUTE_MS, YearMonth
from fut.ports.data import IngestRecord, PartitionKey
from tests.fakes import make_frame

JAN = YearMonth(2025, 1)
FEB = YearMonth(2025, 2)


def key(m: YearMonth, ds: Dataset = Dataset.KLINES) -> PartitionKey:
    return PartitionKey(ds, "BTCUSDT", "1m", m)


def rec(name: str, source: Source = Source.MONTHLY, rows: int = 0) -> IngestRecord:
    return IngestRecord(name, source, None, rows, 0)


def test_partition_layout(tmp_path):
    store = ParquetStore(tmp_path)
    store.merge_partition(key(JAN), make_frame(JAN.start_ms, 10), rec("a"))
    p = tmp_path / "klines/dataset=klines/symbol=BTCUSDT/interval=1m/month=2025-01"
    assert (p / "data.parquet").exists() and (p / "_manifest.json").exists()
    assert store.partitions() == [key(JAN)]
    assert not list(p.glob(".tmp-*"))


def test_rejects_rows_outside_partition(tmp_path):
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="outside partition"):
        store.merge_partition(key(JAN), make_frame(FEB.start_ms, 10), rec("a"))


def test_merge_idempotent_and_manifest(tmp_path):
    store = ParquetStore(tmp_path)
    f = make_frame(JAN.start_ms, 100)
    s1 = store.merge_partition(key(JAN), f, rec("a"))
    s2 = store.merge_partition(key(JAN), f, rec("a"))
    assert (s1.added, s2.added, s2.revised) == (100, 0, 0)
    assert len(store.read_klines(Dataset.KLINES, "BTCUSDT", "1m")) == 100
    assert store.manifest(key(JAN)).has("a")


def test_read_range_across_partitions(tmp_path):
    store = ParquetStore(tmp_path)
    store.merge_partition(key(JAN), make_frame(FEB.start_ms - 60 * MINUTE_MS, 60), rec("j"))
    store.merge_partition(key(FEB), make_frame(FEB.start_ms, 60), rec("f"))
    got = store.read_klines(
        Dataset.KLINES,
        "BTCUSDT",
        "1m",
        FEB.start_ms - 10 * MINUTE_MS,
        FEB.start_ms + 10 * MINUTE_MS,
    )
    assert len(got) == 20
    assert np.all(np.diff(got.open_time) == MINUTE_MS)
    assert store.last_open_time(Dataset.KLINES, "BTCUSDT", "1m") == FEB.start_ms + 59 * MINUTE_MS
    # datasets are separate series
    assert store.read_klines(Dataset.MARK_PRICE_KLINES, "BTCUSDT", "1m").open_time.size == 0
    stats = store.series_stats(Dataset.KLINES, "BTCUSDT", "1m")
    assert stats["rows"] == 120 and stats["months"] == 2 and stats["by_source"] == {"MONTHLY": 120}


def test_funding_merge_incremental(tmp_path):
    store = ParquetStore(tmp_path)

    def fr(t: int, mp: str | None = "1") -> FundingRate:
        return FundingRate("BTCUSDT", t, Decimal("0.0001"), Decimal(mp) if mp else None)

    assert store.merge_funding("BTCUSDT", [fr(1, None), fr(2)]) == 2
    assert store.merge_funding("BTCUSDT", [fr(2), fr(3)]) == 1
    s = store.read_funding("BTCUSDT")
    assert s.funding_time_ms == [1, 2, 3]
    assert s.mark_price[0] is None
    assert store.funding_symbols() == ["BTCUSDT"]


def test_snapshot(tmp_path):
    store = ParquetStore(tmp_path)
    store.save_snapshot("exchange_info", date(2026, 9, 23), {"a": 1})
    assert store.latest_snapshot("exchange_info") == {"a": 1}


def test_duckdb_views(tmp_path):
    store = ParquetStore(tmp_path)
    store.merge_partition(key(JAN), make_frame(JAN.start_ms, 10), rec("a"))
    with store.connect() as con:
        row = con.execute("SELECT symbol, month, count(*) FROM klines GROUP BY ALL").fetchone()
    assert row == ("BTCUSDT", "2025-01", 10)
