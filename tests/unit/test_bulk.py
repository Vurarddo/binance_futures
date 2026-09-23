import hashlib
import io
import zipfile
from datetime import date

import httpx
import numpy as np
import pytest

from fut.adapters.binance.bulk import (
    BinanceVisionArchive,
    ChecksumMismatchError,
    parse_kline_csv,
    verify_checksum,
)
from fut.domain.entities import Dataset, Source
from fut.domain.time import YearMonth

HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
)
ROWS = (
    "1735689600000,93548.80,93599.90,93514.20,93599.90,71.187,1735689659999,6658941.51700,1660,"
    "35.553,3325936.37090,0\n"
    "1735689660000,93599.90,93637.70,93577.60,93637.70,39.526,1735689719999,3699698.66020,1371,"
    "24.390,2282997.20480,0\n"
)


def _zip(name: str, content: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, content)
    return buf.getvalue()


def test_verify_checksum_ok_and_mismatch():
    data = b"hello"
    digest = hashlib.sha256(data).hexdigest()
    assert verify_checksum(data, f"{digest}  f.zip", "f.zip") == digest
    with pytest.raises(ChecksumMismatchError, match="sha256"):
        verify_checksum(b"tampered", f"{digest}  f.zip", "f.zip")
    with pytest.raises(ChecksumMismatchError, match="names"):
        verify_checksum(data, f"{digest}  other.zip", "f.zip")
    with pytest.raises(ChecksumMismatchError, match="malformed"):
        verify_checksum(data, "garbage", "f.zip")


@pytest.mark.parametrize("header", [True, False])
def test_parse_csv_with_and_without_header(header):
    f = parse_kline_csv(((HEADER if header else "") + ROWS).encode(), Source.MONTHLY)
    assert len(f) == 2
    assert f.open_time.tolist() == [1735689600000, 1735689660000]
    assert f.close[0] == 93599.90
    assert f.trades.tolist() == [1660, 1371]
    assert np.all(f.source == Source.MONTHLY)


def test_parse_csv_microseconds_normalised():
    rows = ROWS.replace("1735689600000,", "1735689600000000,").replace(
        "1735689659999,", "1735689659999999,"
    )
    f = parse_kline_csv(rows.encode(), Source.DAILY)
    assert f.open_time[0] == 1735689600000
    assert f.close_time[0] == 1735689659999


def _archive(files: dict[str, bytes]) -> tuple[BinanceVisionArchive, list[str]]:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path.split("/data/futures/um/", 1)[1]
        seen.append(path)
        if path in files:
            return httpx.Response(200, content=files[path])
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return BinanceVisionArchive(client=client, sleep=lambda _s: None), seen


def test_archive_fetch_verifies_and_parses():
    name = "BTCUSDT-1m-2025-01.zip"
    z = _zip("BTCUSDT-1m-2025-01.csv", HEADER + ROWS)
    base = f"monthly/klines/BTCUSDT/1m/{name}"
    arch, _ = _archive(
        {base: z, base + ".CHECKSUM": f"{hashlib.sha256(z).hexdigest()}  {name}".encode()}
    )
    bf = arch.monthly(Dataset.KLINES, "BTCUSDT", "1m", YearMonth(2025, 1))
    assert bf is not None and bf.name == name and len(bf.frame) == 2


def test_archive_rejects_corrupt_file():
    name = "BTCUSDT-1m-2025-01.zip"
    z = _zip("BTCUSDT-1m-2025-01.csv", HEADER + ROWS)
    base = f"monthly/klines/BTCUSDT/1m/{name}"
    arch, _ = _archive(
        {
            base: z[:-3] + b"xyz",
            base + ".CHECKSUM": f"{hashlib.sha256(z).hexdigest()}  {name}".encode(),
        }
    )
    with pytest.raises(ChecksumMismatchError):
        arch.monthly(Dataset.KLINES, "BTCUSDT", "1m", YearMonth(2025, 1))


def test_archive_missing_returns_none_without_downloading_zip():
    arch, seen = _archive({})
    assert arch.monthly(Dataset.MARK_PRICE_KLINES, "BTCUSDT", "1m", YearMonth(2025, 1)) is None
    assert seen == ["monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip.CHECKSUM"]


def test_archive_retries_5xx():
    name = "BTCUSDT-1m-2025-01-02.zip"
    z = _zip("x.csv", ROWS)
    base = f"daily/klines/BTCUSDT/1m/{name}"
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path.split("/data/futures/um/", 1)[1]
        if path == base:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, content=z)
        if path == base + ".CHECKSUM":
            return httpx.Response(200, content=f"{hashlib.sha256(z).hexdigest()}  {name}".encode())
        return httpx.Response(404)

    arch = BinanceVisionArchive(
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None
    )
    bf = arch.daily(Dataset.KLINES, "BTCUSDT", "1m", date(2025, 1, 2))
    assert bf is not None and calls["n"] == 3
