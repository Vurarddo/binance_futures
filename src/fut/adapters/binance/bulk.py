"""data.binance.vision bulk archive adapter (USDⓈ-M futures)."""

from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from collections.abc import Callable
from datetime import date

import httpx
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.csv as pacsv

from fut.domain.entities import Dataset, Source
from fut.domain.klines import KlineFrame
from fut.domain.time import YearMonth, normalize_epoch_ms
from fut.ports.data import BulkFile, daily_file_name, monthly_file_name

log = logging.getLogger(__name__)

DEFAULT_BASE = "https://data.binance.vision/data/futures/um"

CSV_COLUMNS = [
    "open_time",
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
    "ignore",
]
_INT_COLS = {"open_time", "close_time", "trades"}


class ChecksumMismatchError(RuntimeError):
    pass


def verify_checksum(data: bytes, checksum_text: str, expected_name: str) -> str:
    """`.CHECKSUM` files are `<sha256>  <filename>`. Returns the verified digest."""
    parts = checksum_text.split()
    if len(parts) < 2 or len(parts[0]) != 64:
        raise ChecksumMismatchError(f"malformed CHECKSUM for {expected_name}: {checksum_text!r}")
    expected, name = parts[0].lower(), parts[1]
    if name != expected_name:
        raise ChecksumMismatchError(f"CHECKSUM names {name!r}, expected {expected_name!r}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumMismatchError(f"{expected_name}: sha256 {actual} != published {expected}")
    return actual


def parse_kline_csv(csv_bytes: bytes, source: Source) -> KlineFrame:
    """Parse a kline CSV. Older archives have no header row, newer ones do."""
    has_header = not csv_bytes[:1].isdigit()
    table = pacsv.read_csv(
        io.BytesIO(csv_bytes),
        read_options=pacsv.ReadOptions(column_names=CSV_COLUMNS, skip_rows=1 if has_header else 0),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.int64() if c in _INT_COLS else pa.float64() for c in CSV_COLUMNS}
        ),
    )
    n = table.num_rows
    cols: dict[str, npt.NDArray[np.generic]] = {}
    for c in CSV_COLUMNS[:-1]:
        arr = table.column(c).to_numpy()
        if c in ("open_time", "close_time"):
            arr = normalize_epoch_ms(arr.astype(np.int64))
        cols[c] = arr
    cols["source"] = np.full(n, int(source), dtype=np.int8)
    return KlineFrame.from_columns(cols)


def read_zip_csv(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one CSV in archive, got {zf.namelist()}")
        return zf.read(names[0])


class BinanceVisionArchive:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        *,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True
        )
        self.max_retries = max_retries
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def monthly(
        self, dataset: Dataset, symbol: str, interval: str, month: YearMonth
    ) -> BulkFile | None:
        name = monthly_file_name(symbol, interval, month)
        return self._fetch(
            f"monthly/{dataset.value}/{symbol}/{interval}/{name}", name, Source.MONTHLY
        )

    def daily(self, dataset: Dataset, symbol: str, interval: str, day: date) -> BulkFile | None:
        name = daily_file_name(symbol, interval, day)
        return self._fetch(f"daily/{dataset.value}/{symbol}/{interval}/{name}", name, Source.DAILY)

    def _get(self, path: str) -> bytes | None:
        url = f"{self.base_url}/{path}"
        for attempt in range(1, self.max_retries + 2):
            try:
                r = self._client.get(url)
            except httpx.TransportError as e:
                if attempt > self.max_retries:
                    raise
                self._backoff(attempt, f"{type(e).__name__} {path}")
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.content
            if r.status_code >= 500 or r.status_code == 429:
                if attempt > self.max_retries:
                    r.raise_for_status()
                self._backoff(attempt, f"HTTP {r.status_code} {path}")
                continue
            r.raise_for_status()
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, why: str) -> None:
        delay = min(60.0, 1.0 * 2 ** (attempt - 1))
        log.warning("%s; retry %d in %.0fs", why, attempt, delay)
        self._sleep(delay)

    def _fetch(self, path: str, name: str, source: Source) -> BulkFile | None:
        checksum = self._get(path + ".CHECKSUM")
        if checksum is None:
            return None
        data = self._get(path)
        if data is None:
            # CHECKSUM published but archive missing: treat as not available yet.
            log.warning("%s: CHECKSUM exists but archive is missing", path)
            return None
        digest = verify_checksum(data, checksum.decode().strip(), name)
        frame = parse_kline_csv(read_zip_csv(data), source)
        return BulkFile(name=name, sha256=digest, frame=frame)
