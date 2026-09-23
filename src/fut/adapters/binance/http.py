"""Thin signed/unsigned HTTP client for Binance USDⓈ-M futures REST.

- weight-aware client-side rate limiting (+ server header feedback)
- retries only for idempotent GETs (transport errors, 5xx, 429 with Retry-After)
- secrets are held as SecretStr and never appear in logs, reprs or exceptions
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from fut.adapters.binance.rate_limit import WeightLimiter

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {500, 502, 503, 504}


class BinanceApiError(RuntimeError):
    def __init__(self, status: int, code: int | None, msg: str, path: str) -> None:
        super().__init__(f"HTTP {status} code={code} {msg!r} on {path}")
        self.status = status
        self.code = code
        self.msg = msg
        self.path = path


class IpBannedError(BinanceApiError):
    """HTTP 418: the IP is auto-banned. Never retry automatically."""


def sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinanceHttp:
    def __init__(
        self,
        base_url: str,
        *,
        limiter: WeightLimiter,
        api_key: SecretStr | None = None,
        api_secret: SecretStr | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        recv_window_ms: int = 5000,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self.max_retries = max_retries
        self.recv_window_ms = recv_window_ms
        self._sleep = sleep
        self._wall_clock_ms = wall_clock_ms
        self.time_offset_ms = 0

    def __repr__(self) -> str:
        return f"BinanceHttp({self.base_url!r}, signed={self.can_sign})"

    @property
    def can_sign(self) -> bool:
        return bool(self._api_key and self._api_secret)

    def close(self) -> None:
        self._client.close()

    def sync_time(self) -> int:
        """Measure server-local clock offset (ms). Used for signed timestamps."""
        t0 = self._wall_clock_ms()
        data = self.get("/fapi/v1/time", weight=1)
        t1 = self._wall_clock_ms()
        self.time_offset_ms = int(data["serverTime"]) - (t0 + t1) // 2
        return self.time_offset_ms

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        weight: int,
        signed: bool = False,
        extra_limiter: WeightLimiter | None = None,
    ) -> Any:
        return self._request(
            "GET", path, params or {}, weight, signed, extra_limiter, idempotent=True
        )

    def _prepare(self, params: dict[str, Any], signed: bool) -> tuple[str, dict[str, str]]:
        headers: dict[str, str] = {}
        p = {k: v for k, v in params.items() if v is not None}
        if signed:
            if not (self._api_key and self._api_secret):
                raise RuntimeError("signed request without API credentials configured")
            p["recvWindow"] = self.recv_window_ms
            p["timestamp"] = self._wall_clock_ms() + self.time_offset_ms
            q = urlencode(p)
            q += "&signature=" + sign(q, self._api_secret.get_secret_value())
            headers["X-MBX-APIKEY"] = self._api_key.get_secret_value()
            return q, headers
        return urlencode(p), headers

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        weight: int,
        signed: bool,
        extra_limiter: WeightLimiter | None,
        *,
        idempotent: bool,
    ) -> Any:
        attempt = 0
        while True:
            attempt += 1
            self.limiter.acquire(weight)
            if extra_limiter is not None:
                extra_limiter.acquire(1)
            query, headers = self._prepare(params, signed)  # fresh timestamp per attempt
            url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
            try:
                resp = self._client.request(method, url, headers=headers)
            except httpx.TransportError as e:
                if idempotent and attempt <= self.max_retries:
                    self._backoff(attempt, f"{type(e).__name__} on {path}")
                    continue
                raise
            used = resp.headers.get("x-mbx-used-weight-1m")
            if used is not None and used.isdigit():
                self.limiter.observe_server_used(int(used))

            if resp.status_code == 200:
                return resp.json()
            code, msg = _error_body(resp)
            if resp.status_code == 418:
                raise IpBannedError(418, code, msg, path)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", "60"))
                self.limiter.penalize(retry_after)
                if idempotent and attempt <= self.max_retries:
                    log.warning("429 on %s, backing off %.0fs", path, retry_after)
                    continue
            elif (
                resp.status_code in RETRYABLE_STATUS and idempotent and attempt <= self.max_retries
            ):
                self._backoff(attempt, f"HTTP {resp.status_code} on {path}")
                continue
            raise BinanceApiError(resp.status_code, code, msg, path)

    def _backoff(self, attempt: int, why: str) -> None:
        delay = min(60.0, 0.5 * 2 ** (attempt - 1))
        log.warning("%s; retry %d in %.1fs", why, attempt, delay)
        self._sleep(delay)


def _error_body(resp: httpx.Response) -> tuple[int | None, str]:
    try:
        body = resp.json()
    except ValueError:
        return None, resp.text[:200]
    if isinstance(body, dict):
        code = body.get("code")
        return (
            int(code) if isinstance(code, int | str) and str(code).lstrip("-").isdigit() else None,
            str(body.get("msg", body))[:200],
        )
    return None, str(body)[:200]
