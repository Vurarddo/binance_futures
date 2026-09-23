import httpx
import pytest
from pydantic import SecretStr

from fut.adapters.binance.http import BinanceApiError, BinanceHttp, IpBannedError, sign
from fut.adapters.binance.rate_limit import WeightLimiter
from fut.adapters.binance.rest import BinanceRest, kline_weight
from fut.domain.entities import Dataset
from fut.domain.time import MINUTE_MS

# Public example secret from the Binance API docs.
SECRET = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"  # secret-scan: allow


def test_sign_matches_binance_docs_example():
    # HMAC example from binance-spot-api-docs/rest-api.md (same scheme as USDⓈ-M futures),
    # fetched 2026-09-23.
    q = (
        "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
        "&recvWindow=5000&timestamp=1499827319559"
    )
    assert sign(q, SECRET) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_kline_weight_table():
    # Measured via X-MBX-USED-WEIGHT-1M on 2026-09-23.
    assert [kline_weight(n) for n in (99, 100, 499, 500, 1000, 1001, 1500)] == [
        1,
        2,
        2,
        5,
        5,
        10,
        10,
    ]


def _http(handler, **kw) -> BinanceHttp:
    return BinanceHttp(
        "https://fapi.binance.com",
        limiter=WeightLimiter(2400, 60.0),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _s: None,
        **kw,
    )


def test_get_retries_5xx_then_succeeds():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        return httpx.Response(502) if n["c"] < 3 else httpx.Response(200, json={"ok": 1})

    assert _http(h).get("/x", weight=1) == {"ok": 1}
    assert n["c"] == 3


def test_client_error_not_retried():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        return httpx.Response(400, json={"code": -1130, "msg": "bad limit"})

    with pytest.raises(BinanceApiError) as e:
        _http(h).get("/x", weight=1)
    assert e.value.code == -1130 and n["c"] == 1


def test_418_raises_immediately():
    with pytest.raises(IpBannedError):
        _http(lambda r: httpx.Response(418, json={"code": -1003, "msg": "banned"})).get(
            "/x", weight=1
        )


def test_429_penalizes_limiter_and_retries():
    n = {"c": 0}

    def h(req):
        n["c"] += 1
        if n["c"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "3"}, json={"code": -1003, "msg": "x"}
            )
        return httpx.Response(200, json=[])

    t = {"now": 0.0}
    lim = WeightLimiter(
        100, 60.0, clock=lambda: t["now"], sleep=lambda s: t.__setitem__("now", t["now"] + s)
    )
    http = BinanceHttp(
        "https://fapi.binance.com",
        limiter=lim,
        client=httpx.Client(transport=httpx.MockTransport(h)),
        sleep=lambda _s: None,
    )
    assert http.get("/x", weight=1) == []
    assert 3.0 <= t["now"] < 4.0  # waited out Retry-After before the second call


def test_signed_request_headers_and_no_secret_leak():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-mbx-apikey")
        return httpx.Response(200, json=[])

    http = _http(
        h,
        api_key=SecretStr("KEY123"),
        api_secret=SecretStr(SECRET),
        wall_clock_ms=lambda: 1591702613943,
    )
    http.get("/fapi/v1/leverageBracket", weight=1, signed=True)
    assert seen["key"] == "KEY123"
    assert "timestamp=1591702613943" in seen["url"] and "signature=" in seen["url"]
    assert SECRET not in seen["url"]
    assert "KEY123" not in repr(http) and SECRET not in repr(http)


def test_signed_without_credentials_refused():
    with pytest.raises(RuntimeError, match="credentials"):
        _http(lambda r: httpx.Response(200, json=[])).get("/x", weight=1, signed=True)


def test_server_weight_header_is_observed():
    lim = WeightLimiter(2400, 60.0)
    http = BinanceHttp(
        "https://fapi.binance.com",
        limiter=lim,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, headers={"x-mbx-used-weight-1m": "700"}, json={})
            )
        ),
    )
    http.get("/x", weight=1)
    assert lim.used() == 700


def _kline_row(t):
    return [t, "1", "2", "0.5", "1.5", "10", t + MINUTE_MS - 1, "15", 3, "5", "7.5", "0"]


def test_rest_klines_paginates_and_clips():
    start = 1_700_000_000_000 // MINUTE_MS * MINUTE_MS
    end = start + 1200 * MINUTE_MS
    calls = []

    def h(req):
        p = dict(req.url.params)
        calls.append(p)
        s, e, lim = int(p["startTime"]), int(p["endTime"]), int(p["limit"])
        rows = [_kline_row(t) for t in range(s, e + 1, MINUTE_MS)][:lim]
        return httpx.Response(200, json=rows)

    rest = BinanceRest(_http(h))
    f = rest.klines(Dataset.KLINES, "BTCUSDT", "1m", start, end)
    assert len(f) == 1201
    assert f.open_time[0] == start and f.open_time[-1] == end
    assert len(calls) == 3 and all(c["limit"] == "499" for c in calls)


def test_rest_funding_paginates():
    rows = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": i * 8 * 3600_000 + (4 if i % 3 == 0 else 0),
            "fundingRate": "0.0001",
            "markPrice": "" if i < 5 else "100.5",
            "rateType": "Regular",
        }
        for i in range(1, 2501)
    ]

    def h(req):
        p = dict(req.url.params)
        s, e = int(p["startTime"]), int(p["endTime"])
        sel = [r for r in rows if s <= r["fundingTime"] <= e][: int(p["limit"])]
        return httpx.Response(200, json=sel)

    rest = BinanceRest(_http(h))
    out = rest.funding_rates("BTCUSDT", 0, 10**14)
    assert len(out) == 2500
    assert out[0].mark_price is None and out[-1].mark_price is not None
    assert out[2].funding_time_ms % 1000 == 4  # jitter preserved
