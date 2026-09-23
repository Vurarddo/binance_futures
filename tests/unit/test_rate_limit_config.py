import pytest
from pydantic import ValidationError

from fut.adapters.binance.rate_limit import WeightLimiter
from fut.config import Settings


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s


def test_limiter_blocks_until_window_frees():
    t = FakeTime()
    lim = WeightLimiter(10, 60.0, safety=1.0, clock=t.clock, sleep=t.sleep)
    for _ in range(5):
        assert lim.acquire(2) == 0
    slept = lim.acquire(2)
    assert 60.0 <= slept < 61.0
    assert lim.used() == 2 + 0  # the first five expired


def test_limiter_rejects_weight_over_budget():
    with pytest.raises(ValueError, match="exceeds budget"):
        WeightLimiter(10, 60.0, safety=0.5).acquire(6)


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_default_is_testnet():
    s = _settings()
    assert s.trading_env == "testnet" and "testnet" in s.trading_url


def test_mainnet_requires_explicit_flag():
    with pytest.raises(ValidationError, match="ENABLE_MAINNET"):
        _settings(trading_env="mainnet", trading_url="https://fapi.binance.com")


def test_testnet_env_with_mainnet_url_refused():
    with pytest.raises(ValidationError, match="not a known testnet host"):
        _settings(trading_env="testnet", trading_url="https://fapi.binance.com")


def test_mainnet_env_with_testnet_url_refused():
    with pytest.raises(ValidationError, match="not mainnet"):
        _settings(
            trading_env="mainnet",
            enable_mainnet=True,
            trading_url="https://testnet.binancefuture.com",
        )


def test_half_credentials_refused():
    with pytest.raises(ValidationError, match="both"):
        _settings(api_key="abc")


def test_secrets_not_in_repr():
    s = _settings(api_key="SUPERKEY", api_secret="SUPERSECRET")  # secret-scan: allow
    assert "SUPERKEY" not in repr(s) and "SUPERSECRET" not in repr(s)
