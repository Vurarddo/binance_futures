"""Settings (pydantic-settings). Environment variables use the FUT_ prefix; `.env` is read.

Safety by construction:
- research market data comes from the public mainnet endpoint (read-only, unsigned);
- anything signed/trading goes to `trading_env`, which defaults to testnet;
- mainnet trading needs `FUT_ENABLE_MAINNET=true` *and* a mainnet trading URL, and the pair
  is cross-checked so a testnet key can never be pointed at mainnet by accident (and v.v.).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fut.domain.entities import Dataset

DEFAULT_UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT")

MAINNET_FAPI_HOSTS = frozenset({"fapi.binance.com"})
TESTNET_FAPI_HOSTS = frozenset({"testnet.binancefuture.com", "demo-fapi.binance.com"})


class UnsafeConfigError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FUT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    history_start: date = date(2021, 1, 1)
    intervals: tuple[str, ...] = ("1m",)
    datasets: tuple[Dataset, ...] = (Dataset.KLINES, Dataset.MARK_PRICE_KLINES)
    download_workers: int = Field(default=4, ge=1, le=16)

    # Public market data (unsigned GETs only).
    market_data_url: str = "https://fapi.binance.com"
    bulk_data_url: str = "https://data.binance.vision/data/futures/um"

    # Signed / trading endpoint.
    trading_env: Literal["testnet", "mainnet"] = "testnet"
    trading_url: str = "https://testnet.binancefuture.com"
    enable_mainnet: bool = False
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    @model_validator(mode="after")
    def _guard(self) -> Settings:
        host = urlparse(self.trading_url).hostname or ""
        if self.trading_env == "testnet" and host not in TESTNET_FAPI_HOSTS:
            raise UnsafeConfigError(
                f"trading_env=testnet but trading_url host {host!r} is not a known testnet host"
            )
        if self.trading_env == "mainnet":
            if not self.enable_mainnet:
                raise UnsafeConfigError("trading_env=mainnet requires FUT_ENABLE_MAINNET=true")
            if host not in MAINNET_FAPI_HOSTS:
                raise UnsafeConfigError(f"trading_env=mainnet but host {host!r} is not mainnet")
        md_host = urlparse(self.market_data_url).hostname or ""
        if md_host not in MAINNET_FAPI_HOSTS | TESTNET_FAPI_HOSTS:
            raise UnsafeConfigError(f"unknown market data host {md_host!r}")
        if bool(self.api_key) != bool(self.api_secret):
            raise UnsafeConfigError("set both FUT_API_KEY and FUT_API_SECRET, or neither")
        return self

    @property
    def has_credentials(self) -> bool:
        return self.api_key is not None and self.api_secret is not None
