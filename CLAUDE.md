# fut — research-first trading system for Binance USDⓈ-M perpetual futures

Collect data → backtest with realistic costs → judge with honest statistics → only then paper-trade
survivors on the **testnet**. Real-money trading is out of scope until the owner explicitly asks.

Communicate with the owner in **Ukrainian**; code, identifiers, commits and docs in English.

## Commands

```bash
uv sync                                   # create .venv, install deps (Python 3.12)
uv run pre-commit install                 # once per clone: secret scan, ruff, mypy, fast tests
uv run pytest -q                          # tests (network tests are marked `network`)
uv run ruff format . && uv run ruff check .
uv run mypy                               # --strict on src/

uv run fut backfill [--symbols ..] [--start YYYY-MM-DD] [--datasets klines markPriceKlines]
                    [--intervals 1m] [--no-funding] [--no-topup] [--workers N]
uv run fut coverage                       # what is stored, per series + funding
uv run fut quality [--symbols ..] [--no-recheck]   # data-quality report (+ JSON in data/reports/)
uv run fut specs                          # exchangeInfo snapshot (+ leverage brackets if keyed)
```

`fut backfill` is resumable and idempotent: re-running fetches only what is missing. Run it
daily to extend history; the monthly archive replaces daily/REST copies once published.

## Architecture (hexagonal)

```
src/fut/
  domain/      pure: time (UTC ms), entities (SymbolSpec, FundingRate, LeverageBracket),
               KlineFrame (numpy columns) + merge_klines, quality checks. No I/O.
  ports/       Protocols: BulkArchive, MarketData, AccountData, HistoricalStore, Clock.
  adapters/    binance/http.py   thin signed client, weight limiter, GET-only retries
               binance/rest.py   MarketData/AccountData over REST
               binance/bulk.py   data.binance.vision archives (+ .CHECKSUM verification)
               storage/parquet_store.py  hive-partitioned Parquet, queried with DuckDB
  use_cases/   backfill.py, inspect.py (coverage, quality)
  config.py    pydantic-settings (FUT_ prefix, .env) with safety guards
  cli.py       `fut` entry point
```

Rules:
- Raw Binance payloads never leave `adapters/`; they are parsed into domain types in `parse.py`.
- One strategy code path for backtest and live (Phase 2+): strategies emit order intents; only
  the execution adapter differs. Ports `MarketDataFeed`, `ExecutionVenue`, `AccountState` are
  added when their phase starts (not before).
- Decimal at the exchange boundary (specs, funding, orders); float64 inside vectorised research.

### Storage layout (`data/`, gitignored)

```
data/klines/dataset={klines|markPriceKlines}/symbol=X/interval=1m/month=YYYY-MM/data.parquet
data/klines/.../month=YYYY-MM/_manifest.json   # ingested sources (name, sha256, rows, revisions)
data/funding/symbol=X/data.parquet             # funding_time (actual ms), funding_rate, mark_price
data/snapshots/exchange_info/YYYY-MM-DD.json   # raw exchangeInfo per day
data/reports/quality-YYYY-MM-DD.json
```

Kline columns: open_time, open, high, low, close, volume, close_time, quote_volume, trades,
taker_buy_volume, taker_buy_quote_volume, source (1=REST, 2=DAILY, 3=MONTHLY).
Merge rule: per open_time the higher source wins; on a tie the newer copy wins. Any value
difference between two copies of a bar is counted as a **revision** in the manifest and shown
by `fut quality` (history must not change after the fact).

Ad-hoc research: `ParquetStore(Path("data")).connect()` gives DuckDB views `klines`, `funding`.

## Verified exchange facts

All checked against the live API / archive on the date shown. Re-verify before relying on a
fact that is older than ~3 months or when behaviour looks off. Docs pages on
developers.binance.com render client-side and cannot be fetched by tools; weights below were
**measured** from the `X-MBX-USED-WEIGHT-1M` response header.

**2026-09-23 — mainnet `https://fapi.binance.com` (public endpoints)**
- `exchangeInfo.rateLimits`: REQUEST_WEIGHT 2400/min; ORDERS 1200/min and 300/10s. timezone UTC.
  907 symbols listed.
- Weights: `/fapi/v1/time` 1; `/fapi/v1/exchangeInfo` 1; `/fapi/v1/klines` and
  `/fapi/v1/markPriceKlines`: limit 1–99 → 1, 100–499 → 2, 500–1000 → 5, 1001–1500 → 10.
  Max limit 1500 (1501 → error -1130). We page with limit=499 (best rows per weight).
- `/fapi/v1/fundingRate` and `/fapi/v1/fundingInfo` do **not** return the weight header (they
  have a separate limit; docs say 500 req / 5 min / IP — not measured). We cap our own use at
  250 / 5 min. `fundingRate` limit max is 1000 (1001 → error); `startTime`/`endTime` are
  both **inclusive**. Response fields: symbol,
  fundingTime, fundingRate, markPrice (**empty string** in old records, e.g. 2020), rateType.
- `fundingTime` is not always on the hour: e.g. `1790150400004` (+4 ms). Stored as reported.
- Funding interval history since 2021-01-01: all universe symbols 8h, **except SOLUSDT
  2022-11-09 20:00 → 4h, 2022-11-10 06:00 → 2h, back to 8h 2022-11-18 16:00** (FTX crash;
  rate hit the 2% cap). So funding must be applied at stored timestamps, never on a fixed grid.
- `fundingInfo` (2026-09-23): all 6 universe symbols have `fundingIntervalHours=8`. Caps/floors:
  BTC/ETH ±0.300%, SOL/BNB/XRP ±0.375%, DOGE ±0.4875%. Intervals can change over time —
  the quality report derives intervals from actual settlement timestamps.
- Kline REST rows: 12 fields `[openTime, o, h, l, c, v, closeTime, quoteVol, trades,
  takerBuyBase, takerBuyQuote, ignore]`, numbers as strings, times in ms. The last row can be
  the still-open bar → we only store bars with open_time ≤ last closed bar.
- Mark-price klines: volume/trade fields are 0; `trades` column holds a count of ~60.
- `leverageBracket` and `commissionRate` are signed (USER_DATA); unsigned calls return -2014.
  **Not yet verified** (no key configured).
- Universe specs (mainnet, 2026-09-23):

  | symbol | tick | step | minNotional | maintMargin% (tier1) | liquidationFee | onboard (UTC) |
  |---|---|---|---|---|---|---|
  | BTCUSDT | 0.10 | 0.001 | 50 | 2.5 | 0.0125 | 2019-09-08 |
  | ETHUSDT | 0.01 | 0.001 | 20 | 2.5 | 0.0125 | 2019-11-27 |
  | SOLUSDT | 0.0100 | 0.01 | 5 | 2.5 | 0.015 | 2020-09-14 |
  | BNBUSDT | 0.010 | 0.01 | 5 | 2.5 | 0.0125 | 2020-02-10 |
  | XRPUSDT | 0.0001 | 0.1 | 5 | 2.5 | 0.0125 | 2020-01-06 |
  | DOGEUSDT | 0.000010 | 1 | 5 | 2.5 | 0.015 | 2020-07-10 |

  Order types: LIMIT, MARKET, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET,
  TRAILING_STOP_MARKET. TIF: GTC, IOC, FOK, GTX (post-only), GTD. PERCENT_PRICE ±5%
  (BTC), marketTakeBound 0.05.

**2026-09-23 — data.binance.vision (USDⓈ-M, `data/futures/um/`)**
- Paths: `{monthly|daily}/{klines|markPriceKlines}/SYM/1m/SYM-1m-YYYY-MM[-DD].zip` and a
  sibling `.zip.CHECKSUM` = `<sha256>  <filename>`. Verified sha256 matches for a sample.
- Old CSVs (e.g. 2020-01, 2022-01) have **no header**; newer ones (2023-06+, 2025, 2026) have
  a header row. Timestamps are **ms** (the spot µs switch does not apply to futures as of
  2026-09-23; the parser guards anyway).
- Daily file for D is available on D+1 (2026-09-22 existed on 2026-09-23, 2026-09-23 did
  not). The monthly archive for the previous month existed by the 23rd; exact publish lag
  not measured.
- Archive 1m bars equal REST 1m bars: 0 differences on 3 sampled days × 12 series
  (51,840 bars) plus a live test on a recent BTCUSDT day.
- **Monthly archives can silently omit whole days** that the daily archives and REST still
  have (e.g. klines SOLUSDT/XRPUSDT 2022-02-26..28 and 2022-04-01..02; markPriceKlines for
  all six symbols on several days in 2021-07, 2022-07/10, 2023-02, 2026-06). The checksum still
  matches — the file is complete as published, just incomplete as data. `fut backfill` repairs
  such days (daily archive first, then REST) and records each attempt in the manifest.
- Genuine gaps (absent from archive *and* REST): markPriceKlines 2022-07-12 ~12:57–13:21 and
  2024-08-12 10:02–10:03 (1–7 bars per symbol). Klines have none since 2021-01-01.
- Klines contain **zero-volume bars in exchange-wide maintenance windows** (same dates across
  all symbols, e.g. 2021-03-02 59 min, 2024-10-29 74 min; BTCUSDT 2023-11-10 99 min). The bars
  exist but nothing traded — the backtester must treat them as non-tradable.
- Also available (not used yet): `monthly/fundingRate/SYM/SYM-fundingRate-YYYY-MM.zip` with
  columns calc_time, funding_interval_hours, last_funding_rate.
- Listing is possible via the S3 API
  (`https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=...&delimiter=/`);
  we don't need it — missing files are detected by a 404 on `.CHECKSUM`.

**2026-09-23 — testnet (`testnet.binancefuture.com`, `demo-fapi.binance.com`)**
- Both hosts answer `/fapi/v1/time` and `exchangeInfo` with identical content (741 symbols);
  REQUEST_WEIGHT limit there is **6000/min** (mainnet 2400).
- Testnet specs differ from mainnet: BTCUSDT step 0.0001 (mainnet 0.001), liquidationFee
  0.02 for all universe symbols (mainnet 0.0125–0.015). → Paper-trading parity must use the
  testnet's own filters; backtests must use mainnet's.

**Signing**: HMAC-SHA256 over the exact query string, hex digest, `X-MBX-APIKEY` header;
verified against the published example in binance-spot-api-docs (same scheme for futures).

## Backtest conventions (binding from Phase 2)

- Signals on **closed** bars only; fills no earlier than the next bar's open. No negative
  shifts, centred windows or full-sample normalisation. Warm-up bars never trade.
- Intrabar TP+SL ambiguity: resolve with 1m sub-bars when available, else worst case (SL
  first); report the count of ambiguous trades.
- Costs: maker/taker by tier (fetch VIP0 via `commissionRate`, never hard-code), slippage
  (fixed bps + fraction of bar range, larger for stops), funding at real settlement times.
- Isolated margin; liquidation from leverage brackets (mark price), with liquidation fee.
- The most recent **6 months are a hold-out**, opened once, by the owner's decision.
- Count every variant tried; report Deflated Sharpe; verdict NO_EDGE | INCONCLUSIVE | CANDIDATE.
- Martingale / grid-recovery staking is not a strategy — only as a simulated comparison.

## Safety rules

- Testnet is the default for anything signed. Mainnet requires `FUT_TRADING_ENV=mainnet`,
  `FUT_ENABLE_MAINNET=true` and a mainnet `FUT_TRADING_URL`; `Settings` refuses any
  env/host mismatch (testnet env + mainnet host, or vice versa).
- Research market data uses public mainnet endpoints, unsigned, GET only.
- API keys must never have withdrawal permission. Secrets are `SecretStr`, never logged,
  printed or committed; `.env` is gitignored; the pre-commit secret scan blocks key-like
  strings (mark a deliberate public fixture with `# secret-scan: allow`).
- Retries only for idempotent requests (GET). HTTP 418 (IP ban) is never retried.
- Ask the owner before adding any dependency that talks to the network.

## Research findings log

Detailed entries live in `research/JOURNAL.md` (including failures). Summary:

- 2026-09-23: Phase 1 data collected (6 symbols, 2021-01-01 → now, 1m klines + mark price,
  funding). Klines 100% complete after repair; see JOURNAL for numbers. No strategy research yet.
