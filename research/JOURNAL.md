# Research journal

Every hypothesis tested gets an entry — **including failures**, so dead ideas are not silently
retried. Variants are counted for multiple-testing correction (Deflated Sharpe).

Template:

```
## YYYY-MM-DD — <short name>
- Hypothesis:
- Spec: (path to JSON spec / commit)
- Data window: (train / test / hold-out untouched?)
- Variants counted: params × symbols × timeframes × rules = N (cumulative for family: M)
- Costs: fees (tier, source/date), slippage model, funding
- Results: net PnL, Sharpe (daily), DSR, max DD, trades, fees+funding share of gross, vs buy&hold
- Robustness: plateau, cost ×2, per-symbol, per-year, bootstrap CI
- Verdict: NO_EDGE | INCONCLUSIVE | CANDIDATE — reasons
```

---

## 2026-09-23 — Phase 1: data collection (no hypothesis tested)

- Universe: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT (all PERPETUAL, TRADING).
- Window: 2021-01-01 00:00 → 2026-09-23 ~11:00 UTC (~5.7 years); 1m klines and 1m mark-price
  klines from data.binance.vision (68 monthly + 22 daily archives per series, all sha256-verified),
  REST top-up of today's bars; funding history from REST.
- Coverage: klines 3,011,702 bars per symbol = 100.000% of expected. Mark price 99.9996–99.9999%
  (2–13 missing 1m bars per symbol, absent in REST as well). Funding: 6,275 settlements per
  symbol, 6,350 for SOLUSDT. Storage 1.6 GB Parquet (zstd).
- Quality: 0 duplicates, 0 misaligned timestamps, 0 OHLC violations, 0 non-positive prices.
  REST re-fetch of 3 days per series: 0 differing bars out of 4,320 each.
- Archive defect found: monthly archives omitted 5 whole days for SOLUSDT/XRPUSDT klines and
  3–9 days per symbol of mark price → repaired from daily archives / REST (~7.2k–13k bars/series).
- Zero-volume 1m bars: 272–367 per symbol (maintenance windows shared by all symbols).
- Largest 1m moves (|log return| over max(5%, 20×robust σ)): 7 (BTC) to 213 (DOGE) bars; they
  cluster on 2021-05-19 and 2025-10-10 (market-wide crashes) — real, not data errors.
- Funding: max |rate| BTC 0.249%, ETH 0.375%, XRP 0.323%, BNB 0.442%, DOGE 0.750%, SOL 2.000%
  (Nov 2022, when SOL's interval dropped to 2h for 9 days).
- Verdict: n/a (data phase). Data is fit for bar-based research at ≥1m.
