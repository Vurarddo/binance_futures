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

