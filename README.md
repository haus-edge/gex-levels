# GEX Levels

Daily gamma exposure (GEX) key levels for ES and NQ futures, computed from SPY/QQQ options chains.

## Levels

| Level | Description |
|-------|-------------|
| **Gamma Flip** | Price where net dealer gamma crosses zero (regime boundary) |
| **Call Wall** | Strike with highest call gamma concentration (resistance) |
| **Put Wall** | Strike with highest put gamma concentration (support) |

Levels are pre-converted to index price space (SPX/NDX) so they map directly to ES/NQ futures.

## Data

Updated automatically at 6:30 AM ET on trading days via GitHub Actions.

- [`data/gex_SPX.txt`](data/gex_SPX.txt) — ES / S&P 500
- [`data/gex_QQQ.txt`](data/gex_QQQ.txt) — NQ / Nasdaq-100

Raw URLs for programmatic access:
```
https://raw.githubusercontent.com/haus-edge/gex-levels/main/data/gex_SPX.txt
https://raw.githubusercontent.com/haus-edge/gex-levels/main/data/gex_QQQ.txt
```

## How It Works

1. Downloads full options chains from Yahoo Finance (no API key needed)
2. Computes per-strike gamma using Black-Scholes with DTE-weighted open interest
3. Finds gamma flip via skew-corrected sweep (sticky-delta blend)
4. Applies 10% hysteresis to call/put walls to prevent day-to-day flip-flop
5. Converts from ETF (SPY/QQQ) to index (SPX/NDX) price space

## Sierra Chart Integration

The [AMT GEX Levels](https://github.com/haus-edge/amt-studies) ACSIL study fetches these files automatically and draws the levels on your ES/NQ charts. No Python install required.
