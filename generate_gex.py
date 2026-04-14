"""
generate_gex.py - Compute proxy GEX levels from Yahoo Finance options data

Downloads options chains for SPY and QQQ via yfinance, computes gamma
exposure per strike using Black-Scholes, and derives key levels:
  - Gamma Flip: spot price where net dealer GEX crosses zero
  - Call Wall:  strike with highest call gamma concentration (resistance)
  - Put Wall:   strike with highest put gamma concentration (support)

These are ETF-derived proxy levels, not true SPX/NDX index option GEX.
Levels are converted to index price space for ES/NQ chart overlay.

Improvements over naive GEX:
  - DTE-weighted OI (exp decay, tau=14d) — near-term gamma dominates
  - Skew-corrected gamma flip — IV shifts with hypothetical spot via
    empirical ATM skew slope (sticky-delta blend, alpha=0.5)
  - Hysteresis on walls — wall only moves if new candidate exceeds
    current wall's GEX by >10%, preventing day-to-day flip-flop

Writes key=value text files to data/ for the AMT GEX Levels ACSIL study
to fetch via raw.githubusercontent.com.

Usage:
    python generate_gex.py              # compute both SPY and QQQ
    python generate_gex.py SPY          # compute SPY only
    python generate_gex.py QQQ          # compute QQQ only

Dependencies: pip install yfinance numpy scipy
No API key needed.
"""

import os
import sys
from datetime import datetime, timezone

try:
    import numpy as np
    from scipy.stats import norm
    import yfinance as yf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: python -m pip install yfinance numpy scipy")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMBOLS = ["SPY", "QQQ"]
MAX_DTE = 90            # include up to 90 DTE (weighted down by decay)
DTE_TAU = 14.0          # exponential decay time constant (days)
RISK_FREE_RATE = 0.043
SKEW_ALPHA = 0.5        # blend: 0=sticky strike, 1=sticky delta
WALL_HYSTERESIS = 0.10  # 10% — wall only moves if new candidate is 10%+ stronger

# Map ETF -> index ticker (for pre-converting levels to futures price space)
INDEX_MAP = {
    "SPY": "^GSPC",   # SPX index — ES futures trade at this level
    "QQQ": "^NDX",    # Nasdaq-100 — NQ futures trade at this level
}

# Map ETF -> volatility index ticker (for Edge Levels expected move)
VIX_MAP = {
    "SPY": "^VIX",    # VIX for ES/SPX
    "QQQ": "^VXN",    # VXN for NQ/NDX
}


def bs_gamma(S, K, T, r, sigma):
    """Vectorized Black-Scholes gamma."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def collect_chain(ticker, spot, max_dte):
    """Download options chain with DTE-weighted OI.

    Returns (calls, puts, exp_count) where each array is Nx4:
    [strike, weighted_OI, T_years, implied_vol]
    OI is pre-multiplied by exp(-DTE/tau) so near-term options dominate.
    """
    now = datetime.now()
    calls_list = []
    puts_list = []
    exp_count = 0

    for exp_str in ticker.options:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if dte <= 0 or dte > max_dte:
            continue
        T = dte / 365.0
        dte_weight = np.exp(-dte / DTE_TAU)
        exp_count += 1

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            print(f"    Skip {exp_str}: {e}")
            continue

        # Filter: positive IV, OI, and strike; skip deep OTM (>30% from spot)
        for df, out_list in [(chain.calls, calls_list), (chain.puts, puts_list)]:
            mask = (
                (df["impliedVolatility"] > 0.001)
                & (df["openInterest"] > 0)
                & (df["strike"] > 0)
                & (df["strike"] > spot * 0.70)
                & (df["strike"] < spot * 1.30)
            )
            filtered = df[mask]
            for _, row in filtered.iterrows():
                out_list.append([
                    row["strike"],
                    row["openInterest"] * dte_weight,
                    T,
                    row["impliedVolatility"],
                ])

    calls = np.array(calls_list) if calls_list else np.empty((0, 4))
    puts = np.array(puts_list) if puts_list else np.empty((0, 4))
    return calls, puts, exp_count


def compute_per_strike_gex(arr, spot, sign=1.0):
    """Aggregate dollar gamma per 1% move by strike.

    Uses: sign * gamma * OI * 100 * S^2 * 0.01
    This is the standard institutional scaling for dollar gamma exposure.
    Calls get sign=+1 (dealers long gamma), puts get sign=-1 (dealers short gamma).
    """
    if len(arr) == 0:
        return {}
    K, OI, T, IV = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    gamma = bs_gamma(spot, K, T, RISK_FREE_RATE, IV)
    gex = sign * gamma * OI * 100 * spot * spot * 0.01
    result = {}
    for i in range(len(K)):
        result[K[i]] = result.get(K[i], 0.0) + gex[i]
    return result


def compute_skew_slope(calls, puts, spot):
    """Compute empirical ATM skew slope (dIV/dStrike) from near-ATM puts.

    Returns a negative number for typical equity index skew.
    Used as a linear approximation — real skew is nonlinear, so treat
    the resulting gamma flip as a smoothed proxy, not a precise level.
    """
    if len(puts) == 0:
        return 0.0

    K = puts[:, 0]
    IV = puts[:, 3]
    near_atm = (K > spot * 0.95) & (K < spot * 1.05)
    K_near = K[near_atm]
    IV_near = IV[near_atm]

    if len(K_near) < 3:
        return 0.0

    slope, _ = np.polyfit(K_near, IV_near, 1)
    return float(slope)


def find_gamma_flip(calls, puts, spot, skew_slope):
    """Find gamma flip with skew-corrected IV.

    Sweeps hypothetical spot levels and computes net GEX at each using
    S^2 * 0.01 scaling. The zero crossing nearest current spot is the flip.
    """
    if len(calls) == 0 and len(puts) == 0:
        return spot

    hyp = np.linspace(spot * 0.85, spot * 1.15, 300)
    net_gex = np.zeros(len(hyp))

    spot_shift = spot - hyp

    for arr, sign in [(calls, 1.0), (puts, -1.0)]:
        if len(arr) == 0:
            continue
        K = arr[:, 0]
        OI = arr[:, 1]
        T = arr[:, 2]
        IV = arr[:, 3]

        S = hyp[:, np.newaxis]
        IV_base = IV[np.newaxis, :]
        T_b = T[np.newaxis, :]
        OI_b = OI[np.newaxis, :]

        IV_adj = IV_base + SKEW_ALPHA * skew_slope * spot_shift[:, np.newaxis]
        IV_adj = np.clip(IV_adj, 0.01, 5.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_T = np.sqrt(T_b)
            d1 = (np.log(S / K[np.newaxis, :]) +
                  (RISK_FREE_RATE + IV_adj**2 / 2) * T_b) / (IV_adj * sqrt_T)
            gamma = norm.pdf(d1) / (S * IV_adj * sqrt_T)
            gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

        gex = gamma * OI_b * 100 * S * S * 0.01
        net_gex += sign * np.sum(gex, axis=1)

    sign_changes = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_changes) > 0:
        midpoints = (hyp[sign_changes] + hyp[sign_changes + 1]) / 2
        closest = sign_changes[np.argmin(np.abs(midpoints - spot))]
        i = closest
        denom = abs(net_gex[i]) + abs(net_gex[i + 1])
        if denom > 0:
            frac = abs(net_gex[i]) / denom
            return float(hyp[i] + frac * (hyp[i + 1] - hyp[i]))

    return float(spot)


def read_previous_etf_walls(symbol):
    """Read previous ETF-space walls from existing file for hysteresis.

    These are the raw ETF strikes before index conversion, stored as
    ETF_CALL_WALL / ETF_PUT_WALL in the output file.
    """
    out_symbol = {"SPY": "SPX"}.get(symbol, symbol)
    path = os.path.join(OUTPUT_DIR, f"gex_{out_symbol}.txt")
    prev = {"ETF_CALL_WALL": 0.0, "ETF_PUT_WALL": 0.0}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key in prev:
                    prev[key] = float(val)
    except (FileNotFoundError, ValueError):
        pass
    return prev["ETF_CALL_WALL"], prev["ETF_PUT_WALL"]


def apply_hysteresis(gex_map, new_wall, prev_wall):
    """Only move the wall if new candidate is >10% stronger than previous.

    Both new_wall and prev_wall must be in the same price space as gex_map
    (ETF strikes). Comparison uses absolute GEX magnitude.
    """
    if prev_wall == 0.0 or prev_wall not in gex_map:
        return new_wall

    new_strength = abs(gex_map[new_wall])
    prev_strength = abs(gex_map.get(prev_wall, 0.0))

    if prev_strength == 0:
        return new_wall

    if new_strength > prev_strength * (1.0 + WALL_HYSTERESIS):
        return new_wall
    else:
        return prev_wall


def compute_gex_levels(symbol, max_dte=MAX_DTE):
    """Full GEX computation for one symbol."""
    print(f"  Downloading {symbol} options chain...")
    ticker = yf.Ticker(symbol)

    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
    if not spot or spot <= 0:
        raise ValueError(f"Could not get price for {symbol}")
    print(f"  Spot: ${spot:.2f}")

    calls, puts, exp_count = collect_chain(ticker, spot, max_dte)
    print(f"  {exp_count} expirations, {len(calls)} calls, {len(puts)} puts")

    if len(calls) == 0 and len(puts) == 0:
        raise ValueError(f"No options data for {symbol}")

    # --- Per-strike GEX (ETF space) ---
    call_gex = compute_per_strike_gex(calls, spot, sign=+1.0)
    put_gex = compute_per_strike_gex(puts, spot, sign=-1.0)

    # Raw wall candidates (ETF strikes)
    # Call wall: strike with largest positive GEX
    raw_call_wall = max(call_gex, key=call_gex.get) if call_gex else 0.0
    # Put wall: strike with largest absolute negative GEX
    raw_put_wall = max(put_gex, key=lambda k: abs(put_gex[k])) if put_gex else 0.0

    # Hysteresis: compare in ETF space against previous ETF-space walls
    prev_cw, prev_pw = read_previous_etf_walls(symbol)
    call_wall = apply_hysteresis(call_gex, raw_call_wall, prev_cw)
    put_wall = apply_hysteresis(put_gex, raw_put_wall, prev_pw)

    if call_wall != raw_call_wall:
        print(f"  Call wall held at {prev_cw:.2f} (hysteresis — new candidate {raw_call_wall:.2f} not 10%+ stronger)")
    if put_wall != raw_put_wall:
        print(f"  Put wall held at {prev_pw:.2f} (hysteresis — new candidate {raw_put_wall:.2f} not 10%+ stronger)")

    # Net GEX (relative, not institutional-grade absolute)
    net_gex = sum(call_gex.values()) + sum(put_gex.values())
    regime = "positive_gamma" if net_gex >= 0 else "negative_gamma"

    # --- Skew-corrected gamma flip (ETF space) ---
    skew_slope = compute_skew_slope(calls, puts, spot)
    print(f"  ATM skew slope: {skew_slope:.6f} dIV/d$ (used for gamma flip correction)")
    print(f"  Computing gamma flip...")
    gamma_flip = find_gamma_flip(calls, puts, spot, skew_slope)

    # Save ETF-space walls for next run's hysteresis comparison
    etf_call_wall = float(call_wall)
    etf_put_wall = float(put_wall)
    etf_gamma_flip = float(gamma_flip)

    # --- Convert to index/futures price space ---
    out_symbol = {"SPY": "SPX"}.get(symbol, symbol)
    index_ticker = INDEX_MAP.get(symbol)
    if index_ticker:
        try:
            idx = yf.Ticker(index_ticker)
            index_price = idx.fast_info["lastPrice"]
            ratio = index_price / spot
            print(f"  Index {index_ticker}: {index_price:.2f} (ratio {ratio:.2f}x)")
            gamma_flip *= ratio
            call_wall *= ratio
            put_wall *= ratio
            spot = index_price
        except Exception as e:
            print(f"  Warning: could not fetch {index_ticker}, levels stay in ETF space: {e}")

    # --- Fetch volatility index close (VIX/VXN) for Edge Levels study ---
    vol_close = 0.0
    vol_ticker = VIX_MAP.get(symbol)
    if vol_ticker:
        try:
            vt = yf.Ticker(vol_ticker)
            vol_close = vt.fast_info["previousClose"]
            print(f"  {vol_ticker} previous close: {vol_close:.2f}")
        except Exception as e:
            print(f"  Warning: could not fetch {vol_ticker}: {e}")

    return {
        "symbol": out_symbol,
        "underlying": float(spot),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regime": regime,
        "gamma_flip": float(gamma_flip),
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "net_gex": float(net_gex),
        # ETF-space values for hysteresis on next run
        "etf_gamma_flip": etf_gamma_flip,
        "etf_call_wall": etf_call_wall,
        "etf_put_wall": etf_put_wall,
        # Volatility index close for Edge Levels study
        "vol_close": float(vol_close),
        "vol_ticker": vol_ticker or "",
    }


def write_gex_file(data):
    """Write GEX data to key=value text file for ACSIL."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"gex_{data['symbol']}.txt")
    with open(path, "w") as f:
        f.write(f"SYMBOL={data['symbol']}\n")
        f.write(f"UNDERLYING={data['underlying']:.2f}\n")
        f.write(f"TIMESTAMP={data['timestamp']}\n")
        f.write(f"REGIME={data['regime']}\n")
        f.write(f"GAMMA_FLIP={data['gamma_flip']:.2f}\n")
        f.write(f"CALL_WALL={data['call_wall']:.2f}\n")
        f.write(f"PUT_WALL={data['put_wall']:.2f}\n")
        f.write(f"NET_GEX={data['net_gex']:.0f}\n")
        # ETF-space walls for hysteresis stability across runs
        f.write(f"ETF_GAMMA_FLIP={data['etf_gamma_flip']:.2f}\n")
        f.write(f"ETF_CALL_WALL={data['etf_call_wall']:.2f}\n")
        f.write(f"ETF_PUT_WALL={data['etf_put_wall']:.2f}\n")
        # Volatility index close for Edge Levels expected move
        if data.get('vol_close', 0) > 0:
            # VIX_CLOSE for SPX, VXN_CLOSE for QQQ
            vol_key = "VXN_CLOSE" if data['symbol'] == "QQQ" else "VIX_CLOSE"
            f.write(f"{vol_key}={data['vol_close']:.2f}\n")
    print(f"  Wrote {path}")


def main():
    if len(sys.argv) > 1:
        symbols = []
        for arg in sys.argv[1:]:
            sym = arg.upper()
            if sym not in SYMBOLS:
                print(f"Unknown symbol '{sym}'. Valid: {', '.join(SYMBOLS)}")
                continue
            symbols.append(sym)
        if not symbols:
            sys.exit(1)
    else:
        symbols = list(SYMBOLS)

    print(f"GEX Level Calculator (yfinance) -- {len(symbols)} symbol(s)\n")

    for symbol in symbols:
        try:
            print(f"[{symbol}]")
            data = compute_gex_levels(symbol)
            write_gex_file(data)
            print(f"  Gamma Flip: ${data['gamma_flip']:.2f}")
            print(f"  Call Wall:  ${data['call_wall']:.2f}")
            print(f"  Put Wall:   ${data['put_wall']:.2f}")
            print(f"  Net GEX:    {data['net_gex']:,.0f} ({data['regime']})")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")

    print(f"Done. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
