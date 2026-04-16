"""
generate_gex_0dte.py - Live intraday 0DTE GEX levels

Fetches the same-day expiry SPY options chain, computes per-strike gamma
exposure, and derives intraday gamma walls and flip level. 0DTE gamma is
10-50x larger than 30DTE due to time decay acceleration, making these
levels highly actionable for intraday trading.

Data sources (auto-selected):
  - Massive.com API (if MASSIVE_API_KEY set) — live OI, greeks, IV
  - Tradier sandbox API (if TRADIER_API_KEY set) — 15-min delayed, greeks included
  - yfinance fallback (no key needed) — EOD OI, BS-computed gamma

Key differences from daily generate_gex.py:
  - Single expiration (today only) — no DTE weighting needed
  - No hysteresis — intraday walls should be reactive
  - OI + volume hybrid: effective_OI = max(OI, cumulative_volume)
  - Simplified gamma flip (no skew correction — negligible for 0DTE)
  - Faster refresh cycle (designed for 5-min polling)

Output: C:\\SierraChart\\Data\\gex_0dte_SPX.txt (same key=value format
as daily GEX, compatible with AMT_GEX_Levels.cpp "0DTE SPX" source).

Usage:
    python generate_gex_0dte.py              # single run
    run_0dte.bat                              # loop every 5 min during RTH

Dependencies: pip install numpy scipy yfinance requests massive
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
    print("Install with: python -m pip install numpy scipy yfinance")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRADIER_BASE = "https://sandbox.tradier.com/v1/markets"
OUTPUT_PATH = r"C:\SierraChart\Data\gex_0dte_SPX.txt"
RISK_FREE_RATE = 0.043


def _load_api_key():
    """Load Tradier API key from env or .env file. Returns None if not set."""
    key = os.environ.get("TRADIER_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TRADIER_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val and val != "your_sandbox_token_here":
                        return val
    return None

API_KEY = _load_api_key()


def _load_massive_key():
    """Load Massive.com API key from env or .env file. Returns None if not set."""
    key = os.environ.get("MASSIVE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MASSIVE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val and val != "your_massive_key_here":
                        return val
    return None


MASSIVE_KEY = _load_massive_key()


def _load_schwab_creds():
    """Load Schwab OAuth creds from env or .env file. Returns (client_id, secret, callback) or Nones."""
    cid = os.environ.get("SCHWAB_CLIENT_ID")
    sec = os.environ.get("SCHWAB_CLIENT_SECRET")
    cb = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
    if cid and sec:
        return cid, sec, cb
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        vals = {}
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                for key in ("SCHWAB_CLIENT_ID", "SCHWAB_CLIENT_SECRET", "SCHWAB_CALLBACK_URL"):
                    if line.startswith(key + "="):
                        vals[key] = line.split("=", 1)[1].strip().strip("'\"")
        cid = vals.get("SCHWAB_CLIENT_ID")
        sec = vals.get("SCHWAB_CLIENT_SECRET")
        cb = vals.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
        if cid and sec:
            return cid, sec, cb
    return None, None, None


SCHWAB_ID, SCHWAB_SECRET, SCHWAB_CALLBACK = _load_schwab_creds()
SCHWAB_TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".schwab_token.json")


# ---------------------------------------------------------------------------
# Black-Scholes gamma
# ---------------------------------------------------------------------------

def bs_gamma(S, K, T, r, sigma):
    """Vectorized Black-Scholes gamma."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Vectorized Black-Scholes delta.

    Returns call delta [0,+1] or put delta [-1,0].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
        call_delta = norm.cdf(d1)
    call_delta = np.nan_to_num(call_delta, nan=0.0, posinf=0.0, neginf=0.0)
    if option_type == "put":
        return call_delta - 1.0
    return call_delta


# ---------------------------------------------------------------------------
# Data source: Tradier
# ---------------------------------------------------------------------------

def _tradier_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }


def fetch_tradier(today_str):
    """Fetch SPY spot + 0DTE chain from Tradier. Returns (spot, chain)."""
    import requests

    # Spot
    r = requests.get(
        f"{TRADIER_BASE}/quotes",
        params={"symbols": "SPY", "greeks": "false"},
        headers=_tradier_headers(),
        timeout=10,
    )
    r.raise_for_status()
    spot = float(r.json()["quotes"]["quote"]["last"])

    # Chain
    r = requests.get(
        f"{TRADIER_BASE}/options/chains",
        params={"symbol": "SPY", "expiration": today_str, "greeks": "true"},
        headers=_tradier_headers(),
        timeout=15,
    )
    r.raise_for_status()
    options = r.json().get("options", {})
    if not options:
        return spot, []
    chain = options.get("option", [])
    if isinstance(chain, dict):
        chain = [chain]

    # Normalize to common format: list of dicts with keys:
    #   strike, open_interest, volume, option_type, implied_volatility, greeks
    normalized = []
    for opt in chain:
        normalized.append({
            "strike": float(opt["strike"]),
            "open_interest": int(opt.get("open_interest") or 0),
            "volume": int(opt.get("volume") or 0),
            "option_type": opt.get("option_type", "").lower(),
            "implied_volatility": float(opt.get("greeks", {}).get("mid_iv") or 0),
            "greeks": opt.get("greeks"),  # pre-computed gamma etc.
        })
    return spot, normalized


# ---------------------------------------------------------------------------
# Data source: Schwab (free, real-time, OAuth 2.0)
# ---------------------------------------------------------------------------

def _schwab_refresh_token():
    """Refresh the Schwab access token using the refresh token."""
    import json, base64
    import requests as req

    with open(SCHWAB_TOKEN_PATH, "r") as f:
        token_data = json.load(f)

    auth_str = base64.b64encode(f"{SCHWAB_ID}:{SCHWAB_SECRET}".encode()).decode()
    resp = req.post(
        "https://api.schwabapi.com/v1/oauth/token",
        headers={"Authorization": f"Basic {auth_str}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token",
              "refresh_token": token_data["refresh_token"]},
    )
    resp.raise_for_status()
    new_token = resp.json()
    # Preserve refresh_token if not returned in refresh response
    if "refresh_token" not in new_token:
        new_token["refresh_token"] = token_data["refresh_token"]
    with open(SCHWAB_TOKEN_PATH, "w") as f:
        json.dump(new_token, f)
    return new_token["access_token"]


def _schwab_get_access_token():
    """Get a valid Schwab access token, refreshing if needed."""
    import json

    if not os.path.exists(SCHWAB_TOKEN_PATH):
        raise FileNotFoundError("No Schwab token — run: python schwab_auth.py")

    with open(SCHWAB_TOKEN_PATH, "r") as f:
        token_data = json.load(f)

    # Try the existing access token first, refresh if it fails
    return token_data["access_token"]


def fetch_schwab(today_str):
    """Fetch SPY spot + 0DTE chain from Schwab. Returns (spot, chain)."""
    import requests as req

    token = _schwab_get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    resp = req.get(
        "https://api.schwabapi.com/marketdata/v1/chains",
        params={"symbol": "SPY", "fromDate": today_str, "toDate": today_str},
        headers=headers,
        timeout=15,
    )

    # If 401, refresh token and retry once
    if resp.status_code == 401:
        print("  Schwab: Token expired, refreshing...")
        token = _schwab_refresh_token()
        headers["Authorization"] = f"Bearer {token}"
        resp = req.get(
            "https://api.schwabapi.com/marketdata/v1/chains",
            params={"symbol": "SPY", "fromDate": today_str, "toDate": today_str},
            headers=headers,
            timeout=15,
        )

    resp.raise_for_status()
    data = resp.json()

    spot = data.get("underlyingPrice") or data.get("underlying", {}).get("last")
    if not spot:
        raise ValueError("Schwab: no underlying price in response")
    spot = float(spot)

    normalized = []
    for map_key, opt_type in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
        exp_map = data.get(map_key, {})
        for exp_date, strikes in exp_map.items():
            for strike_str, contracts in strikes.items():
                for opt in contracts:
                    strike = float(opt.get("strikePrice", 0))
                    if strike <= 0:
                        continue

                    oi = int(opt.get("openInterest", 0))
                    vol = int(opt.get("totalVolume", 0))
                    iv = float(opt.get("volatility", 0)) / 100.0  # Schwab returns %

                    greeks_dict = None
                    delta = opt.get("delta")
                    gamma = opt.get("gamma")
                    if delta is not None or gamma is not None:
                        greeks_dict = {
                            "delta": float(delta) if delta is not None else None,
                            "gamma": float(gamma) if gamma is not None else None,
                            "theta": float(opt["theta"]) if opt.get("theta") is not None else None,
                            "vega": float(opt["vega"]) if opt.get("vega") is not None else None,
                        }

                    normalized.append({
                        "strike": strike,
                        "open_interest": oi,
                        "volume": vol,
                        "option_type": opt_type,
                        "implied_volatility": iv,
                        "greeks": greeks_dict,
                    })

    return spot, normalized


# ---------------------------------------------------------------------------
# Data source: Massive.com (live OI, greeks, IV)
# ---------------------------------------------------------------------------

def fetch_massive(today_str):
    """Fetch SPY spot + 0DTE chain from Massive.com. Returns (spot, chain)."""
    from massive import RESTClient

    client = RESTClient(api_key=MASSIVE_KEY)
    snapshots = list(client.list_snapshot_options_chain(
        "SPY", params={"expiration_date": today_str},
    ))

    if not snapshots:
        return None, []

    # Spot price from first option's underlying asset
    spot = None
    for snap in snapshots:
        if snap.underlying_asset and snap.underlying_asset.price:
            spot = float(snap.underlying_asset.price)
            break
    if not spot:
        raise ValueError("Massive: no underlying price in snapshot")

    normalized = []
    for opt in snapshots:
        details = opt.details
        if not details or not details.strike_price:
            continue

        day = opt.day or type("D", (), {"volume": 0})()
        greeks_obj = opt.greeks

        # Convert greeks object to dict for downstream compat (greeks.get("gamma"))
        greeks_dict = None
        if greeks_obj:
            greeks_dict = {
                "delta": greeks_obj.delta,
                "gamma": greeks_obj.gamma,
                "theta": greeks_obj.theta,
                "vega": greeks_obj.vega,
            }

        normalized.append({
            "strike": float(details.strike_price),
            "open_interest": int(opt.open_interest or 0),
            "volume": int(day.volume or 0),
            "option_type": (details.contract_type or "").lower(),
            "implied_volatility": float(opt.implied_volatility or 0),
            "greeks": greeks_dict,
        })

    return spot, normalized


# ---------------------------------------------------------------------------
# Data source: yfinance (fallback — no API key needed)
# ---------------------------------------------------------------------------

def fetch_yfinance(today_str):
    """Fetch SPY spot + 0DTE chain from yfinance. Returns (spot, chain)."""
    ticker = yf.Ticker("SPY")

    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
    if not spot or spot <= 0:
        raise ValueError("Could not get SPY price from yfinance")

    # Check if today's expiration exists
    expirations = ticker.options
    if today_str not in expirations:
        # Find nearest expiration (might be next 0DTE day — MWF)
        print(f"  No expiration for {today_str}")
        print(f"  Available: {', '.join(expirations[:5])}...")
        # Use the nearest one for testing
        if expirations:
            nearest = expirations[0]
            print(f"  Using nearest: {nearest}")
            today_str = nearest
        else:
            return spot, []

    chain_data = ticker.option_chain(today_str)
    normalized = []

    for df, opt_type in [(chain_data.calls, "call"), (chain_data.puts, "put")]:
        for _, row in df.iterrows():
            iv = row.get("impliedVolatility", 0)
            oi = row.get("openInterest", 0)
            vol = row.get("volume", 0)
            # yfinance returns NaN for missing data
            if np.isnan(iv) if isinstance(iv, float) else False:
                iv = 0
            if np.isnan(oi) if isinstance(oi, float) else False:
                oi = 0
            if np.isnan(vol) if isinstance(vol, float) else False:
                vol = 0
            oi = int(oi)
            vol = int(vol)
            strike = float(row["strike"])

            normalized.append({
                "strike": strike,
                "open_interest": oi,
                "volume": vol,
                "option_type": opt_type,
                "implied_volatility": float(iv),
                "greeks": None,  # yfinance doesn't provide greeks
            })

    return spot, normalized


# ---------------------------------------------------------------------------
# GEX computation (source-agnostic — works with normalized chain)
# ---------------------------------------------------------------------------

def compute_0dte_gex(chain, spot):
    """Compute per-strike GEX from 0DTE chain.

    Returns (call_gex, put_gex) dicts mapping strike -> dollar gamma.
    Uses pre-computed gamma when available, falls back to BS.
    effective_OI = max(OI, cumulative_volume) to capture intraday building.
    """
    now = datetime.now()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    seconds_left = max((market_close - now).total_seconds(), 60)
    T = seconds_left / (252 * 6.5 * 3600)  # annualized trading time

    call_gex = {}
    put_gex = {}

    for opt in chain:
        strike = opt["strike"]
        oi = opt["open_interest"]
        volume = opt["volume"]
        option_type = opt["option_type"]

        if strike < spot * 0.85 or strike > spot * 1.15:
            continue

        effective_oi = max(oi, volume)
        if effective_oi <= 0:
            continue

        # Get gamma — prefer pre-computed, fallback to BS
        greeks = opt.get("greeks") or {}
        gamma = 0.0
        if greeks and greeks.get("gamma") is not None:
            gamma = float(greeks["gamma"])
        else:
            iv = opt.get("implied_volatility", 0) or 0
            if iv > 0.001:
                gamma = float(bs_gamma(spot, np.array([strike]), T,
                                       RISK_FREE_RATE, np.array([iv]))[0])

        if gamma <= 0:
            continue

        sign = 1.0 if option_type == "call" else -1.0
        dollar_gex = sign * gamma * effective_oi * 100 * spot * spot * 0.01

        target = call_gex if option_type == "call" else put_gex
        target[strike] = target.get(strike, 0.0) + dollar_gex

    return call_gex, put_gex


def compute_0dte_dex(chain, spot):
    """Compute per-strike dealer delta exposure (DEX) from 0DTE chain.

    Dealers are short options, so:
      call_DEX = -call_delta * OI * 100 * S * 0.01  (negative contribution)
      put_DEX  = -put_delta  * OI * 100 * S * 0.01  (positive contribution)

    Returns (call_dex, put_dex) dicts mapping strike -> dollar DEX.
    Uses Tradier's pre-computed delta when available, BS fallback.
    """
    now = datetime.now()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    seconds_left = max((market_close - now).total_seconds(), 60)
    T = seconds_left / (252 * 6.5 * 3600)

    call_dex = {}
    put_dex = {}

    for opt in chain:
        strike = opt["strike"]
        oi = opt["open_interest"]
        volume = opt["volume"]
        option_type = opt["option_type"]

        if strike < spot * 0.85 or strike > spot * 1.15:
            continue

        effective_oi = max(oi, volume)
        if effective_oi <= 0:
            continue

        # Get delta — prefer pre-computed, fallback to BS
        greeks = opt.get("greeks") or {}
        delta = None
        if greeks and greeks.get("delta") is not None:
            delta = float(greeks["delta"])
        else:
            iv = opt.get("implied_volatility", 0) or 0
            if iv > 0.001:
                delta = float(bs_delta(spot, np.array([strike]), T,
                                       RISK_FREE_RATE, np.array([iv]),
                                       option_type)[0])

        if delta is None or delta == 0.0:
            continue

        # Dealer perspective: dealers are short, so negate
        dollar_dex = -delta * effective_oi * 100 * spot * 0.01

        target = call_dex if option_type == "call" else put_dex
        target[strike] = target.get(strike, 0.0) + dollar_dex

    return call_dex, put_dex


def compute_cpr(chain, spot):
    """Compute call/put ratio in raw OI and notional terms.

    Returns (cpr_raw, cpr_notional).
    """
    total_call_oi = 0
    total_put_oi = 0
    call_notional = 0.0
    put_notional = 0.0

    for opt in chain:
        strike = opt["strike"]
        oi = opt["open_interest"]
        volume = opt["volume"]
        effective_oi = max(oi, volume)

        if strike < spot * 0.85 or strike > spot * 1.15:
            continue
        if effective_oi <= 0:
            continue

        if opt["option_type"] == "call":
            total_call_oi += effective_oi
            call_notional += effective_oi * strike * 100
        else:
            total_put_oi += effective_oi
            put_notional += effective_oi * strike * 100

    cpr_raw = total_call_oi / total_put_oi if total_put_oi > 0 else 0.0
    cpr_notional = call_notional / put_notional if put_notional > 0 else 0.0

    return cpr_raw, cpr_notional


def find_gamma_flip_0dte(chain, spot):
    """Find gamma flip for 0DTE — simplified sweep without skew correction."""
    now = datetime.now()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    seconds_left = max((market_close - now).total_seconds(), 60)
    T = seconds_left / (252 * 6.5 * 3600)

    strikes_data = []
    for opt in chain:
        strike = opt["strike"]
        oi = opt["open_interest"]
        volume = opt["volume"]
        option_type = opt["option_type"]

        if strike < spot * 0.85 or strike > spot * 1.15:
            continue

        effective_oi = max(oi, volume)
        if effective_oi <= 0:
            continue

        iv = opt.get("implied_volatility", 0) or 0
        if iv <= 0.001:
            iv = 0.30  # fallback assumption

        sign = 1.0 if option_type == "call" else -1.0
        strikes_data.append((strike, effective_oi, iv, sign))

    if not strikes_data:
        return spot

    K_arr = np.array([d[0] for d in strikes_data])
    OI_arr = np.array([d[1] for d in strikes_data])
    IV_arr = np.array([d[2] for d in strikes_data])
    sign_arr = np.array([d[3] for d in strikes_data])

    hyp = np.linspace(spot * 0.92, spot * 1.08, 200)
    net_gex = np.zeros(len(hyp))

    for i, S in enumerate(hyp):
        gamma = bs_gamma(S, K_arr, T, RISK_FREE_RATE, IV_arr)
        gex = sign_arr * gamma * OI_arr * 100 * S * S * 0.01
        net_gex[i] = np.sum(gex)

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


def build_profile(call_vals, put_vals):
    """Build full profile: net value per strike across calls and puts."""
    profile = {}
    for s, v in call_vals.items():
        profile[s] = profile.get(s, 0.0) + v
    for s, v in put_vals.items():
        profile[s] = profile.get(s, 0.0) + v
    return profile


# ---------------------------------------------------------------------------
# SPY -> SPX conversion
# ---------------------------------------------------------------------------

def get_spy_to_spx_ratio():
    """Get SPX/SPY ratio for converting ETF strikes to index space."""
    try:
        spx = yf.Ticker("^GSPC")
        spx_price = spx.fast_info["lastPrice"]
        return spx_price
    except Exception as e:
        print(f"  Warning: could not fetch ^GSPC, estimating ratio: {e}")
        return None


def get_es_basis(spx_price):
    """Get ES-SPX basis (ES futures premium over SPX cash).

    ES futures trade at a premium/discount vs SPX cash due to cost of carry.
    Returns basis in index points (typically +15 to +40).
    """
    try:
        es = yf.Ticker("ES=F")
        es_price = es.fast_info["lastPrice"]
        basis = es_price - spx_price
        return es_price, basis
    except Exception as e:
        print(f"  Warning: could not fetch ES=F, no basis adjustment: {e}")
        return None, 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(data):
    """Write 0DTE GEX data in key=value format for SC study."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(f"SYMBOL={data['symbol']}\n")
        f.write(f"UNDERLYING={data['underlying']:.2f}\n")
        f.write(f"TIMESTAMP={data['timestamp']}\n")
        f.write(f"REGIME={data['regime']}\n")
        f.write(f"GAMMA_FLIP={data['gamma_flip']:.2f}\n")
        f.write(f"CALL_WALL={data['call_wall']:.2f}\n")
        if data.get('call_wall_low', 0) > 0:
            f.write(f"CALL_WALL_LOW={data['call_wall_low']:.2f}\n")
            f.write(f"CALL_WALL_HIGH={data['call_wall_high']:.2f}\n")
        f.write(f"PUT_WALL={data['put_wall']:.2f}\n")
        if data.get('put_wall_low', 0) > 0:
            f.write(f"PUT_WALL_LOW={data['put_wall_low']:.2f}\n")
            f.write(f"PUT_WALL_HIGH={data['put_wall_high']:.2f}\n")
        if data.get('vol_trigger', 0) > 0:
            f.write(f"VOL_TRIGGER={data['vol_trigger']:.2f}\n")
        if data.get('hvl', 0) > 0:
            f.write(f"HVL={data['hvl']:.2f}\n")
        if data.get('key_call_2', 0) > 0:
            f.write(f"KEY_CALL_2={data['key_call_2']:.2f}\n")
        if data.get('key_call_3', 0) > 0:
            f.write(f"KEY_CALL_3={data['key_call_3']:.2f}\n")
        if data.get('key_put_2', 0) > 0:
            f.write(f"KEY_PUT_2={data['key_put_2']:.2f}\n")
        if data.get('key_put_3', 0) > 0:
            f.write(f"KEY_PUT_3={data['key_put_3']:.2f}\n")
        f.write(f"NET_GEX={data['net_gex']:.0f}\n")
        profile = data.get("gex_profile", [])
        if profile:
            pairs = ",".join(f"{strike}:{gex}" for strike, gex in profile)
            f.write(f"GEX_PROFILE={pairs}\n")
        # DEX + CPR fields (backwards-compatible — old parsers ignore unknown keys)
        if "net_dex" in data:
            f.write(f"NET_DEX={data['net_dex']:.0f}\n")
            f.write(f"DEX_REGIME={data['dex_regime']}\n")
        if "cpr_raw" in data:
            f.write(f"CPR_RAW={data['cpr_raw']:.4f}\n")
            f.write(f"CPR_NOTIONAL={data['cpr_notional']:.4f}\n")
        dex_profile = data.get("dex_profile", [])
        if dex_profile:
            pairs = ",".join(f"{strike}:{dex}" for strike, dex in dex_profile)
            f.write(f"DEX_PROFILE={pairs}\n")
    print(f"  Wrote {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    today = datetime.now().strftime("%Y-%m-%d")

    # Pick data source: Schwab → Massive → Tradier → yfinance
    if SCHWAB_ID:
        source = "Schwab"
        print(f"0DTE GEX Calculator (Schwab) — {today}\n")
    elif MASSIVE_KEY:
        source = "Massive"
        print(f"0DTE GEX Calculator (Massive) — {today}\n")
    elif API_KEY:
        source = "Tradier"
        print(f"0DTE GEX Calculator (Tradier) — {today}\n")
    else:
        source = "yfinance"
        print(f"0DTE GEX Calculator (yfinance) — {today}")
        print(f"  (Set SCHWAB_CLIENT_ID for free real-time data)\n")

    # 1. Fetch SPY spot + 0DTE chain
    print("  Fetching SPY quote + 0DTE chain...")
    try:
        if source == "Schwab":
            spy_spot, chain = fetch_schwab(today)
        elif source == "Massive":
            spy_spot, chain = fetch_massive(today)
        elif source == "Tradier":
            spy_spot, chain = fetch_tradier(today)
        else:
            spy_spot, chain = fetch_yfinance(today)
    except Exception as e:
        if source != "yfinance":
            print(f"  {source} failed: {e}")
            print(f"  Falling back to yfinance...")
            source = "yfinance"
            try:
                spy_spot, chain = fetch_yfinance(today)
            except Exception as e2:
                print(f"  yfinance also failed: {e2}")
                sys.exit(1)
        else:
            print(f"  Error fetching data: {e}")
            sys.exit(1)

    print(f"  SPY spot: ${spy_spot:.2f}")

    if not chain:
        print("  No 0DTE chain available (market closed or no expiration today).")
        sys.exit(0)

    calls = [o for o in chain if o["option_type"] == "call"]
    puts = [o for o in chain if o["option_type"] == "put"]
    print(f"  Chain: {len(calls)} calls, {len(puts)} puts ({source})")

    # 2. Compute per-strike GEX
    call_gex, put_gex = compute_0dte_gex(chain, spy_spot)
    print(f"  Active strikes: {len(call_gex)} call, {len(put_gex)} put")

    if not call_gex and not put_gex:
        print("  No valid GEX data — all strikes filtered out.")
        sys.exit(0)

    # 2b. Compute DEX (delta exposure)
    call_dex, put_dex = compute_0dte_dex(chain, spy_spot)
    net_dex = sum(call_dex.values()) + sum(put_dex.values())
    dex_regime = "dealer_long" if net_dex >= 0 else "dealer_short"
    print(f"  DEX strikes: {len(call_dex)} call, {len(put_dex)} put")
    print(f"  Net DEX: {net_dex:,.0f} ({dex_regime})")

    # 2c. Compute CPR (call/put ratio)
    cpr_raw, cpr_notional = compute_cpr(chain, spy_spot)
    print(f"  CPR: {cpr_raw:.2f} raw, {cpr_notional:.2f} notional")

    # 3. Walls — zone detection for 0DTE
    #    Walk from spot outward, accumulate gamma. Zone = 25%-75% cumulative band.
    #    CW_LOW/CW_HIGH = call zone edges, PW_LOW/PW_HIGH = put zone edges.
    #    CALL_WALL = CW_HIGH (75% threshold, same as old single-wall behavior).
    #    PUT_WALL  = PW_LOW  (75% threshold, same as old single-wall behavior).
    call_wall = spy_spot
    call_wall_low = 0.0
    call_wall_high = 0.0

    if call_gex:
        # Strikes above (or at) spot, sorted ascending
        above = sorted([s for s in call_gex if s >= spy_spot])
        if above:
            total_call = sum(call_gex[s] for s in above)
            if total_call > 0:
                cum = 0.0
                thresh_lo = total_call * 0.25
                thresh_hi = total_call * 0.75
                for s in above:
                    cum += call_gex[s]
                    if call_wall_low == 0.0 and cum >= thresh_lo:
                        call_wall_low = s
                    if cum >= thresh_hi:
                        call_wall_high = s
                        call_wall = s  # 75% threshold = wall (same as old behavior)
                        break
                else:
                    if call_wall_high == 0.0:
                        call_wall_high = above[-1]
                    call_wall = call_wall_high
                if call_wall_low == 0.0:
                    call_wall_low = above[0]
            else:
                call_wall = max(call_gex, key=call_gex.get)
        else:
            call_wall = max(call_gex, key=call_gex.get)

    put_wall = spy_spot
    put_wall_low = 0.0
    put_wall_high = 0.0

    if put_gex:
        # Strikes below (or at) spot, sorted descending (start from spot, walk down)
        below = sorted([s for s in put_gex if s <= spy_spot], reverse=True)
        if below:
            total_put = sum(abs(put_gex[s]) for s in below)
            if total_put > 0:
                cum = 0.0
                thresh_lo = total_put * 0.25
                thresh_hi = total_put * 0.75
                for s in below:
                    cum += abs(put_gex[s])
                    if put_wall_high == 0.0 and cum >= thresh_lo:
                        put_wall_high = s  # first hit walking down = top of zone
                    if cum >= thresh_hi:
                        put_wall_low = s   # deeper = bottom of zone
                        put_wall = s       # 75% threshold = wall (same as old behavior)
                        break
                else:
                    if put_wall_low == 0.0:
                        put_wall_low = below[-1]
                    put_wall = put_wall_low
                if put_wall_high == 0.0:
                    put_wall_high = below[0]
            else:
                put_wall = max(put_gex, key=lambda k: abs(put_gex[k]))
        else:
            put_wall = max(put_gex, key=lambda k: abs(put_gex[k]))

    # Sanity: call wall must be above spot, put wall below spot.
    # Fallback paths (thin data) can violate this — clamp to spot.
    if call_wall < spy_spot and call_gex:
        above_only = [s for s in call_gex if s >= spy_spot]
        call_wall = max(above_only, key=lambda s: call_gex[s]) if above_only else spy_spot
    if put_wall > spy_spot and put_gex:
        below_only = [s for s in put_gex if s <= spy_spot]
        put_wall = min(below_only, key=lambda s: put_gex[s]) if below_only else spy_spot

    # 3b. Key gamma strikes + HVL
    #  Vol Trigger = max single call gamma above spot
    #  Key Call 2/3 = 2nd/3rd highest call gamma above spot
    vol_trigger = 0.0
    key_call_2 = 0.0
    key_call_3 = 0.0
    if call_gex:
        above_calls = sorted(
            [(s, call_gex[s]) for s in call_gex if s >= spy_spot],
            key=lambda x: x[1], reverse=True,
        )
        if len(above_calls) >= 1:
            vol_trigger = above_calls[0][0]
        if len(above_calls) >= 2:
            key_call_2 = above_calls[1][0]
        if len(above_calls) >= 3:
            key_call_3 = above_calls[2][0]

    #  Key Put 2/3 = 2nd/3rd highest |put gamma| below spot
    key_put_2 = 0.0
    key_put_3 = 0.0
    if put_gex:
        below_puts = sorted(
            [(s, abs(put_gex[s])) for s in put_gex if s <= spy_spot],
            key=lambda x: x[1], reverse=True,
        )
        if len(below_puts) >= 2:
            key_put_2 = below_puts[1][0]
        if len(below_puts) >= 3:
            key_put_3 = below_puts[2][0]

    #  HVL = highest total option volume strike
    hvl = 0.0
    vol_by_strike = {}
    for opt in chain:
        s = opt["strike"]
        v = opt["volume"]
        if s < spy_spot * 0.85 or s > spy_spot * 1.15:
            continue
        vol_by_strike[s] = vol_by_strike.get(s, 0) + v
    if vol_by_strike:
        hvl = max(vol_by_strike, key=vol_by_strike.get)

    # 4. Net GEX + regime
    net_gex = sum(call_gex.values()) + sum(put_gex.values())
    regime = "positive_gamma" if net_gex >= 0 else "negative_gamma"

    # 5. Gamma flip
    print("  Computing gamma flip...")
    gamma_flip = find_gamma_flip_0dte(chain, spy_spot)

    # 6. Profiles (all active strikes, net per strike)
    profile_etf = build_profile(call_gex, put_gex)
    dex_profile_etf = build_profile(call_dex, put_dex)

    # 7. Convert SPY -> SPX index space
    spx_price = get_spy_to_spx_ratio()
    if spx_price:
        ratio = spx_price / spy_spot
        print(f"  SPX: {spx_price:.2f} (ratio {ratio:.2f}x)")
        gamma_flip *= ratio
        call_wall *= ratio
        put_wall *= ratio
        call_wall_low *= ratio
        call_wall_high *= ratio
        put_wall_low *= ratio
        put_wall_high *= ratio
        vol_trigger *= ratio
        hvl *= ratio
        key_call_2 *= ratio
        key_call_3 *= ratio
        key_put_2 *= ratio
        key_put_3 *= ratio
        underlying = spx_price
    else:
        ratio = 10.0
        gamma_flip *= ratio
        call_wall *= ratio
        put_wall *= ratio
        call_wall_low *= ratio
        call_wall_high *= ratio
        put_wall_low *= ratio
        put_wall_high *= ratio
        vol_trigger *= ratio
        hvl *= ratio
        key_call_2 *= ratio
        key_call_3 *= ratio
        key_put_2 *= ratio
        key_put_3 *= ratio
        underlying = spy_spot * ratio
        print(f"  Using approximate ratio {ratio:.0f}x")

    # Convert profiles to index space
    gex_profile = sorted(
        [(round(s * ratio), int(profile_etf[s])) for s in profile_etf],
        key=lambda p: p[0],
    )
    dex_profile = sorted(
        [(round(s * ratio), int(dex_profile_etf[s])) for s in dex_profile_etf],
        key=lambda p: p[0],
    )

    # 7b. Apply ES-SPX basis (levels displayed on ES chart, not SPX)
    es_price, basis = get_es_basis(underlying)
    if basis != 0:
        print(f"  ES=F: {es_price:.2f} (basis +{basis:.1f})")
        gamma_flip += basis
        call_wall += basis
        put_wall += basis
        call_wall_low += basis
        call_wall_high += basis
        put_wall_low += basis
        put_wall_high += basis
        vol_trigger += basis
        hvl += basis
        key_call_2 += basis
        key_call_3 += basis
        key_put_2 += basis
        key_put_3 += basis
        underlying = es_price
        # Shift profiles too
        gex_profile = [(s + round(basis), g) for s, g in gex_profile]
        dex_profile = [(s + round(basis), d) for s, d in dex_profile]

    # 8. Write output
    data = {
        "symbol": "SPX",
        "underlying": underlying,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regime": regime,
        "gamma_flip": gamma_flip,
        "call_wall": call_wall,
        "call_wall_low": call_wall_low,
        "call_wall_high": call_wall_high,
        "put_wall": put_wall,
        "put_wall_low": put_wall_low,
        "put_wall_high": put_wall_high,
        "vol_trigger": vol_trigger,
        "hvl": hvl,
        "key_call_2": key_call_2,
        "key_call_3": key_call_3,
        "key_put_2": key_put_2,
        "key_put_3": key_put_3,
        "net_gex": net_gex,
        "gex_profile": gex_profile,
        "net_dex": net_dex,
        "dex_regime": dex_regime,
        "cpr_raw": cpr_raw,
        "cpr_notional": cpr_notional,
        "dex_profile": dex_profile,
    }
    write_output(data)

    print(f"\n  Gamma Flip:  ${data['gamma_flip']:.2f}")
    print(f"  Call Wall:   ${data['call_wall']:.2f}  [{data['call_wall_low']:.2f} - {data['call_wall_high']:.2f}]")
    print(f"  Put Wall:    ${data['put_wall']:.2f}  [{data['put_wall_low']:.2f} - {data['put_wall_high']:.2f}]")
    print(f"  Vol Trigger: ${data['vol_trigger']:.2f}")
    print(f"  HVL:         ${data['hvl']:.2f}")
    print(f"  Key Call 2:  ${data['key_call_2']:.2f}")
    print(f"  Key Call 3:  ${data['key_call_3']:.2f}")
    print(f"  Key Put 2:   ${data['key_put_2']:.2f}")
    print(f"  Key Put 3:   ${data['key_put_3']:.2f}")
    print(f"  Net GEX:     {data['net_gex']:,.0f} ({data['regime']})")
    print(f"  Net DEX:     {data['net_dex']:,.0f} ({data['dex_regime']})")
    print(f"  CPR:         {data['cpr_raw']:.2f} ({data['cpr_notional']:.2f}n)")
    print(f"  GEX Profile: {len(gex_profile)} strikes")
    print(f"  DEX Profile: {len(dex_profile)} strikes")
    print()


if __name__ == "__main__":
    main()
