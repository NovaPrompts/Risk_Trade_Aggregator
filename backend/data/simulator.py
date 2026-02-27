"""
Deterministic market data simulator.
Uses seeded numpy RNG so results are reproducible across restarts.
When a live API key is configured, this module is bypassed by the feed manager.
"""
import asyncio
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import numpy as np

# Base seeds and parameters per symbol
_SYMBOL_PARAMS: Dict[str, dict] = {
    "SPY":  {"seed": 42,  "price": 540.0,  "vol": 0.13, "drift": 0.08},
    "QQQ":  {"seed": 7,   "price": 460.0,  "vol": 0.18, "drift": 0.10},
    "AAPL": {"seed": 13,  "price": 220.0,  "vol": 0.22, "drift": 0.09},
    "TSLA": {"seed": 99,  "price": 250.0,  "vol": 0.55, "drift": 0.05},
    "NVDA": {"seed": 37,  "price": 880.0,  "vol": 0.45, "drift": 0.15},
    "BTC":  {"seed": 101, "price": 67000.0,"vol": 0.70, "drift": 0.12},
    "ETH":  {"seed": 55,  "price": 3500.0, "vol": 0.65, "drift": 0.10},
    "GLD":  {"seed": 21,  "price": 225.0,  "vol": 0.12, "drift": 0.03},
}

_DEFAULT_SYMBOLS = list(_SYMBOL_PARAMS.keys())


def _gbm_path(seed: int, S0: float, mu: float, sigma: float, n: int, dt: float = 1/252) -> np.ndarray:
    """Geometric Brownian Motion price path."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    prices = S0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[S0], prices])


def _current_bar_index() -> int:
    """Map wall-clock time to a bar index (advances every 5 seconds in sim mode)."""
    epoch = int(time.time())
    return epoch // 5  # one new bar every 5 seconds


def get_current_price(symbol: str) -> float:
    """Get simulated current price for a symbol."""
    params = _SYMBOL_PARAMS.get(symbol.upper(), {"seed": hash(symbol) % 1000, "price": 100.0, "vol": 0.25, "drift": 0.07})
    idx = _current_bar_index() % 10000  # cycle every ~14 hours
    path = _gbm_path(params["seed"], params["price"], params["drift"], params["vol"], idx + 1)
    return round(float(path[idx]), 4)


def get_ohlcv(symbol: str, bars: int = 60) -> List[dict]:
    """Return the last N 5-second OHLCV bars."""
    params = _SYMBOL_PARAMS.get(symbol.upper(), {"seed": hash(symbol) % 1000, "price": 100.0, "vol": 0.25, "drift": 0.07})
    end_idx = _current_bar_index()
    start_idx = max(0, end_idx - bars)
    total = end_idx + 1

    path = _gbm_path(params["seed"], params["price"], params["drift"], params["vol"], total)
    rng = np.random.default_rng(params["seed"] + 1000)

    result = []
    now_ts = int(time.time())
    bar_duration = 5  # seconds

    for i in range(start_idx, end_idx):
        close = float(path[i + 1])
        open_ = float(path[i])
        hi_mult = 1 + abs(float(rng.normal(0, params["vol"] * 0.01)))
        lo_mult = 1 - abs(float(rng.normal(0, params["vol"] * 0.01)))
        high = max(open_, close) * hi_mult
        low = min(open_, close) * lo_mult
        volume = float(rng.integers(10_000, 500_000))
        ts = (now_ts - (end_idx - i) * bar_duration)

        result.append({
            "timestamp": ts,
            "open": round(open_, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close, 4),
            "volume": volume,
        })

    return result


def get_options_chain(
    symbol: str,
    expiries: int = 3,
    strikes_per_expiry: int = 7,
    r: float = 0.05,
) -> List[dict]:
    """
    Generate a synthetic options chain.
    Returns a list of option contracts with BS-computed prices and greeks.
    """
    from ..engine.risk import black_scholes_greeks, bs_price

    S = get_current_price(symbol)
    params = _SYMBOL_PARAMS.get(symbol.upper(), {"vol": 0.25})
    base_iv = params["vol"]
    rng = np.random.default_rng(params.get("seed", 42) + 2000)

    contracts = []
    today = datetime.now(timezone.utc)

    for exp_offset in [30, 60, 90][:expiries]:
        expiry_date = today + timedelta(days=exp_offset)
        T = exp_offset / 365.0
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        # ATM strike rounded to nearest $5
        atm = round(S / 5) * 5
        strike_range = [atm + (i - strikes_per_expiry // 2) * 5 for i in range(strikes_per_expiry)]

        for K in strike_range:
            for opt_type in ("call", "put"):
                moneyness = math.log(S / K) if K > 0 else 0
                # Volatility smile: higher IV for OTM options
                smile_adj = float(rng.uniform(0.02, 0.06)) * abs(moneyness) * 5
                iv = max(0.05, base_iv + smile_adj)

                price = bs_price(S, K, T, r, iv, opt_type)
                greeks = black_scholes_greeks(S, K, T, r, iv, opt_type)

                bid = round(max(0.01, price * (1 - float(rng.uniform(0.01, 0.03)))), 2)
                ask = round(price * (1 + float(rng.uniform(0.01, 0.03))), 2)
                oi = int(rng.integers(100, 50_000))
                volume = int(rng.integers(0, oi // 2))

                contracts.append({
                    "symbol": symbol.upper(),
                    "expiry": expiry_str,
                    "strike": K,
                    "type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2),
                    "iv": round(iv, 4),
                    "open_interest": oi,
                    "volume": volume,
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                    "rho": greeks.rho,
                    "T_years": round(T, 4),
                    "spot": round(S, 4),
                })

    return contracts


def get_market_snapshot() -> dict:
    """Returns a snapshot of all tracked symbols."""
    snapshot = {}
    for sym in _DEFAULT_SYMBOLS:
        bars = get_ohlcv(sym, bars=2)
        if len(bars) >= 2:
            prev_close = bars[-2]["close"]
            curr_close = bars[-1]["close"]
            change_pct = (curr_close - prev_close) / prev_close * 100 if prev_close else 0
        else:
            curr_close = get_current_price(sym)
            change_pct = 0.0

        snapshot[sym] = {
            "price": round(curr_close, 4),
            "change_pct": round(change_pct, 3),
            "timestamp": int(time.time()),
        }
    return snapshot


async def price_stream(symbol: str, interval: float = 5.0):
    """Async generator: yields a new price tick every `interval` seconds."""
    while True:
        yield {
            "symbol": symbol,
            "price": get_current_price(symbol),
            "timestamp": int(time.time()),
        }
        await asyncio.sleep(interval)
