"""
Deterministic risk engine — Black-Scholes Greeks, VaR, portfolio metrics.
No LLM involved: all math is numpy/scipy.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from scipy.stats import norm


# ─── Greeks ───────────────────────────────────────────────────────────────────

@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # per calendar day
    vega: float = 0.0    # per 1% move in IV
    rho: float = 0.0
    iv: float = 0.0


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except (ValueError, ZeroDivisionError):
        return None, None


def black_scholes_greeks(
    S: float,        # spot price
    K: float,        # strike
    T: float,        # time to expiry in years
    r: float,        # risk-free rate (e.g. 0.05)
    sigma: float,    # implied volatility (e.g. 0.20)
    option_type: str = "call",
) -> Greeks:
    """Compute all five Black-Scholes Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return Greeks(iv=sigma)

    N = norm.cdf
    n = norm.pdf
    sign = 1 if option_type.lower() == "call" else -1

    delta = sign * N(sign * d1)
    gamma = n(d1) / (S * sigma * math.sqrt(T))
    theta_raw = (
        -(S * n(d1) * sigma) / (2 * math.sqrt(T))
        - sign * r * K * math.exp(-r * T) * N(sign * d2)
    )
    theta = theta_raw / 365.0  # per calendar day
    vega = S * n(d1) * math.sqrt(T) / 100.0   # per 1% IV move
    rho = sign * K * T * math.exp(-r * T) * N(sign * d2) / 100.0

    return Greeks(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
        iv=sigma,
    )


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes option price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    N = norm.cdf
    sign = 1 if option_type.lower() == "call" else -1
    price = sign * (S * N(sign * d1) - K * math.exp(-r * T) * N(sign * d2))
    return max(0.0, price)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver."""
    if T <= 0:
        return 0.0
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        if d1 is None:
            break
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))
    return round(sigma, 6)


# ─── Theta Decay Schedule ─────────────────────────────────────────────────────

def theta_decay_schedule(
    S: float,
    K: float,
    T_days: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    steps: int = 30,
) -> List[dict]:
    """
    Returns a list of {day, price, theta, delta} across the remaining life
    of the option — used to render the decay curve in the frontend.
    """
    result = []
    days = np.linspace(T_days, 0, steps, endpoint=False)
    for d in days:
        T = d / 365.0
        price = bs_price(S, K, T, r, sigma, option_type)
        g = black_scholes_greeks(S, K, T, r, sigma, option_type)
        result.append({
            "days_remaining": round(float(d), 2),
            "price": round(price, 4),
            "theta": round(g.theta, 6),
            "delta": round(g.delta, 6),
        })
    return result


# ─── Value at Risk ─────────────────────────────────────────────────────────────

def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR at the given confidence level."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def parametric_var(
    position_value: float,
    daily_vol: float,
    confidence: float = 0.95,
    holding_days: int = 1,
) -> float:
    """Parametric (normal) VaR."""
    z = norm.ppf(confidence)
    return position_value * daily_vol * z * math.sqrt(holding_days)


def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ─── Portfolio Aggregation ────────────────────────────────────────────────────

@dataclass
class PortfolioMetrics:
    total_notional: float = 0.0
    net_pnl: float = 0.0
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    portfolio_var_95: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    open_positions: int = 0
    alerts: List[str] = field(default_factory=list)


def aggregate_portfolio(positions: List[dict]) -> PortfolioMetrics:
    """
    Aggregate a list of position dicts into portfolio-level metrics.
    Each position dict: {pnl, notional, delta, gamma, theta, vega, direction}
    """
    m = PortfolioMetrics()
    pnls = []

    for p in positions:
        sign = 1 if p.get("direction", "long") == "long" else -1
        m.total_notional += abs(p.get("notional", 0))
        pnl = p.get("pnl", 0)
        m.net_pnl += pnl
        pnls.append(pnl)
        m.net_delta += sign * p.get("delta", 0)
        m.net_gamma += p.get("gamma", 0)
        m.net_theta += p.get("theta", 0)
        m.net_vega += sign * p.get("vega", 0)
        m.open_positions += 1

    if pnls:
        arr = np.array(pnls)
        m.portfolio_var_95 = parametric_var(
            m.total_notional, float(np.std(arr) / max(m.total_notional, 1))
        )
        wins = sum(1 for p in pnls if p > 0)
        m.win_rate = wins / len(pnls) * 100
        std = float(np.std(arr))
        m.sharpe_ratio = float(np.mean(arr) / std) if std > 0 else 0.0

    # Simple threshold alerts
    if abs(m.net_delta) > 500:
        m.alerts.append(f"High directional exposure: net delta {m.net_delta:.1f}")
    if m.net_theta < -200:
        m.alerts.append(f"Heavy theta burn: {m.net_theta:.2f}/day")
    if m.portfolio_var_95 > m.total_notional * 0.05:
        m.alerts.append("VaR exceeds 5% of notional")

    return m
