from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import math

from ..database import get_db, Trade, TradeStatus, TradeDirection, RiskSnapshot
from ..data.simulator import get_current_price
from ..engine.risk import (
    black_scholes_greeks, theta_decay_schedule, implied_volatility,
    aggregate_portfolio, parametric_var, PortfolioMetrics
)
from ..streaming.queue import sse_stream, CHANNEL_ALERTS, CHANNEL_PORTFOLIO

router = APIRouter(prefix="/risk", tags=["risk"])

RISK_FREE_RATE = 0.05


# ─── Schemas ──────────────────────────────────────────────────────────────────

class GreeksRequest(BaseModel):
    S: float
    K: float
    T_days: float
    sigma: float
    option_type: str = "call"
    r: float = RISK_FREE_RATE


class VarRequest(BaseModel):
    position_value: float
    daily_vol: float
    confidence: float = 0.95
    holding_days: int = 1


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/greeks")
async def compute_greeks(req: GreeksRequest):
    """Compute Black-Scholes Greeks for arbitrary option parameters."""
    T = req.T_days / 365.0
    greeks = black_scholes_greeks(req.S, req.K, T, req.r, req.sigma, req.option_type)
    return {
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "theta": greeks.theta,
        "vega": greeks.vega,
        "rho": greeks.rho,
        "iv": greeks.iv,
    }


@router.get("/greeks/trade/{trade_id}")
async def greeks_for_trade(trade_id: int, db: AsyncSession = Depends(get_db)):
    """Compute live Greeks for an options trade by ID."""
    trade = await db.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    if trade.asset_class.value != "option":
        raise HTTPException(status_code=400, detail="Trade is not an option")
    if not all([trade.strike, trade.expiry, trade.option_type]):
        raise HTTPException(status_code=400, detail="Option fields incomplete")

    S = get_current_price(trade.symbol)
    expiry_dt = datetime.strptime(trade.expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    T = max(0.0, (expiry_dt - datetime.now(timezone.utc)).days / 365.0)

    # Estimate IV from entry price using Newton-Raphson
    iv = implied_volatility(trade.entry_price, S, trade.strike, T, RISK_FREE_RATE, trade.option_type)
    greeks = black_scholes_greeks(S, trade.strike, T, RISK_FREE_RATE, iv, trade.option_type)

    # Persist snapshot
    snap = RiskSnapshot(
        trade_id=trade_id,
        delta=greeks.delta, gamma=greeks.gamma,
        theta=greeks.theta, vega=greeks.vega,
        rho=greeks.rho, iv=iv,
    )
    db.add(snap)
    await db.commit()

    return {
        "trade_id": trade_id,
        "symbol": trade.symbol,
        "spot": S,
        "strike": trade.strike,
        "expiry": trade.expiry,
        "days_to_expiry": round(T * 365),
        "iv": iv,
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "theta": greeks.theta,
        "vega": greeks.vega,
        "rho": greeks.rho,
    }


@router.get("/decay/{trade_id}")
async def theta_decay(
    trade_id: int,
    steps: int = Query(default=30, ge=5, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Theta decay curve for an options trade."""
    trade = await db.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    if not all([trade.strike, trade.expiry, trade.option_type]):
        raise HTTPException(status_code=400, detail="Option fields incomplete")

    S = get_current_price(trade.symbol)
    expiry_dt = datetime.strptime(trade.expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    T_days = max(1.0, (expiry_dt - datetime.now(timezone.utc)).days)
    iv = implied_volatility(trade.entry_price, S, trade.strike, T_days / 365.0, RISK_FREE_RATE, trade.option_type)

    curve = theta_decay_schedule(S, trade.strike, T_days, RISK_FREE_RATE, iv, trade.option_type, steps=steps)
    return {"trade_id": trade_id, "symbol": trade.symbol, "decay_curve": curve}


@router.post("/var")
async def compute_var(req: VarRequest):
    """Parametric VaR calculation."""
    var = parametric_var(req.position_value, req.daily_vol, req.confidence, req.holding_days)
    return {
        "var_95": round(var, 2),
        "position_value": req.position_value,
        "confidence": req.confidence,
        "holding_days": req.holding_days,
        "daily_vol": req.daily_vol,
    }


@router.get("/portfolio")
async def portfolio_summary(db: AsyncSession = Depends(get_db)):
    """Aggregate portfolio-level risk metrics across all open trades."""
    result = await db.execute(select(Trade).where(Trade.status == TradeStatus.OPEN))
    trades = result.scalars().all()

    positions = []
    for t in trades:
        S = get_current_price(t.symbol)
        t.current_price = S
        pos = {
            "pnl": t.pnl,
            "notional": t.notional or (t.entry_price * t.quantity),
            "direction": t.direction.value,
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

        if t.asset_class.value == "option" and t.strike and t.expiry and t.option_type:
            try:
                expiry_dt = datetime.strptime(t.expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                T = max(0.001, (expiry_dt - datetime.now(timezone.utc)).days / 365.0)
                iv = implied_volatility(t.entry_price, S, t.strike, T, RISK_FREE_RATE, t.option_type)
                g = black_scholes_greeks(S, t.strike, T, RISK_FREE_RATE, iv, t.option_type)
                pos.update({
                    "delta": g.delta * t.quantity,
                    "gamma": g.gamma * t.quantity,
                    "theta": g.theta * t.quantity,
                    "vega": g.vega * t.quantity,
                })
            except Exception:
                pass

        positions.append(pos)

    metrics = aggregate_portfolio(positions)
    return {
        "open_positions": metrics.open_positions,
        "total_notional": round(metrics.total_notional, 2),
        "net_pnl": round(metrics.net_pnl, 2),
        "net_delta": round(metrics.net_delta, 4),
        "net_gamma": round(metrics.net_gamma, 6),
        "net_theta": round(metrics.net_theta, 4),
        "net_vega": round(metrics.net_vega, 4),
        "portfolio_var_95": round(metrics.portfolio_var_95, 2),
        "sharpe_ratio": round(metrics.sharpe_ratio, 4),
        "win_rate": round(metrics.win_rate, 2),
        "alerts": metrics.alerts,
    }


@router.get("/stream/alerts")
async def stream_alerts():
    """SSE stream of risk alerts."""
    return StreamingResponse(
        sse_stream(CHANNEL_ALERTS),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/portfolio")
async def stream_portfolio():
    """SSE stream of portfolio metric snapshots."""
    return StreamingResponse(
        sse_stream(CHANNEL_PORTFOLIO),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
