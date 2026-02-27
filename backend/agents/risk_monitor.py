"""
Background risk monitor agent.
Runs every N seconds, checks all open positions for threshold breaches,
and publishes alerts + portfolio snapshots to the event bus.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from ..database import AsyncSessionLocal, Trade, TradeStatus, Alert
from ..data.simulator import get_current_price
from ..engine.risk import black_scholes_greeks, implied_volatility, aggregate_portfolio
from ..streaming.queue import publish, CHANNEL_ALERTS, CHANNEL_PORTFOLIO, CHANNEL_PRICES

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05
CHECK_INTERVAL = 10  # seconds between full portfolio sweeps
PRICE_TICK_INTERVAL = 5  # seconds between price broadcasts


# ─── Price Broadcaster ────────────────────────────────────────────────────────

async def _price_broadcaster():
    """Broadcasts current prices for all tracked symbols every PRICE_TICK_INTERVAL seconds."""
    from ..data.simulator import _DEFAULT_SYMBOLS, get_market_snapshot
    while True:
        try:
            snapshot = get_market_snapshot()
            await publish(CHANNEL_PRICES, {
                "type": "snapshot",
                "data": snapshot,
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
            })
        except Exception as e:
            logger.error("Price broadcaster error: %s", e)
        await asyncio.sleep(PRICE_TICK_INTERVAL)


# ─── Portfolio Sweeper ────────────────────────────────────────────────────────

async def _portfolio_sweeper():
    """
    Every CHECK_INTERVAL seconds:
    1. Fetch all open trades
    2. Update current prices
    3. Check stop-loss / take-profit breaches
    4. Compute portfolio Greeks and VaR
    5. Emit alerts and portfolio snapshots
    """
    while True:
        try:
            await _run_sweep()
        except Exception as e:
            logger.error("Portfolio sweep error: %s", e)
        await asyncio.sleep(CHECK_INTERVAL)


async def _run_sweep():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Trade).where(Trade.status == TradeStatus.OPEN))
        trades = result.scalars().all()

        if not trades:
            return

        positions = []
        new_alerts = []

        for trade in trades:
            S = get_current_price(trade.symbol)
            trade.current_price = S

            pos = {
                "pnl": trade.pnl,
                "notional": trade.notional or trade.entry_price * trade.quantity,
                "direction": trade.direction.value,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
            }

            # Options Greeks
            if trade.asset_class.value == "option" and trade.strike and trade.expiry and trade.option_type:
                try:
                    expiry_dt = datetime.strptime(trade.expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    T = max(0.001, (expiry_dt - datetime.now(timezone.utc)).days / 365.0)
                    iv = implied_volatility(
                        trade.entry_price, S, trade.strike, T, RISK_FREE_RATE, trade.option_type
                    )
                    g = black_scholes_greeks(S, trade.strike, T, RISK_FREE_RATE, iv, trade.option_type)
                    pos.update({
                        "delta": g.delta * trade.quantity,
                        "gamma": g.gamma * trade.quantity,
                        "theta": g.theta * trade.quantity,
                        "vega": g.vega * trade.quantity,
                    })
                except Exception:
                    pass

            # Stop-loss check
            if trade.stop_loss:
                if trade.direction.value == "long" and S <= trade.stop_loss:
                    new_alerts.append(Alert(
                        trade_id=trade.id,
                        level="critical",
                        message=f"STOP-LOSS: {trade.symbol} hit ${S:.2f} ≤ stop ${trade.stop_loss:.2f}",
                    ))
                elif trade.direction.value == "short" and S >= trade.stop_loss:
                    new_alerts.append(Alert(
                        trade_id=trade.id,
                        level="critical",
                        message=f"STOP-LOSS: {trade.symbol} hit ${S:.2f} ≥ stop ${trade.stop_loss:.2f}",
                    ))

            # Take-profit check
            if trade.take_profit:
                if trade.direction.value == "long" and S >= trade.take_profit:
                    new_alerts.append(Alert(
                        trade_id=trade.id,
                        level="info",
                        message=f"TAKE-PROFIT: {trade.symbol} hit ${S:.2f} ≥ target ${trade.take_profit:.2f}",
                    ))
                elif trade.direction.value == "short" and S <= trade.take_profit:
                    new_alerts.append(Alert(
                        trade_id=trade.id,
                        level="info",
                        message=f"TAKE-PROFIT: {trade.symbol} hit ${S:.2f} ≤ target ${trade.take_profit:.2f}",
                    ))

            positions.append(pos)

        # Portfolio-level metrics
        metrics = aggregate_portfolio(positions)

        # Portfolio-level alerts
        for alert_msg in metrics.alerts:
            new_alerts.append(Alert(level="warn", message=alert_msg))

        # Persist alerts and broadcast
        for alert in new_alerts:
            db.add(alert)
            await publish(CHANNEL_ALERTS, {
                "type": "alert",
                "level": alert.level,
                "message": alert.message,
                "trade_id": alert.trade_id,
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
            })

        if new_alerts:
            await db.commit()

        # Broadcast portfolio snapshot
        await publish(CHANNEL_PORTFOLIO, {
            "type": "portfolio_snapshot",
            "open_positions": metrics.open_positions,
            "total_notional": round(metrics.total_notional, 2),
            "net_pnl": round(metrics.net_pnl, 2),
            "net_delta": round(metrics.net_delta, 4),
            "net_theta": round(metrics.net_theta, 4),
            "net_vega": round(metrics.net_vega, 4),
            "portfolio_var_95": round(metrics.portfolio_var_95, 2),
            "win_rate": round(metrics.win_rate, 2),
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
        })

        logger.debug(
            "Sweep complete: %d positions | PnL: $%.2f | Alerts: %d",
            len(trades), metrics.net_pnl, len(new_alerts),
        )


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def start_agents():
    """Start all background agents as asyncio tasks."""
    logger.info("Starting risk monitor agents")
    asyncio.create_task(_price_broadcaster(), name="price-broadcaster")
    asyncio.create_task(_portfolio_sweeper(), name="portfolio-sweeper")
