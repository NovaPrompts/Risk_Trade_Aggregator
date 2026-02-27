from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone

from ..database import get_db, Trade, TradeDirection, AssetClass, TradeStatus
from ..data.simulator import get_current_price
from ..engine.risk import black_scholes_greeks

router = APIRouter(prefix="/trades", tags=["trades"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TradeCreate(BaseModel):
    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    direction: TradeDirection = TradeDirection.LONG
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Options
    strike: Optional[float] = None
    expiry: Optional[str] = None
    option_type: Optional[str] = None  # "call" or "put"


class TradeUpdate(BaseModel):
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: Optional[TradeStatus] = None
    exit_price: Optional[float] = None


class TradeOut(BaseModel):
    id: int
    symbol: str
    asset_class: str
    direction: str
    status: str
    entry_price: float
    current_price: Optional[float]
    exit_price: Optional[float]
    quantity: float
    pnl: float
    pnl_pct: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strike: Optional[float]
    expiry: Optional[str]
    option_type: Optional[str]
    opened_at: datetime

    model_config = {"from_attributes": True}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("", response_model=List[TradeOut])
async def list_trades(
    status: Optional[TradeStatus] = None,
    symbol: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Trade)
    if status:
        q = q.where(Trade.status == status)
    if symbol:
        q = q.where(Trade.symbol == symbol.upper())
    q = q.order_by(Trade.opened_at.desc())
    result = await db.execute(q)
    trades = result.scalars().all()

    # Refresh current prices from simulator
    out = []
    for t in trades:
        if t.status == TradeStatus.OPEN:
            t.current_price = get_current_price(t.symbol)
        out.append(TradeOut(
            id=t.id,
            symbol=t.symbol,
            asset_class=t.asset_class.value,
            direction=t.direction.value,
            status=t.status.value,
            entry_price=t.entry_price,
            current_price=t.current_price,
            exit_price=t.exit_price,
            quantity=t.quantity,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct,
            stop_loss=t.stop_loss,
            take_profit=t.take_profit,
            strike=t.strike,
            expiry=t.expiry,
            option_type=t.option_type,
            opened_at=t.opened_at,
        ))
    return out


@router.post("", response_model=TradeOut, status_code=201)
async def create_trade(payload: TradeCreate, db: AsyncSession = Depends(get_db)):
    current = get_current_price(payload.symbol.upper())
    notional = payload.entry_price * payload.quantity

    trade = Trade(
        symbol=payload.symbol.upper(),
        asset_class=payload.asset_class,
        direction=payload.direction,
        entry_price=payload.entry_price,
        quantity=payload.quantity,
        current_price=current,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        notional=notional,
        strike=payload.strike,
        expiry=payload.expiry,
        option_type=payload.option_type,
    )
    db.add(trade)
    await db.commit()
    await db.refresh(trade)

    return TradeOut(
        id=trade.id,
        symbol=trade.symbol,
        asset_class=trade.asset_class.value,
        direction=trade.direction.value,
        status=trade.status.value,
        entry_price=trade.entry_price,
        current_price=trade.current_price,
        exit_price=trade.exit_price,
        quantity=trade.quantity,
        pnl=trade.pnl,
        pnl_pct=trade.pnl_pct,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        strike=trade.strike,
        expiry=trade.expiry,
        option_type=trade.option_type,
        opened_at=trade.opened_at,
    )


@router.get("/{trade_id}", response_model=TradeOut)
async def get_trade(trade_id: int, db: AsyncSession = Depends(get_db)):
    trade = await db.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    if trade.status == TradeStatus.OPEN:
        trade.current_price = get_current_price(trade.symbol)
    return TradeOut(
        id=trade.id, symbol=trade.symbol, asset_class=trade.asset_class.value,
        direction=trade.direction.value, status=trade.status.value,
        entry_price=trade.entry_price, current_price=trade.current_price,
        exit_price=trade.exit_price, quantity=trade.quantity,
        pnl=trade.pnl, pnl_pct=trade.pnl_pct,
        stop_loss=trade.stop_loss, take_profit=trade.take_profit,
        strike=trade.strike, expiry=trade.expiry, option_type=trade.option_type,
        opened_at=trade.opened_at,
    )


@router.patch("/{trade_id}", response_model=TradeOut)
async def update_trade(trade_id: int, payload: TradeUpdate, db: AsyncSession = Depends(get_db)):
    trade = await db.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    for field, val in payload.model_dump(exclude_none=True).items():
        setattr(trade, field, val)

    if payload.status == TradeStatus.CLOSED and not trade.closed_at:
        trade.closed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(trade)
    return TradeOut(
        id=trade.id, symbol=trade.symbol, asset_class=trade.asset_class.value,
        direction=trade.direction.value, status=trade.status.value,
        entry_price=trade.entry_price, current_price=trade.current_price,
        exit_price=trade.exit_price, quantity=trade.quantity,
        pnl=trade.pnl, pnl_pct=trade.pnl_pct,
        stop_loss=trade.stop_loss, take_profit=trade.take_profit,
        strike=trade.strike, expiry=trade.expiry, option_type=trade.option_type,
        opened_at=trade.opened_at,
    )


@router.delete("/{trade_id}", status_code=204)
async def delete_trade(trade_id: int, db: AsyncSession = Depends(get_db)):
    trade = await db.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    await db.delete(trade)
    await db.commit()
