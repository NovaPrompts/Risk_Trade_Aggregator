from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from ..data.simulator import get_ohlcv, get_options_chain, get_market_snapshot, get_current_price
from ..streaming.queue import sse_stream, CHANNEL_PRICES

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/snapshot")
async def market_snapshot():
    """Current prices + % change for all tracked symbols."""
    return get_market_snapshot()


@router.get("/price/{symbol}")
async def price(symbol: str):
    return {"symbol": symbol.upper(), "price": get_current_price(symbol.upper())}


@router.get("/ohlcv/{symbol}")
async def ohlcv(symbol: str, bars: int = Query(default=60, ge=1, le=500)):
    return {"symbol": symbol.upper(), "bars": get_ohlcv(symbol.upper(), bars=bars)}


@router.get("/options/{symbol}")
async def options_chain(
    symbol: str,
    expiries: int = Query(default=3, ge=1, le=5),
    strikes: int = Query(default=7, ge=3, le=15),
    r: float = Query(default=0.05, ge=0.0, le=0.20),
):
    """Full options chain with Greeks for a symbol."""
    chain = get_options_chain(symbol.upper(), expiries=expiries, strikes_per_expiry=strikes, r=r)
    return {"symbol": symbol.upper(), "contracts": chain}


@router.get("/stream/prices")
async def stream_prices():
    """SSE stream of real-time price ticks for all symbols."""
    return StreamingResponse(
        sse_stream(CHANNEL_PRICES),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
