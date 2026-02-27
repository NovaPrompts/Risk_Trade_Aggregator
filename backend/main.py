import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .streaming.queue import init_event_bus
from .agents.risk_monitor import start_agents
from .routes import market, trades, risk

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Initialising database…")
    await init_db()

    logger.info("Initialising event bus…")
    redis_url = settings.redis_url if not settings.use_simulated_data else None
    await init_event_bus(redis_url)

    logger.info("Starting background agents…")
    await start_agents()

    mode = "SIMULATED" if settings.use_simulated_data else "LIVE"
    logger.info("Finance Aggregator ready — market data mode: %s", mode)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down…")


app = FastAPI(
    title="Alternative Finance Risk & Trade Aggregator",
    description=(
        "Real-time options Greeks, portfolio VaR, theta decay visualisation, "
        "and multi-asset trade management. All risk math is deterministic numpy/scipy — "
        "no LLM in the critical path."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routers ──────────────────────────────────────────────────────────────

app.include_router(market.router, prefix="/api/v1")
app.include_router(trades.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")

# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Dashboard not found — run the frontend build</h1>", status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
