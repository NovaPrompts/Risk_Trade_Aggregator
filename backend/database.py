from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Enum as SAEnum
from datetime import datetime, timezone
from typing import Optional, List
import enum

from .config import settings


# ─── Engine ───────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ─── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── Enums ────────────────────────────────────────────────────────────────────

class TradeDirection(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class AssetClass(str, enum.Enum):
    EQUITY = "equity"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURE = "future"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


# ─── Models ───────────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    asset_class: Mapped[AssetClass] = mapped_column(SAEnum(AssetClass), nullable=False)
    direction: Mapped[TradeDirection] = mapped_column(SAEnum(TradeDirection), nullable=False)
    status: Mapped[TradeStatus] = mapped_column(SAEnum(TradeStatus), default=TradeStatus.OPEN)

    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Options-specific fields
    strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expiry: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    option_type: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)  # call/put

    # Risk fields
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notional: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    risk_snapshots: Mapped[List["RiskSnapshot"]] = relationship(
        back_populates="trade", cascade="all, delete-orphan"
    )

    @property
    def pnl(self) -> float:
        price = self.exit_price or self.current_price or self.entry_price
        raw = (price - self.entry_price) * self.quantity
        return raw if self.direction == TradeDirection.LONG else -raw

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        price = self.exit_price or self.current_price or self.entry_price
        return ((price - self.entry_price) / self.entry_price) * 100


class RiskSnapshot(Base):
    __tablename__ = "risk_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id"), nullable=False, index=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    delta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gamma: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    theta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vega: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rho: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    iv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    var_95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    trade: Mapped["Trade"] = relationship(back_populates="risk_snapshots")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[Optional[int]] = mapped_column(ForeignKey("trades.id"), nullable=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)  # info / warn / critical
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# ─── Init ─────────────────────────────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
