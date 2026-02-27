"""
Event queue with in-memory fallback.
If REDIS_URL is reachable, uses Redis pub/sub.
Otherwise, uses asyncio.Queue — zero-dependency mode.
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional, Dict, List

logger = logging.getLogger(__name__)

# ─── In-Memory Bus ────────────────────────────────────────────────────────────

class InMemoryBus:
    """Fan-out pub/sub using asyncio.Queue per subscriber."""

    def __init__(self):
        self._channels: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, channel: str, message: dict):
        async with self._lock:
            queues = self._channels.get(channel, [])
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass  # drop slow consumer

    async def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._channels.setdefault(channel, []).append(q)
        return q

    async def unsubscribe(self, channel: str, q: asyncio.Queue):
        async with self._lock:
            subscribers = self._channels.get(channel, [])
            if q in subscribers:
                subscribers.remove(q)


# ─── Redis Bus (optional) ─────────────────────────────────────────────────────

class RedisBus:
    def __init__(self, redis_url: str):
        self._url = redis_url
        self._redis = None
        self._available = False

    async def connect(self) -> bool:
        try:
            import redis.asyncio as aioredis  # type: ignore
            client = aioredis.from_url(self._url, decode_responses=True, socket_connect_timeout=2)
            await client.ping()
            self._redis = client
            self._available = True
            logger.info("Redis connected: %s", self._url)
            return True
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-memory queue", e)
            return False

    async def publish(self, channel: str, message: dict):
        if self._redis:
            await self._redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str) -> AsyncGenerator[dict, None]:
        if not self._redis:
            return
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                yield json.loads(msg["data"])


# ─── Unified EventBus ─────────────────────────────────────────────────────────

_mem_bus = InMemoryBus()
_redis_bus: Optional[RedisBus] = None
_use_redis = False


async def init_event_bus(redis_url: Optional[str] = None):
    global _redis_bus, _use_redis
    if redis_url:
        _redis_bus = RedisBus(redis_url)
        _use_redis = await _redis_bus.connect()
    if not _use_redis:
        logger.info("Event bus: in-memory mode")


async def publish(channel: str, event: dict):
    """Publish an event to a channel."""
    if _use_redis and _redis_bus:
        await _redis_bus.publish(channel, event)
    else:
        await _mem_bus.publish(channel, event)


async def sse_stream(channel: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields Server-Sent Event formatted strings
    for a given channel subscription.
    """
    q = await _mem_bus.subscribe(channel)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                # SSE keepalive comment
                yield ": keepalive\n\n"
    finally:
        await _mem_bus.unsubscribe(channel, q)


# ─── Channel Names ─────────────────────────────────────────────────────────────

CHANNEL_PRICES = "prices"
CHANNEL_ALERTS = "alerts"
CHANNEL_PORTFOLIO = "portfolio"
