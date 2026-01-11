"""SQLite cache for LLM responses."""

import functools
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import aiosqlite

T = TypeVar("T")


class SqliteCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._initialized = False

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def get(self, cache_key: str) -> dict[str, Any] | None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        response_text = row[0]
        return json.loads(response_text)

    async def set(self, cache_key: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO llm_cache (cache_key, payload, response, created_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (
                    cache_key,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(response, sort_keys=True, separators=(",", ":")),
                ),
            )
            await db.commit()


def cached_async(
    *,
    cache: SqliteCache,
    key_fn: Callable[..., tuple[str, dict[str, Any]]],
    serializer: Callable[[T], dict[str, Any]],
    deserializer: Callable[[dict[str, Any]], T],
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_key, payload = key_fn(*args, **kwargs)
            cached = await cache.get(cache_key)
            if cached is not None:
                return deserializer(cached)
            value: T = await func(*args, **kwargs)
            await cache.set(cache_key, payload, serializer(value))
            return value

        return wrapper

    return decorator
