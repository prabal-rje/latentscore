import asyncio
from pathlib import Path

from data_work.llm_cache import SqliteCache


def test_sqlite_cache_roundtrip(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "cache.sqlite")
    asyncio.run(cache.initialize())
    asyncio.run(cache.set("key", {"payload": 1}, {"value": 2}))
    cached = asyncio.run(cache.get("key"))
    assert cached == {"value": 2}
