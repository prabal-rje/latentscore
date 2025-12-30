from __future__ import annotations

import asyncio
from typing import Coroutine, TypeVar

import uvloop

T = TypeVar("T")


def install_uvloop_policy() -> None:
    """Install uvloop as the default event loop policy."""
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def run(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine with uvloop installed."""
    install_uvloop_policy()
    return asyncio.run(coro)
