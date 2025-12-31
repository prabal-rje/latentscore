from __future__ import annotations

import asyncio
import sys
from typing import Any, Coroutine, TypeVar

uvloop: Any | None
if sys.platform.startswith("win"):
    uvloop = None
else:
    try:
        import uvloop as _uvloop
    except ImportError:
        uvloop = None
    else:
        uvloop = _uvloop

T = TypeVar("T")


def install_uvloop_policy() -> None:
    """Install uvloop as the default event loop policy when available."""
    if uvloop is None:
        return
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def run(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine with uvloop installed."""
    install_uvloop_policy()
    return asyncio.run(coro)
