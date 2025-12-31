from __future__ import annotations

import asyncio
import time
from pathlib import Path


async def wait_for_signal_clear(path: Path, *, timeout: float, interval: float) -> None:
    if not path.exists() or timeout <= 0:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not path.exists():
            return
        await asyncio.sleep(interval)
