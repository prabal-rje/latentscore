"""Async resilience helpers (retry, rate limiting, concurrency, EWMA error tracking)."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

_LOGGER = logging.getLogger("data_work.resilience")

T = TypeVar("T")


class AsyncRateLimiter:
    def __init__(self, max_per_second: float) -> None:
        self._max_per_second = max_per_second
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        if self._max_per_second <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                await asyncio.sleep(self._next_allowed - now)
            interval = 1.0 / self._max_per_second
            self._next_allowed = max(now, self._next_allowed) + interval


def rate_limited(
    limiter: AsyncRateLimiter,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_semaphore(
    semaphore: asyncio.Semaphore,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def retry_async(
    *,
    max_retries: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    log = logger or _LOGGER

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1
                    log.warning(
                        "Retryable failure on attempt %s/%s: %s",
                        attempt,
                        max_retries,
                        exc,
                        exc_info=True,
                    )
                    if attempt > max_retries:
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


class EWMAErrorTracker:
    def __init__(self, *, alpha: float, min_samples: int = 1) -> None:
        self._alpha = alpha
        self._min_samples = min_samples
        self._error_rate: float | None = None
        self._total = 0

    def update(self, *, success: bool) -> float:
        error_value = 0.0 if success else 1.0
        if self._error_rate is None:
            self._error_rate = error_value
        else:
            self._error_rate = (self._alpha * error_value) + ((1 - self._alpha) * self._error_rate)
        self._total += 1
        return self._error_rate

    @property
    def error_rate(self) -> float:
        return self._error_rate or 0.0

    @property
    def total(self) -> int:
        return self._total

    def threshold_reached(self, threshold: float) -> bool:
        if self._total < self._min_samples:
            return False
        return self.error_rate > threshold
