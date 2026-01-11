import asyncio

import pytest

from data_work.resilience import EWMAErrorTracker, retry_async


async def _always_fail() -> None:
    raise RuntimeError("boom")


def test_ewma_error_threshold() -> None:
    tracker = EWMAErrorTracker(alpha=0.5, min_samples=1)
    tracker.update(success=False)
    tracker.update(success=False)
    assert tracker.error_rate > 0.25
    assert tracker.threshold_reached(0.25)


def test_retry_async_exhausts() -> None:
    wrapped = retry_async(max_retries=1)(_always_fail)
    with pytest.raises(RuntimeError):
        asyncio.run(wrapped())
