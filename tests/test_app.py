import asyncio

import pytest

from latentscore.app import demo_run
from latentscore.loop import run


async def _current_loop_name() -> str:
    loop = asyncio.get_running_loop()
    return type(loop).__name__


def test_run_installs_uvloop() -> None:
    loop_name = run(_current_loop_name())
    assert "uvloop" in loop_name.lower()


@pytest.mark.asyncio
async def test_demo_run_defaults() -> None:
    assert await demo_run() == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_demo_run_custom_values() -> None:
    assert await demo_run([10, 20]) == pytest.approx(15.0)
