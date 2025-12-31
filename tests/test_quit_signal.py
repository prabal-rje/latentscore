import asyncio

import pytest

from app.quit_signal import wait_for_signal_clear


@pytest.mark.asyncio
async def test_wait_returns_when_signal_missing(tmp_path, monkeypatch) -> None:
    signal_path = tmp_path / "quit.signal"
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await wait_for_signal_clear(signal_path, timeout=1.0, interval=0.05)

    assert sleeps == []


@pytest.mark.asyncio
async def test_wait_returns_after_signal_clears(tmp_path, monkeypatch) -> None:
    signal_path = tmp_path / "quit.signal"
    signal_path.write_text("quit", encoding="utf-8")
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if signal_path.exists():
            signal_path.unlink()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await wait_for_signal_clear(signal_path, timeout=1.0, interval=0.05)

    assert sleeps == [0.05]
