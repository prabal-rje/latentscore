import asyncio
import threading
from contextlib import aclosing

import numpy as np
import pytest

import latentscore as ls
import latentscore.main as main
from latentscore.config import MusicConfig, SynthConfig


class _StubModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig()


class _SlowWarmupModel:
    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def warmup(self) -> None:
        self._gate.wait()

    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig()


class _GateModel:
    def __init__(self, gate: asyncio.Event, config: MusicConfig) -> None:
        self._gate = gate
        self._config = config

    async def generate(self, vibe: str) -> MusicConfig:
        await self._gate.wait()
        return self._config


class _ImmediateModel:
    def __init__(self, config: MusicConfig) -> None:
        self._config = config

    async def generate(self, vibe: str) -> MusicConfig:
        return self._config


def _stub_assemble(
    config: SynthConfig,
    duration: float = 16.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    _ = config
    _ = normalize
    _ = rng
    size = max(1, int(ls.SAMPLE_RATE * duration))
    return np.zeros(size, dtype=np.float32)


def test_stream_accepts_multiple_vibes_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "assemble", _stub_assemble)
    stream = ls.stream(
        ["one", "two"],
        duration=0.2,
        transition=0.0,
        chunk_seconds=0.1,
        model=_StubModel(),
        hooks=ls.StreamHooks(),
    )
    chunks = list(stream)
    assert chunks


def test_stream_chunk_assemble_not_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool | None] = {"normalize": None}

    def _capture_assemble(
        config: SynthConfig,
        duration: float = 16.0,
        normalize: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        called["normalize"] = normalize
        return _stub_assemble(config, duration=duration, normalize=normalize, rng=rng)

    monkeypatch.setattr(main, "assemble", _capture_assemble)
    stream = ls.stream(
        "test",
        duration=0.2,
        transition=0.0,
        chunk_seconds=0.1,
        model=_StubModel(),
        hooks=ls.StreamHooks(),
    )
    list(stream)
    assert called["normalize"] is False


@pytest.mark.asyncio
async def test_preview_streams_before_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "assemble", _stub_assemble)
    gate = threading.Event()
    model = _SlowWarmupModel(gate)
    fallback = _StubModel()
    items = [
        ls.Streamable(
            content="late night neon",
            duration=0.2,
            transition_duration=0.0,
        )
    ]

    async with aclosing(
        ls.astream_raw(
            items,
            chunk_seconds=0.1,
            model=model,
            fallback_model=fallback,
            preview=True,
            hooks=ls.StreamHooks(),
        )
    ) as stream:
        try:
            iterator = stream.__aiter__()
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=0.5)
        finally:
            gate.set()
    assert chunk.size > 0


@pytest.mark.asyncio
async def test_preview_prefers_primary_when_fallback_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_gate = asyncio.Event()
    primary = _ImmediateModel(MusicConfig(tempo="fast"))
    fallback = _GateModel(fallback_gate, MusicConfig(tempo="slow"))
    fast_tempo = MusicConfig(tempo="fast").to_internal().tempo
    slow_tempo = MusicConfig(tempo="slow").to_internal().tempo
    configs: list[SynthConfig] = []

    def _capture_assemble(
        config: SynthConfig,
        duration: float = 16.0,
        normalize: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        configs.append(config)
        return _stub_assemble(config, duration=duration, normalize=normalize, rng=rng)

    monkeypatch.setattr(main, "assemble", _capture_assemble)
    items = [
        ls.Streamable(
            content="late night neon",
            duration=0.2,
            transition_duration=0.0,
        )
    ]

    async with aclosing(
        ls.astream_raw(
            items,
            chunk_seconds=0.1,
            model=primary,
            fallback_model=fallback,
            preview=True,
            hooks=ls.StreamHooks(),
        )
    ) as stream:
        iterator = stream.__aiter__()
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=0.5)
        finally:
            fallback_gate.set()

    assert chunk.size > 0
    assert configs
    assert configs[0].tempo == fast_tempo
    assert configs[0].tempo != slow_tempo


@pytest.mark.asyncio
async def test_preview_event_only_when_preview_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "assemble", _stub_assemble)
    fallback_gate = asyncio.Event()
    primary = _ImmediateModel(MusicConfig())
    fallback = _GateModel(fallback_gate, MusicConfig())
    events: list[str] = []
    items = [
        ls.Streamable(
            content="late night neon",
            duration=0.2,
            transition_duration=0.0,
        )
    ]

    async with aclosing(
        ls.astream_raw(
            items,
            chunk_seconds=0.1,
            model=primary,
            fallback_model=fallback,
            preview=True,
            hooks=ls.StreamHooks(on_event=lambda event: events.append(event.kind)),
        )
    ) as stream:
        iterator = stream.__aiter__()
        try:
            _ = await asyncio.wait_for(iterator.__anext__(), timeout=0.5)
        finally:
            fallback_gate.set()

    assert "item_preview_start" not in events
