from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from latentscore import (
    SAMPLE_RATE,
    MusicConfig,
    MusicConfigUpdate,
    Streamable,
    StreamHooks,
    astream,
    render,
    save_wav,
    stream,
    stream_texts,
)
from latentscore.errors import InvalidConfigError


class DummyModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="medium", brightness="bright")


def test_stream_rejects_raw_string() -> None:
    with pytest.raises(InvalidConfigError):
        list(stream("warm sunrise", model=DummyModel()))


def test_stream_streamable_items() -> None:
    chunks = list(
        stream(
            [Streamable(content="warm sunrise", duration=0.04, transition_duration=0.0)],
            chunk_seconds=0.02,
            model=DummyModel(),
        )
    )
    assert len(chunks) >= 1
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)


def test_render_audio_contract() -> None:
    audio = render(
        "warm sunrise",
        duration=0.02,
        model=DummyModel(),
    )
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert float(np.max(np.abs(audio))) <= 1.0


def test_stream_chunk_length_matches_sample_rate() -> None:
    chunk_seconds = 0.02
    chunks = stream(
        [Streamable(content=MusicConfigUpdate(tempo="slow"), duration=0.02)],
        chunk_seconds=chunk_seconds,
        model=DummyModel(),
    )
    chunk = next(iter(chunks))
    expected = int(round(SAMPLE_RATE * chunk_seconds))
    assert abs(len(chunk) - expected) <= 1


def test_save_wav_writes_file(tmp_path: Path) -> None:
    audio = render("gentle", duration=0.02, model=DummyModel())
    target = tmp_path / "out.wav"
    save_wav(str(target), audio)
    assert target.exists()


def test_render_requires_config_models() -> None:
    with pytest.raises(TypeError):
        render("vibe", tempo=0.4)  # type: ignore[call-arg]


def test_stream_texts_wraps_prompts() -> None:
    chunks = list(
        stream_texts(
            ["warm sunrise"],
            duration=0.04,
            chunk_seconds=0.02,
            model=DummyModel(),
        )
    )
    assert len(chunks) >= 1


def test_streamable_accepts_fallback_and_preview() -> None:
    item = Streamable(
        content="warm sunrise",
        duration=0.02,
        transition_duration=0.0,
        fallback="keep_last",
        preview_policy="embedding",
    )
    assert item.fallback == "keep_last"
    assert item.preview_policy == "embedding"


class SlowModel:
    def __init__(self) -> None:
        self.ready = asyncio.Event()

    async def generate(self, vibe: str) -> MusicConfig:
        await self.ready.wait()
        return MusicConfig(tempo="medium", brightness="bright")


class FastModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="slow", brightness="dark")


@pytest.mark.asyncio
async def test_astream_preview_yields_before_llm_ready() -> None:
    slow = SlowModel()
    fast = FastModel()
    items = [Streamable(content="warm sunrise", duration=0.04, transition_duration=0.0)]

    async def first_chunk() -> np.ndarray:
        async for chunk in astream(
            items,
            chunk_seconds=0.02,
            model=slow,
            preview_policy="embedding",
            fallback_model=fast,
        ):
            return chunk
        raise AssertionError("no chunks")

    chunk = await asyncio.wait_for(first_chunk(), timeout=0.5)
    assert isinstance(chunk, np.ndarray)
    slow.ready.set()


class ErrorModel:
    async def generate(self, vibe: str) -> MusicConfig:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_astream_fallback_on_error_keeps_streaming() -> None:
    items = [Streamable(content="warm sunrise", duration=0.02, transition_duration=0.0)]
    chunks = [
        chunk
        async for chunk in astream(
            items,
            chunk_seconds=0.02,
            model=ErrorModel(),
            fallback="embedding",
            fallback_model=FastModel(),
        )
    ]
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_astream_hooks_fire() -> None:
    events: list[str] = []
    hooks = StreamHooks(on_event=lambda event: events.append(event.kind))
    items = [Streamable(content=MusicConfig(), duration=0.02, transition_duration=0.0)]
    chunks = [
        chunk
        async for chunk in astream(
            items,
            chunk_seconds=0.02,
            model=DummyModel(),
            hooks=hooks,
        )
    ]
    assert len(chunks) >= 1
    assert "stream_start" in events
    assert "first_audio_chunk" in events
    assert "stream_end" in events
