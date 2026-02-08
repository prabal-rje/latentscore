from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from latentscore import (
    SAMPLE_RATE,
    MusicConfig,
    MusicConfigUpdate,
    RenderHooks,
    Streamable,
    StreamHooks,
    astream_raw,
    render_raw,
    save_wav,
    stream_raw,
    stream_texts,
)
from latentscore.errors import InvalidConfigError


class DummyModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="medium", brightness="bright")


class ClosableModel:
    def __init__(self) -> None:
        self.closed = False

    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="medium", brightness="bright")

    async def aclose(self) -> None:
        self.closed = True


class SyncClosableModel:
    def __init__(self) -> None:
        self.closed = False

    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="medium", brightness="bright")

    def close(self) -> None:
        self.closed = True


def test_raw_functions_still_exposed() -> None:
    assert callable(render_raw)
    assert callable(stream_raw)
    assert callable(astream_raw)


def test_stream_rejects_raw_string() -> None:
    with pytest.raises(InvalidConfigError):
        list(stream_raw("warm sunrise", model=DummyModel()))


def test_stream_streamable_items() -> None:
    chunks = list(
        stream_raw(
            [Streamable(content="warm sunrise", duration=1.0, transition_duration=0.0)],
            model=DummyModel(),
        )
    )
    assert len(chunks) >= 1
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)


def test_render_audio_contract() -> None:
    audio = render_raw(
        "warm sunrise",
        duration=0.02,
        model=DummyModel(),
    )
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert float(np.max(np.abs(audio))) <= 1.0


def test_render_does_not_close_user_model() -> None:
    model = ClosableModel()
    _ = render_raw("warm sunrise", duration=0.02, model=model)
    assert model.closed is False


def test_render_does_not_close_user_sync_model() -> None:
    model = SyncClosableModel()
    _ = render_raw("warm sunrise", duration=0.02, model=model)
    assert model.closed is False


def test_render_closes_external_model_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ClosableModel()

    def fake_build_external_adapter(*_: object, **__: object) -> ClosableModel:
        return model

    monkeypatch.setattr(
        "latentscore.models._build_external_adapter",
        fake_build_external_adapter,
    )

    _ = render_raw("warm sunrise", duration=0.02, model="external:fake-model")
    assert model.closed is True


def test_render_hooks_fire() -> None:
    events: list[str] = []
    hooks = RenderHooks(
        on_start=lambda: events.append("start"),
        on_end=lambda: events.append("end"),
    )
    _ = render_raw(
        "warm sunrise",
        duration=0.02,
        model=DummyModel(),
        hooks=hooks,
    )
    assert events == ["start", "end"]


def test_stream_chunk_length_matches_sample_rate() -> None:
    chunk_seconds = 1.0
    chunks = stream_raw(
        [Streamable(content=MusicConfigUpdate(tempo="slow"), duration=1.0)],
        model=DummyModel(),
    )
    chunk = next(iter(chunks))
    expected = int(round(SAMPLE_RATE * chunk_seconds))
    assert abs(len(chunk) - expected) <= 1


def test_save_wav_writes_file(tmp_path: Path) -> None:
    audio = render_raw("gentle", duration=0.02, model=DummyModel())
    target = tmp_path / "out.wav"
    save_wav(str(target), audio)
    assert target.exists()


def test_render_requires_config_models() -> None:
    with pytest.raises(TypeError):
        render_raw("vibe", tempo=0.4)  # type: ignore[call-arg]


def test_stream_texts_wraps_prompts() -> None:
    chunks = list(
        stream_texts(
            ["warm sunrise"],
            duration=1.0,
            model=DummyModel(),
        )
    )
    assert len(chunks) >= 1


def test_streamable_accepts_fallback() -> None:
    item = Streamable(
        content="warm sunrise",
        duration=0.02,
        transition_duration=0.0,
        fallback="keep_last",
    )
    assert item.fallback == "keep_last"


class FastModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="slow", brightness="dark")


@pytest.mark.asyncio
async def test_astream_does_not_close_user_model() -> None:
    model = ClosableModel()
    items = [Streamable(content="warm sunrise", duration=0.04, transition_duration=0.0)]
    _ = [
        chunk
        async for chunk in astream_raw(
            items,
            model=model,
        )
    ]
    assert model.closed is False


class ErrorModel:
    async def generate(self, vibe: str) -> MusicConfig:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_astream_fallback_on_error_keeps_streaming() -> None:
    items = [Streamable(content="warm sunrise", duration=0.02, transition_duration=0.0)]
    chunks = [
        chunk
        async for chunk in astream_raw(
            items,
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
        async for chunk in astream_raw(
            items,
            model=DummyModel(),
            hooks=hooks,
        )
    ]
    assert len(chunks) >= 1
    assert "stream_start" in events
    assert "first_audio_chunk" in events
    assert "stream_end" in events
