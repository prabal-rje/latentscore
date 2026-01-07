from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentscore import (
    SAMPLE_RATE,
    MusicConfig,
    MusicConfigUpdate,
    Streamable,
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
