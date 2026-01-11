import logging

import numpy as np
import pytest

import latentscore as ls
import latentscore.main as main
from latentscore.config import MusicConfig, SynthConfig
from latentscore.dx import Audio, Playlist, Track


class _StubModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig()


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


def test_audio_normalizes_samples() -> None:
    audio = Audio(samples=np.array([2.0, -2.0], dtype=np.float32))
    assert float(audio.samples.max()) <= 1.0


def test_track_to_streamable() -> None:
    track = Track(content="warm sunrise", duration=1.5, transition=0.2)
    streamable = track.to_streamable()
    assert streamable.duration == 1.5
    assert streamable.transition_duration == 0.2


def test_playlist_roundtrip() -> None:
    playlist = Playlist(tracks=(Track(content="warm sunrise"),))
    assert len(playlist.tracks) == 1


def test_stream_warns_on_aggressive_timing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "assemble", _stub_assemble)
    caplog.set_level(logging.WARNING, logger="latentscore.main")
    stream = ls.stream(
        "vibe",
        duration=0.4,
        transition=0.5,
        chunk_seconds=0.1,
        model=_StubModel(),
        hooks=ls.StreamHooks(),
    )
    list(stream)
    assert "Track duration" in caplog.text
    assert "Transition duration" in caplog.text
