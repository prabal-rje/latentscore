import numpy as np
import pytest

import latentscore as ls
import latentscore.main as main
from latentscore.config import MusicConfig, SynthConfig


class _StubModel:
    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig()


def _stub_assemble(
    config: SynthConfig,
    duration: float = 16.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
    t_offset: float = 0.0,
) -> np.ndarray:
    _ = config
    _ = normalize
    _ = rng
    _ = t_offset
    size = max(1, int(ls.SAMPLE_RATE * duration))
    return np.zeros(size, dtype=np.float32)


def test_stream_accepts_multiple_vibes_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "assemble", _stub_assemble)
    stream = ls.stream(
        ["one", "two"],
        duration=0.2,
        transition=0.0,
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
        t_offset: float = 0.0,
    ) -> np.ndarray:
        called["normalize"] = normalize
        return _stub_assemble(
            config, duration=duration, normalize=normalize, rng=rng, t_offset=t_offset
        )

    monkeypatch.setattr(main, "assemble", _capture_assemble)
    stream = ls.stream(
        "test",
        duration=0.2,
        transition=0.0,
        model=_StubModel(),
        hooks=ls.StreamHooks(),
    )
    list(stream)
    assert called["normalize"] is False
