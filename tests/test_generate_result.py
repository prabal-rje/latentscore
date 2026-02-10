"""Tests for GenerateResult metadata flow through render → Audio."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import latentscore as ls
import latentscore.main as main
from common.music_schema import Palette, PaletteColor
from latentscore.config import GenerateResult, MusicConfig, SynthConfig
from latentscore.dx import Audio

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_palette() -> Palette:
    """Build a minimal valid palette (5 colors)."""
    colors = [
        PaletteColor(hex="#1a1a2e", weight="xxl"),
        PaletteColor(hex="#16213e", weight="xl"),
        PaletteColor(hex="#0f3460", weight="lg"),
        PaletteColor(hex="#e94560", weight="md"),
        PaletteColor(hex="#533483", weight="sm"),
    ]
    return Palette(colors=colors)


def _stub_assemble(
    config: SynthConfig,
    duration: float = 16.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
    t_offset: float = 0.0,
) -> NDArray[np.float32]:
    _ = config, normalize, rng, t_offset
    size = max(1, int(ls.SAMPLE_RATE * duration))
    return np.zeros(size, dtype=np.float32)


class _StubModelReturningConfig:
    """Model that returns plain MusicConfig (like fast model)."""

    async def generate(self, vibe: str) -> MusicConfig:
        return MusicConfig(tempo="slow", brightness="dark")


class _StubModelReturningResult:
    """Model that returns GenerateResult (like external LLM)."""

    async def generate(self, vibe: str) -> GenerateResult:
        return GenerateResult(
            config=MusicConfig(tempo="fast", brightness="bright"),
            title="Neon Rain Drift",
            thinking="cyberpunk + rain = dark, fast, electronic",
            palettes=(_make_palette(), _make_palette(), _make_palette()),
        )


# ---------------------------------------------------------------------------
# GenerateResult model tests
# ---------------------------------------------------------------------------


class TestGenerateResult:
    def test_construction_with_all_fields(self) -> None:
        result = GenerateResult(
            config=MusicConfig(),
            title="Test Title",
            thinking="Some reasoning",
            palettes=(_make_palette(), _make_palette(), _make_palette()),
        )
        assert result.config.tempo == "medium"
        assert result.title == "Test Title"
        assert result.thinking == "Some reasoning"
        assert len(result.palettes) == 3

    def test_construction_minimal(self) -> None:
        result = GenerateResult(config=MusicConfig())
        assert result.title is None
        assert result.thinking is None
        assert result.palettes == ()

    def test_config_is_required(self) -> None:
        with pytest.raises(Exception):
            GenerateResult()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Audio metadata tests
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    def test_audio_accepts_metadata(self) -> None:
        result = GenerateResult(
            config=MusicConfig(),
            title="Test",
            palettes=(_make_palette(), _make_palette(), _make_palette()),
        )
        audio = Audio(
            samples=np.zeros(100, dtype=np.float32),
            metadata=result,
        )
        assert audio.metadata is not None
        assert audio.metadata.title == "Test"
        assert audio.metadata.config.tempo == "medium"
        assert len(audio.metadata.palettes) == 3

    def test_audio_metadata_defaults_to_none(self) -> None:
        audio = Audio(samples=np.zeros(100, dtype=np.float32))
        assert audio.metadata is None


# ---------------------------------------------------------------------------
# Render integration tests (stub synth, real render path)
# ---------------------------------------------------------------------------


class TestRenderMetadataFlow:
    def test_render_with_plain_model_has_no_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "assemble", _stub_assemble)
        audio = ls.render(
            "warm sunset",
            duration=6.0,
            model=_StubModelReturningConfig(),
            hooks=ls.RenderHooks(),
        )
        assert audio.metadata is None

    def test_render_with_result_model_has_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "assemble", _stub_assemble)
        audio = ls.render(
            "cyberpunk rain",
            duration=6.0,
            model=_StubModelReturningResult(),
            hooks=ls.RenderHooks(),
        )
        assert audio.metadata is not None
        assert audio.metadata.title == "Neon Rain Drift"
        assert audio.metadata.thinking == "cyberpunk + rain = dark, fast, electronic"
        assert len(audio.metadata.palettes) == 3
        assert audio.metadata.config.tempo == "fast"
        assert audio.metadata.config.brightness == "bright"


# ---------------------------------------------------------------------------
# Public re-export tests
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_generate_result_importable_from_latentscore(self) -> None:
        from latentscore import GenerateResult as GR

        assert GR is GenerateResult

    def test_palette_importable_from_latentscore(self) -> None:
        from latentscore import Palette as P

        assert P is Palette

    def test_palette_color_importable_from_latentscore(self) -> None:
        from latentscore import PaletteColor as PC

        assert PC is PaletteColor
