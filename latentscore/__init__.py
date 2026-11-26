from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

from common.music_schema import Palette, PaletteColor

from .audio import SAMPLE_RATE
from .config import (
    AccentStyle,
    AttackStyle,
    BassStyle,
    BrightnessLabel,
    DensityLevel,
    EchoLabel,
    GenerateResult,
    GrainStyle,
    HumanFeelLabel,
    MelodyStyle,
    ModeName,
    MotionLabel,
    MusicConfig,
    MusicConfigUpdate,
    PadStyle,
    RhythmStyle,
    RootNote,
    SpaceLabel,
    Step,
    StereoLabel,
    TempoLabel,
    TextureStyle,
)
from .dx import Audio, AudioStream, LiveStream, Playlist, Track, arender, live, render, stream
from .logging_utils import configure_logging as _configure_logging
from .main import (
    FallbackInput,
    FirstAudioSpinner,
    ModelLoadRole,
    RenderHooks,
    Streamable,
    StreamEvent,
    StreamHooks,
    astream,
    save_wav,
    stream_configs,
    stream_texts,
    stream_updates,
)
from .main import (
    astream as astream_raw,
)
from .main import (
    render as render_raw,
)
from .main import (
    stream as stream_raw,
)
from .models import ExternalModelSpec, ModelForGeneratingMusicConfig, ModelSpec
from .prefetch import prefetch
from .score import Score, compute_bar_aligned_duration, fade_into_nothingness, mood_from_text

__all__ = [
    "SAMPLE_RATE",
    "AccentStyle",
    "AttackStyle",
    "BassStyle",
    "BrightnessLabel",
    "DensityLevel",
    "EchoLabel",
    "GenerateResult",
    "GrainStyle",
    "HumanFeelLabel",
    "MelodyStyle",
    "ModeName",
    "MotionLabel",
    "MusicConfig",
    "MusicConfigUpdate",
    "ModelSpec",
    "Palette",
    "PaletteColor",
    "ModelForGeneratingMusicConfig",
    "ExternalModelSpec",
    "PadStyle",
    "RhythmStyle",
    "RootNote",
    "SpaceLabel",
    "Step",
    "StereoLabel",
    "Score",
    "StreamEvent",
    "StreamHooks",
    "Streamable",
    "TextureStyle",
    "TempoLabel",
    "FallbackInput",
    "FirstAudioSpinner",
    "ModelLoadRole",
    "RenderHooks",
    "Audio",
    "AudioStream",
    "LiveStream",
    "Playlist",
    "Track",
    "arender",
    "compute_bar_aligned_duration",
    "fade_into_nothingness",
    "live",
    "mood_from_text",
    "astream",
    "astream_raw",
    "prefetch",
    "render",
    "render_raw",
    "save_wav",
    "stream",
    "stream_raw",
    "stream_configs",
    "stream_texts",
    "stream_updates",
]

try:
    __version__ = _pkg_version("latentscore")
except _PackageNotFoundError:  # pragma: no cover - editable/dev fallback
    __version__ = "0.0.0+unknown"

_configure_logging()
del _configure_logging
