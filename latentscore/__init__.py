from __future__ import annotations

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
    "StereoLabel",
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
    "live",
    "astream",
    "astream_raw",
    "render",
    "render_raw",
    "save_wav",
    "stream",
    "stream_raw",
    "stream_configs",
    "stream_texts",
    "stream_updates",
]

__version__ = "0.1.2"

_configure_logging()
del _configure_logging
