from __future__ import annotations

import sys as _sys
import warnings as _warnings
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

# Soft Python-version check: latentscore is tested on 3.11-3.12 only.
# Other versions (3.10 below, 3.13+ above) usually work but may surface
# dependency-resolution or behaviour quirks. We emit a single UserWarning
# at import time instead of failing — users on bleeding-edge / older
# Pythons get the latest release rather than being silently downgraded
# by pip to a years-old version that happens to match their interpreter.
_v = _sys.version_info
if (_v.major, _v.minor) < (3, 11) or (_v.major, _v.minor) > (3, 12):
    _warnings.warn(
        f"latentscore is tested on Python 3.11-3.12 only. You're running "
        f"Python {_v.major}.{_v.minor}.{_v.micro} — most things should still "
        f"work, but you may hit dependency-resolution or behaviour quirks. "
        f'Silence this warning with `warnings.filterwarnings("ignore", '
        f'message="latentscore is tested")`.',
        UserWarning,
        stacklevel=2,
    )

_configure_logging()
del _configure_logging
