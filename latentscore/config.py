from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

from .errors import InvalidConfigError

_LOGGER = logging.getLogger("latentscore.config")

TempoLabel = Literal["very_slow", "slow", "medium", "fast", "very_fast"]
BrightnessLabel = Literal["very_dark", "dark", "medium", "bright", "very_bright"]
SpaceLabel = Literal["dry", "small", "medium", "large", "vast"]
MotionLabel = Literal["static", "slow", "medium", "fast", "chaotic"]
StereoLabel = Literal["mono", "narrow", "medium", "wide", "ultra_wide"]
EchoLabel = Literal["none", "subtle", "medium", "heavy", "infinite"]
HumanFeelLabel = Literal["robotic", "tight", "natural", "loose", "drunk"]
RootNote = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
ModeName = Literal["major", "minor", "dorian", "mixolydian"]
DensityLevel = Literal[2, 3, 4, 5, 6]
BassStyle = Literal[
    "drone",
    "sustained",
    "pulsing",
    "walking",
    "fifth_drone",
    "sub_pulse",
    "octave",
    "arp_bass",
]
PadStyle = Literal[
    "warm_slow",
    "dark_sustained",
    "cinematic",
    "thin_high",
    "ambient_drift",
    "stacked_fifths",
    "bright_open",
]
MelodyStyle = Literal[
    "contemplative",
    "rising",
    "falling",
    "minimal",
    "ornamental",
    "arp_melody",
    "contemplative_minor",
    "call_response",
    "heroic",
]
RhythmStyle = Literal[
    "none",
    "minimal",
    "heartbeat",
    "soft_four",
    "hats_only",
    "electronic",
    "kit_light",
    "kit_medium",
    "military",
    "tabla_essence",
    "brush",
]
TextureStyle = Literal[
    "none",
    "shimmer",
    "shimmer_slow",
    "vinyl_crackle",
    "breath",
    "stars",
    "glitch",
    "noise_wash",
    "crystal",
    "pad_whisper",
]
AccentStyle = Literal[
    "none",
    "bells",
    "pluck",
    "chime",
    "bells_dense",
    "blip",
    "blip_random",
    "brass_hit",
    "wind",
    "arp_accent",
    "piano_note",
]
AttackStyle = Literal["soft", "medium", "sharp"]
GrainStyle = Literal["clean", "warm", "gritty"]


T = TypeVar("T")

_TEMPO_VERY_SLOW = 0.15
_TEMPO_SLOW = 0.3
_TEMPO_MEDIUM = 0.5
_TEMPO_FAST = 0.7
_TEMPO_VERY_FAST = 0.9

_BRIGHTNESS_VERY_DARK = 0.1
_BRIGHTNESS_DARK = 0.3
_BRIGHTNESS_MEDIUM = 0.5
_BRIGHTNESS_BRIGHT = 0.7
_BRIGHTNESS_VERY_BRIGHT = 0.9

_SPACE_DRY = 0.1
_SPACE_SMALL = 0.3
_SPACE_MEDIUM = 0.5
_SPACE_LARGE = 0.7
_SPACE_VAST = 0.95

_MOTION_STATIC = 0.1
_MOTION_SLOW = 0.3
_MOTION_MEDIUM = 0.5
_MOTION_FAST = 0.7
_MOTION_CHAOTIC = 0.9

_STEREO_MONO = 0.0
_STEREO_NARROW = 0.25
_STEREO_MEDIUM = 0.5
_STEREO_WIDE = 0.75
_STEREO_ULTRA_WIDE = 1.0

_ECHO_NONE = 0.0
_ECHO_SUBTLE = 0.25
_ECHO_MEDIUM = 0.5
_ECHO_HEAVY = 0.75
_ECHO_INFINITE = 0.95

_HUMAN_ROBOTIC = 0.0
_HUMAN_TIGHT = 0.15
_HUMAN_NATURAL = 0.3
_HUMAN_LOOSE = 0.5
_HUMAN_DRUNK = 0.8


def _optional_map(value: T | None, mapper: Callable[[T], float]) -> float | None:
    match value:
        case None:
            return None
        case _:
            return mapper(value)


def tempo_to_float(value: TempoLabel) -> float:
    match value:
        case "very_slow":
            return _TEMPO_VERY_SLOW
        case "slow":
            return _TEMPO_SLOW
        case "medium":
            return _TEMPO_MEDIUM
        case "fast":
            return _TEMPO_FAST
        case "very_fast":
            return _TEMPO_VERY_FAST
        case _:
            raise InvalidConfigError(f"Unknown tempo label: {value!r}")


def brightness_to_float(value: BrightnessLabel) -> float:
    match value:
        case "very_dark":
            return _BRIGHTNESS_VERY_DARK
        case "dark":
            return _BRIGHTNESS_DARK
        case "medium":
            return _BRIGHTNESS_MEDIUM
        case "bright":
            return _BRIGHTNESS_BRIGHT
        case "very_bright":
            return _BRIGHTNESS_VERY_BRIGHT
        case _:
            raise InvalidConfigError(f"Unknown brightness label: {value!r}")


def space_to_float(value: SpaceLabel) -> float:
    match value:
        case "dry":
            return _SPACE_DRY
        case "small":
            return _SPACE_SMALL
        case "medium":
            return _SPACE_MEDIUM
        case "large":
            return _SPACE_LARGE
        case "vast":
            return _SPACE_VAST
        case _:
            raise InvalidConfigError(f"Unknown space label: {value!r}")


def motion_to_float(value: MotionLabel) -> float:
    match value:
        case "static":
            return _MOTION_STATIC
        case "slow":
            return _MOTION_SLOW
        case "medium":
            return _MOTION_MEDIUM
        case "fast":
            return _MOTION_FAST
        case "chaotic":
            return _MOTION_CHAOTIC
        case _:
            raise InvalidConfigError(f"Unknown motion label: {value!r}")


def stereo_to_float(value: StereoLabel) -> float:
    match value:
        case "mono":
            return _STEREO_MONO
        case "narrow":
            return _STEREO_NARROW
        case "medium":
            return _STEREO_MEDIUM
        case "wide":
            return _STEREO_WIDE
        case "ultra_wide":
            return _STEREO_ULTRA_WIDE
        case _:
            raise InvalidConfigError(f"Unknown stereo label: {value!r}")


def echo_to_float(value: EchoLabel) -> float:
    match value:
        case "none":
            return _ECHO_NONE
        case "subtle":
            return _ECHO_SUBTLE
        case "medium":
            return _ECHO_MEDIUM
        case "heavy":
            return _ECHO_HEAVY
        case "infinite":
            return _ECHO_INFINITE
        case _:
            raise InvalidConfigError(f"Unknown echo label: {value!r}")


def human_to_float(value: HumanFeelLabel) -> float:
    match value:
        case "robotic":
            return _HUMAN_ROBOTIC
        case "tight":
            return _HUMAN_TIGHT
        case "natural":
            return _HUMAN_NATURAL
        case "loose":
            return _HUMAN_LOOSE
        case "drunk":
            return _HUMAN_DRUNK
        case _:
            raise InvalidConfigError(f"Unknown human feel label: {value!r}")


class _MusicConfigInternal(BaseModel):
    """Internal config with numeric values used by the synth engine."""

    schema_version: Literal[1] = 1
    tempo: float = 0.5
    brightness: float = 0.5
    root: RootNote = "c"
    mode: ModeName = "minor"
    space: float = 0.5
    density: DensityLevel = 5

    bass: BassStyle = "drone"
    pad: PadStyle = "warm_slow"
    melody: MelodyStyle = "contemplative"
    rhythm: RhythmStyle = "minimal"
    texture: TextureStyle = "shimmer"
    accent: AccentStyle = "bells"

    motion: float = 0.5
    attack: AttackStyle = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: GrainStyle = "clean"

    model_config = ConfigDict(extra="forbid")


class MusicConfig(BaseModel):
    """Strictly-typed public config with tolerant extra capture."""

    schema_version: Literal[1] = 1
    tempo: TempoLabel = "medium"
    brightness: BrightnessLabel = "medium"
    root: RootNote = "c"
    mode: ModeName = "minor"
    space: SpaceLabel = "medium"
    density: DensityLevel = 5

    bass: BassStyle = "drone"
    pad: PadStyle = "warm_slow"
    melody: MelodyStyle = "contemplative"
    rhythm: RhythmStyle = "minimal"
    texture: TextureStyle = "shimmer"
    accent: AccentStyle = "bells"

    motion: MotionLabel = "medium"
    attack: AttackStyle = "medium"
    stereo: StereoLabel = "medium"
    depth: bool = False
    echo: EchoLabel = "medium"
    human: HumanFeelLabel = "robotic"
    grain: GrainStyle = "clean"

    # extras: dict[str, Any] = Field(default_factory=dict)

    # model_config = ConfigDict(extra="allow")

    def to_internal(self) -> _MusicConfigInternal:
        return _MusicConfigInternal(
            tempo=tempo_to_float(self.tempo),
            brightness=brightness_to_float(self.brightness),
            root=self.root,
            mode=self.mode,
            space=space_to_float(self.space),
            density=self.density,
            bass=self.bass,
            pad=self.pad,
            melody=self.melody,
            rhythm=self.rhythm,
            texture=self.texture,
            accent=self.accent,
            motion=motion_to_float(self.motion),
            attack=self.attack,
            stereo=stereo_to_float(self.stereo),
            depth=self.depth,
            echo=echo_to_float(self.echo),
            human=human_to_float(self.human),
            grain=self.grain,
        )


class _MusicConfigUpdateInternal(BaseModel):
    """Partial update for internal numeric configs."""

    tempo: Optional[float] = None
    brightness: Optional[float] = None
    root: Optional[RootNote] = None
    mode: Optional[ModeName] = None
    space: Optional[float] = None
    density: Optional[DensityLevel] = None

    bass: Optional[BassStyle] = None
    pad: Optional[PadStyle] = None
    melody: Optional[MelodyStyle] = None
    rhythm: Optional[RhythmStyle] = None
    texture: Optional[TextureStyle] = None
    accent: Optional[AccentStyle] = None

    motion: Optional[float] = None
    attack: Optional[AttackStyle] = None
    stereo: Optional[float] = None
    depth: Optional[bool] = None
    echo: Optional[float] = None
    human: Optional[float] = None
    grain: Optional[GrainStyle] = None

    model_config = ConfigDict(extra="forbid")


class MusicConfigUpdate(BaseModel):
    """Partial update for human inputs; unknown keys are rejected."""

    tempo: Optional[TempoLabel] = None
    brightness: Optional[BrightnessLabel] = None
    root: Optional[RootNote] = None
    mode: Optional[ModeName] = None
    space: Optional[SpaceLabel] = None
    density: Optional[DensityLevel] = None

    bass: Optional[BassStyle] = None
    pad: Optional[PadStyle] = None
    melody: Optional[MelodyStyle] = None
    rhythm: Optional[RhythmStyle] = None
    texture: Optional[TextureStyle] = None
    accent: Optional[AccentStyle] = None

    motion: Optional[MotionLabel] = None
    attack: Optional[AttackStyle] = None
    stereo: Optional[StereoLabel] = None
    depth: Optional[bool] = None
    echo: Optional[EchoLabel] = None
    human: Optional[HumanFeelLabel] = None
    grain: Optional[GrainStyle] = None

    model_config = ConfigDict(extra="forbid")

    def to_internal(self) -> _MusicConfigUpdateInternal:
        return _MusicConfigUpdateInternal(
            tempo=_optional_map(self.tempo, tempo_to_float),
            brightness=_optional_map(self.brightness, brightness_to_float),
            root=self.root,
            mode=self.mode,
            space=_optional_map(self.space, space_to_float),
            density=self.density,
            bass=self.bass,
            pad=self.pad,
            melody=self.melody,
            rhythm=self.rhythm,
            texture=self.texture,
            accent=self.accent,
            motion=_optional_map(self.motion, motion_to_float),
            attack=self.attack,
            stereo=_optional_map(self.stereo, stereo_to_float),
            depth=self.depth,
            echo=_optional_map(self.echo, echo_to_float),
            human=_optional_map(self.human, human_to_float),
            grain=self.grain,
        )


ConfigInput = MusicConfig | _MusicConfigInternal
UpdateInput = MusicConfigUpdate | _MusicConfigUpdateInternal


def coerce_internal_config(config: ConfigInput) -> _MusicConfigInternal:
    match config:
        case _MusicConfigInternal():
            return config
        case MusicConfig():
            return config.to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported config type: {type(config).__name__}")


def coerce_internal_update(update: UpdateInput) -> _MusicConfigUpdateInternal:
    match update:
        case _MusicConfigUpdateInternal():
            return update
        case MusicConfigUpdate():
            return update.to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported update type: {type(update).__name__}")


def merge_config(base: MusicConfig, update: MusicConfigUpdate) -> MusicConfig:
    """Apply a partial update to a base config."""

    changes = update.model_dump(exclude_none=True)
    return base.model_copy(update=changes)


def merge_internal_config(
    base: _MusicConfigInternal,
    update: _MusicConfigUpdateInternal,
) -> _MusicConfigInternal:
    """Apply a partial update to a base internal config."""

    changes = update.model_dump(exclude_none=True)
    return base.model_copy(update=changes)


def parse_update(payload: Mapping[str, Any]) -> MusicConfigUpdate:
    """Parse a strict update payload, raising InvalidConfigError on failure."""

    try:
        return MusicConfigUpdate.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Failed to parse config update: %s", exc, exc_info=True)
        raise InvalidConfigError(str(exc)) from exc


def parse_config(payload: Mapping[str, Any]) -> MusicConfig:
    """Parse a tolerant config payload, raising InvalidConfigError on failure."""

    try:
        return MusicConfig.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Failed to parse config payload: %s", exc, exc_info=True)
        raise InvalidConfigError(str(exc)) from exc


def is_empty_update(update: UpdateInput) -> bool:
    """Return True when the update does not specify any fields."""

    return not update.model_dump(exclude_none=True)
