from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    "procedural",
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
MelodyEngine = Literal["pattern", "procedural"]
TensionCurve = Literal["arc", "ramp", "waves"]
HarmonyStyle = Literal["auto", "pop", "jazz", "cinematic", "ambient"]
ChordExtensions = Literal["triads", "sevenths", "lush"]
PhraseLengthBars = Literal[2, 4, 8]
MelodyDensityLabel = Literal["very_sparse", "sparse", "medium", "busy", "very_busy"]
SyncopationLabel = Literal["straight", "light", "medium", "heavy"]
SwingLabel = Literal["none", "light", "medium", "heavy"]
MotifRepeatLabel = Literal["rare", "sometimes", "often"]
StepBiasLabel = Literal["step", "balanced", "leapy"]
ChromaticLabel = Literal["none", "light", "medium", "heavy"]
CadenceLabel = Literal["weak", "medium", "strong"]
ChordChangeLabel = Literal["very_slow", "slow", "medium", "fast"]


T = TypeVar("T")

_TEMPO_MAP: Mapping[TempoLabel, float] = MappingProxyType(
    {
        "very_slow": 0.15,
        "slow": 0.3,
        "medium": 0.5,
        "fast": 0.7,
        "very_fast": 0.9,
    }
)
_BRIGHTNESS_MAP: Mapping[BrightnessLabel, float] = MappingProxyType(
    {
        "very_dark": 0.1,
        "dark": 0.3,
        "medium": 0.5,
        "bright": 0.7,
        "very_bright": 0.9,
    }
)
_SPACE_MAP: Mapping[SpaceLabel, float] = MappingProxyType(
    {
        "dry": 0.1,
        "small": 0.3,
        "medium": 0.5,
        "large": 0.7,
        "vast": 0.95,
    }
)
_MOTION_MAP: Mapping[MotionLabel, float] = MappingProxyType(
    {
        "static": 0.1,
        "slow": 0.3,
        "medium": 0.5,
        "fast": 0.7,
        "chaotic": 0.9,
    }
)
_STEREO_MAP: Mapping[StereoLabel, float] = MappingProxyType(
    {
        "mono": 0.0,
        "narrow": 0.25,
        "medium": 0.5,
        "wide": 0.75,
        "ultra_wide": 1.0,
    }
)
_ECHO_MAP: Mapping[EchoLabel, float] = MappingProxyType(
    {
        "none": 0.0,
        "subtle": 0.25,
        "medium": 0.5,
        "heavy": 0.75,
        "infinite": 0.95,
    }
)
_HUMAN_MAP: Mapping[HumanFeelLabel, float] = MappingProxyType(
    {
        "robotic": 0.0,
        "tight": 0.15,
        "natural": 0.3,
        "loose": 0.5,
        "drunk": 0.8,
    }
)
_MELODY_DENSITY_MAP: Mapping[MelodyDensityLabel, float] = MappingProxyType(
    {
        "very_sparse": 0.15,
        "sparse": 0.30,
        "medium": 0.50,
        "busy": 0.70,
        "very_busy": 0.85,
    }
)
_SYNCOPATION_MAP: Mapping[SyncopationLabel, float] = MappingProxyType(
    {
        "straight": 0.0,
        "light": 0.2,
        "medium": 0.5,
        "heavy": 0.8,
    }
)
_SWING_MAP: Mapping[SwingLabel, float] = MappingProxyType(
    {
        "none": 0.0,
        "light": 0.2,
        "medium": 0.5,
        "heavy": 0.8,
    }
)
_MOTIF_REPEAT_MAP: Mapping[MotifRepeatLabel, float] = MappingProxyType(
    {
        "rare": 0.2,
        "sometimes": 0.5,
        "often": 0.8,
    }
)
_STEP_BIAS_MAP: Mapping[StepBiasLabel, float] = MappingProxyType(
    {
        "step": 0.9,
        "balanced": 0.7,
        "leapy": 0.4,
    }
)
_CHROMATIC_MAP: Mapping[ChromaticLabel, float] = MappingProxyType(
    {
        "none": 0.0,
        "light": 0.05,
        "medium": 0.12,
        "heavy": 0.25,
    }
)
_CADENCE_MAP: Mapping[CadenceLabel, float] = MappingProxyType(
    {
        "weak": 0.3,
        "medium": 0.6,
        "strong": 0.9,
    }
)
_CHORD_CHANGE_BARS_MAP: Mapping[ChordChangeLabel, int] = MappingProxyType(
    {
        "very_slow": 4,
        "slow": 2,
        "medium": 1,
        "fast": 1,
    }
)


def _optional_map(value: T | None, mapper: Callable[[T], float]) -> float | None:
    match value:
        case None:
            return None
        case _:
            return mapper(value)


def tempo_to_float(value: TempoLabel) -> float:
    try:
        return _TEMPO_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown tempo label: {value!r}") from exc


def brightness_to_float(value: BrightnessLabel) -> float:
    try:
        return _BRIGHTNESS_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown brightness label: {value!r}") from exc


def space_to_float(value: SpaceLabel) -> float:
    try:
        return _SPACE_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown space label: {value!r}") from exc


def motion_to_float(value: MotionLabel) -> float:
    try:
        return _MOTION_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown motion label: {value!r}") from exc


def stereo_to_float(value: StereoLabel) -> float:
    try:
        return _STEREO_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown stereo label: {value!r}") from exc


def echo_to_float(value: EchoLabel) -> float:
    try:
        return _ECHO_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown echo label: {value!r}") from exc


def human_to_float(value: HumanFeelLabel) -> float:
    try:
        return _HUMAN_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown human feel label: {value!r}") from exc


def melody_density_to_float(value: MelodyDensityLabel) -> float:
    try:
        return _MELODY_DENSITY_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown melody density label: {value!r}") from exc


def syncopation_to_float(value: SyncopationLabel) -> float:
    try:
        return _SYNCOPATION_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown syncopation label: {value!r}") from exc


def swing_to_float(value: SwingLabel) -> float:
    try:
        return _SWING_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown swing label: {value!r}") from exc


def motif_repeat_to_float(value: MotifRepeatLabel) -> float:
    try:
        return _MOTIF_REPEAT_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown motif repeat label: {value!r}") from exc


def step_bias_to_float(value: StepBiasLabel) -> float:
    try:
        return _STEP_BIAS_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown step bias label: {value!r}") from exc


def chromatic_to_float(value: ChromaticLabel) -> float:
    try:
        return _CHROMATIC_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown chromatic label: {value!r}") from exc


def cadence_to_float(value: CadenceLabel) -> float:
    try:
        return _CADENCE_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown cadence label: {value!r}") from exc


def chord_change_to_bars(value: ChordChangeLabel) -> int:
    try:
        return _CHORD_CHANGE_BARS_MAP[value]
    except KeyError as exc:
        raise InvalidConfigError(f"Unknown chord change label: {value!r}") from exc


class _MusicConfigInternal(BaseModel):
    """Internal config with numeric values used by the synth engine."""

    tempo: float = 0.35
    root: RootNote = "c"
    mode: ModeName = "minor"
    brightness: float = 0.5
    space: float = 0.6
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

    melody_engine: MelodyEngine = "pattern"
    phrase_len_bars: int = 4
    melody_density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: TensionCurve = "arc"
    harmony_style: HarmonyStyle = "auto"
    chord_change_bars: int = 1
    chord_extensions: ChordExtensions = "triads"

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _flatten_layers(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        merged: dict[str, Any] = dict(cast(Mapping[str, Any], data))
        layers = merged.pop("layers", None)
        if isinstance(layers, Mapping):
            for key, value in cast(Mapping[str, Any], layers).items():
                merged.setdefault(key, value)
        return merged

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_MusicConfigInternal":
        """Create config from dict (e.g., from JSON)."""
        return cls.model_validate(data)


class MusicConfig(BaseModel):
    """Strictly-typed public config with tolerant extra capture."""

    schema_version: Literal[1] = 1
    tempo: TempoLabel = "medium"
    root: RootNote = "c"
    mode: ModeName = "minor"
    brightness: BrightnessLabel = "medium"
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

    melody_engine: MelodyEngine = "pattern"
    phrase_len_bars: int = 4
    melody_density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: TensionCurve = "arc"
    harmony_style: HarmonyStyle = "auto"
    chord_change_bars: int = 1
    chord_extensions: ChordExtensions = "triads"

    model_config = ConfigDict(extra="allow")

    @property
    def extras(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

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
            melody_engine=self.melody_engine,
            phrase_len_bars=self.phrase_len_bars,
            melody_density=self.melody_density,
            syncopation=self.syncopation,
            swing=self.swing,
            motif_repeat_prob=self.motif_repeat_prob,
            step_bias=self.step_bias,
            chromatic_prob=self.chromatic_prob,
            cadence_strength=self.cadence_strength,
            register_min_oct=self.register_min_oct,
            register_max_oct=self.register_max_oct,
            tension_curve=self.tension_curve,
            harmony_style=self.harmony_style,
            chord_change_bars=self.chord_change_bars,
            chord_extensions=self.chord_extensions,
        )


SynthConfig = _MusicConfigInternal


_PROMPT_DESC: dict[str, str] = {
    "justification": (
        "Explain the sonic reasoning for the choices. Mention vibe decomposition, sonic "
        "translation, coherence check, and which examples guided the selection."
    ),
    "config": "Music configuration that matches the requested vibe.",
    "tempo": "Tempo label controlling overall speed and energy.",
    "root": "Root note of the scale.",
    "mode": "Scale mode that shapes the emotional color.",
    "brightness": "Filter brightness / spectral tilt label.",
    "space": "Reverb/room size label.",
    "density": "Layer count indicating overall thickness.",
    "bass": "Bass style or movement pattern.",
    "pad": "Pad texture and harmonic bed style.",
    "melody": "Melody style or contour.",
    "rhythm": "Percussion pattern style (or none).",
    "texture": "Background texture or noise layer.",
    "accent": "Sparse accent sound type.",
    "motion": "Modulation/LFO rate label.",
    "attack": "Transient sharpness label.",
    "stereo": "Stereo width label.",
    "depth": "Whether to add sub-bass depth.",
    "echo": "Delay amount label.",
    "human": "Timing/pitch looseness label.",
    "grain": "Oscillator character (clean/warm/gritty).",
    "melody_engine": "Melody generation mode (procedural or pattern).",
    "phrase_len_bars": "Phrase length in bars (2, 4, or 8).",
    "melody_density": "Melody note density label (very_sparse, sparse, medium, busy, very_busy).",
    "syncopation": "Offbeat emphasis label (straight, light, medium, heavy).",
    "swing": "Swing amount label (none, light, medium, heavy).",
    "motif_repeat_prob": "Motif repetition label (rare, sometimes, often).",
    "step_bias": "Melodic motion label (step, balanced, leapy).",
    "chromatic_prob": "Chromaticism label (none, light, medium, heavy).",
    "cadence_strength": "Cadence emphasis label (weak, medium, strong).",
    "register_min_oct": "Lowest melody octave (integer).",
    "register_max_oct": "Highest melody octave (integer).",
    "tension_curve": "Tension shape across the phrase.",
    "harmony_style": "Harmony progression style.",
    "chord_change_bars": "Chord change rate label (very_slow, slow, medium, fast).",
    "chord_extensions": "Chord color/extension level.",
}

_PROMPT_REGISTER_MIN = 1
_PROMPT_REGISTER_MAX = 8


class MusicConfigPrompt(BaseModel):
    """Prompt-only schema matching MusicConfig without defaults."""

    tempo: TempoLabel = Field(description=_PROMPT_DESC["tempo"])
    root: RootNote = Field(description=_PROMPT_DESC["root"])
    mode: ModeName = Field(description=_PROMPT_DESC["mode"])
    brightness: BrightnessLabel = Field(description=_PROMPT_DESC["brightness"])
    space: SpaceLabel = Field(description=_PROMPT_DESC["space"])
    density: DensityLevel = Field(ge=2, le=6, description=_PROMPT_DESC["density"])

    bass: BassStyle = Field(description=_PROMPT_DESC["bass"])
    pad: PadStyle = Field(description=_PROMPT_DESC["pad"])
    melody: MelodyStyle = Field(description=_PROMPT_DESC["melody"])
    rhythm: RhythmStyle = Field(description=_PROMPT_DESC["rhythm"])
    texture: TextureStyle = Field(description=_PROMPT_DESC["texture"])
    accent: AccentStyle = Field(description=_PROMPT_DESC["accent"])

    motion: MotionLabel = Field(description=_PROMPT_DESC["motion"])
    attack: AttackStyle = Field(description=_PROMPT_DESC["attack"])
    stereo: StereoLabel = Field(description=_PROMPT_DESC["stereo"])
    depth: bool = Field(description=_PROMPT_DESC["depth"])
    echo: EchoLabel = Field(description=_PROMPT_DESC["echo"])
    human: HumanFeelLabel = Field(description=_PROMPT_DESC["human"])
    grain: GrainStyle = Field(description=_PROMPT_DESC["grain"])

    melody_engine: MelodyEngine = Field(description=_PROMPT_DESC["melody_engine"])
    phrase_len_bars: PhraseLengthBars = Field(description=_PROMPT_DESC["phrase_len_bars"])
    melody_density: MelodyDensityLabel = Field(description=_PROMPT_DESC["melody_density"])
    syncopation: SyncopationLabel = Field(description=_PROMPT_DESC["syncopation"])
    swing: SwingLabel = Field(description=_PROMPT_DESC["swing"])
    motif_repeat_prob: MotifRepeatLabel = Field(description=_PROMPT_DESC["motif_repeat_prob"])
    step_bias: StepBiasLabel = Field(description=_PROMPT_DESC["step_bias"])
    chromatic_prob: ChromaticLabel = Field(description=_PROMPT_DESC["chromatic_prob"])
    cadence_strength: CadenceLabel = Field(description=_PROMPT_DESC["cadence_strength"])
    register_min_oct: int = Field(
        ge=_PROMPT_REGISTER_MIN,
        le=_PROMPT_REGISTER_MAX,
        description=_PROMPT_DESC["register_min_oct"],
    )
    register_max_oct: int = Field(
        ge=_PROMPT_REGISTER_MIN,
        le=_PROMPT_REGISTER_MAX,
        description=_PROMPT_DESC["register_max_oct"],
    )
    tension_curve: TensionCurve = Field(description=_PROMPT_DESC["tension_curve"])
    harmony_style: HarmonyStyle = Field(description=_PROMPT_DESC["harmony_style"])
    chord_change_bars: ChordChangeLabel = Field(description=_PROMPT_DESC["chord_change_bars"])
    chord_extensions: ChordExtensions = Field(description=_PROMPT_DESC["chord_extensions"])

    def to_config(self) -> MusicConfig:
        data = self.model_dump()
        data["melody_density"] = melody_density_to_float(self.melody_density)
        data["syncopation"] = syncopation_to_float(self.syncopation)
        data["swing"] = swing_to_float(self.swing)
        data["motif_repeat_prob"] = motif_repeat_to_float(self.motif_repeat_prob)
        data["step_bias"] = step_bias_to_float(self.step_bias)
        data["chromatic_prob"] = chromatic_to_float(self.chromatic_prob)
        data["cadence_strength"] = cadence_to_float(self.cadence_strength)
        data["chord_change_bars"] = chord_change_to_bars(self.chord_change_bars)
        return MusicConfig.model_validate(data)

    model_config = ConfigDict(extra="forbid")


class MusicConfigPromptPayload(BaseModel):
    """LLM payload that includes a justification and a config."""

    justification: str = Field(description=_PROMPT_DESC["justification"])
    config: MusicConfigPrompt = Field(description=_PROMPT_DESC["config"])

    model_config = ConfigDict(extra="forbid")


def _assert_prompt_schema_parity() -> None:
    prompt_fields = set(MusicConfigPrompt.model_fields)
    config_fields = set(MusicConfig.model_fields)
    excluded = {"schema_version"}
    if prompt_fields == (config_fields - excluded):
        return
    missing = sorted(config_fields - prompt_fields - excluded)
    extra = sorted(prompt_fields - config_fields)
    raise AssertionError(f"MusicConfigPrompt mismatch: missing={missing!r}, extra={extra!r}")


_assert_prompt_schema_parity()


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

    melody_engine: Optional[MelodyEngine] = None
    phrase_len_bars: Optional[int] = None
    melody_density: Optional[float] = None
    syncopation: Optional[float] = None
    swing: Optional[float] = None
    motif_repeat_prob: Optional[float] = None
    step_bias: Optional[float] = None
    chromatic_prob: Optional[float] = None
    cadence_strength: Optional[float] = None
    register_min_oct: Optional[int] = None
    register_max_oct: Optional[int] = None
    tension_curve: Optional[TensionCurve] = None
    harmony_style: Optional[HarmonyStyle] = None
    chord_change_bars: Optional[int] = None
    chord_extensions: Optional[ChordExtensions] = None

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

    melody_engine: Optional[MelodyEngine] = None
    phrase_len_bars: Optional[int] = None
    melody_density: Optional[float] = None
    syncopation: Optional[float] = None
    swing: Optional[float] = None
    motif_repeat_prob: Optional[float] = None
    step_bias: Optional[float] = None
    chromatic_prob: Optional[float] = None
    cadence_strength: Optional[float] = None
    register_min_oct: Optional[int] = None
    register_max_oct: Optional[int] = None
    tension_curve: Optional[TensionCurve] = None
    harmony_style: Optional[HarmonyStyle] = None
    chord_change_bars: Optional[int] = None
    chord_extensions: Optional[ChordExtensions] = None

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
            melody_engine=self.melody_engine,
            phrase_len_bars=self.phrase_len_bars,
            melody_density=self.melody_density,
            syncopation=self.syncopation,
            swing=self.swing,
            motif_repeat_prob=self.motif_repeat_prob,
            step_bias=self.step_bias,
            chromatic_prob=self.chromatic_prob,
            cadence_strength=self.cadence_strength,
            register_min_oct=self.register_min_oct,
            register_max_oct=self.register_max_oct,
            tension_curve=self.tension_curve,
            harmony_style=self.harmony_style,
            chord_change_bars=self.chord_change_bars,
            chord_extensions=self.chord_extensions,
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
