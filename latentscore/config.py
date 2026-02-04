from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Import shared types from common.music_schema
from common.music_schema import (
    MAX_LONG_FIELD_CHARS,
    MAX_TITLE_CHARS,
    MAX_TITLE_WORDS,
    PALETTES_DESC,
    PROMPT_DESC,
    PROMPT_REGISTER_MAX,
    PROMPT_REGISTER_MIN,
    AccentStyle,
    AttackStyle,
    BassStyle,
    BrightnessLabel,
    CadenceLabel,
    ChordChangeLabel,
    ChordExtensions,
    ChromaticLabel,
    DensityLevel,
    EchoLabel,
    GrainStyle,
    HarmonyStyle,
    HumanFeelLabel,
    MelodyDensityLabel,
    MelodyEngine,
    MelodyStyle,
    ModeName,
    MotifRepeatLabel,
    MotionLabel,
    PadStyle,
    Palette,
    PaletteColor,  # noqa: F401 - re-exported for schema parity tests
    PhraseLengthBars,
    RhythmStyle,
    RootNote,
    SpaceLabel,
    StepBiasLabel,
    StereoLabel,
    SwingLabel,
    SyncopationLabel,
    TempoLabel,
    TensionCurve,
    TextureStyle,
    WeightLabel,  # noqa: F401 - re-exported for schema parity tests
)

from .errors import InvalidConfigError

_LOGGER = logging.getLogger("latentscore.config")


T = TypeVar("T")


# -----------------------------------------------------------------------------
# Step: relative adjustment for steppable fields
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Step:
    """Relative adjustment for ordered label fields.

    Used in MusicConfigUpdate to step a field up/down by N levels
    instead of setting an absolute value.

    Example:
        update = MusicConfigUpdate(
            brightness=Step(+1),  # one level brighter
            tempo=Step(-2),       # two levels slower
        )
        new_config = update.apply_to(base_config)
    """

    delta: int

    def __repr__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return f"Step({sign}{self.delta})"


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


# Use PROMPT_DESC from common for field descriptions
_PROMPT_DESC = PROMPT_DESC
_PROMPT_REGISTER_MIN = PROMPT_REGISTER_MIN
_PROMPT_REGISTER_MAX = PROMPT_REGISTER_MAX


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
    """LLM payload that includes a thinking field, config, and palettes."""

    thinking: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=_PROMPT_DESC["thinking"],
    )
    title: str = Field(
        ...,
        max_length=MAX_TITLE_CHARS,
        min_length=1,
        description=_PROMPT_DESC["title"],
    )
    config: MusicConfigPrompt = Field(description=_PROMPT_DESC["config"])
    palettes: list[Palette] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=PALETTES_DESC,
    )

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        words = [word for word in value.strip().split() if word]
        if not words:
            raise ValueError("title must not be empty")
        if len(words) > MAX_TITLE_WORDS:
            raise ValueError("title exceeds max word count")
        return value

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
    """Partial update for human inputs; unknown keys are rejected.

    Steppable fields (those with ordered levels) accept either an absolute
    value or a Step for relative adjustment:

        update = MusicConfigUpdate(
            brightness=Step(+1),  # relative: one level brighter
            tempo="fast",         # absolute: set to "fast"
        )
        new_config = update.apply_to(base_config)
    """

    # Steppable fields: accept absolute value OR Step
    tempo: Optional[TempoLabel | Step] = None
    brightness: Optional[BrightnessLabel | Step] = None
    space: Optional[SpaceLabel | Step] = None
    density: Optional[DensityLevel | Step] = None
    motion: Optional[MotionLabel | Step] = None
    stereo: Optional[StereoLabel | Step] = None
    echo: Optional[EchoLabel | Step] = None
    human: Optional[HumanFeelLabel | Step] = None

    # Non-steppable fields: absolute values only
    root: Optional[RootNote] = None
    mode: Optional[ModeName] = None

    bass: Optional[BassStyle] = None
    pad: Optional[PadStyle] = None
    melody: Optional[MelodyStyle] = None
    rhythm: Optional[RhythmStyle] = None
    texture: Optional[TextureStyle] = None
    accent: Optional[AccentStyle] = None

    attack: Optional[AttackStyle] = None
    depth: Optional[bool] = None
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
        # Check for unresolved Step values
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Step):
                raise ValueError(
                    f"Cannot convert to internal: field '{field_name}' has unresolved Step. "
                    f"Call apply_to(base_config) first to resolve Step values."
                )
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

    def apply_to(self, base: MusicConfig) -> MusicConfig:
        """Apply this update to a base config, resolving any Step values.

        Step values are resolved by finding the current level in the ordered
        list of valid values and moving up/down by the step delta. Values
        saturate at the min/max (no overflow/underflow).

        Args:
            base: The base configuration to update.

        Returns:
            A new MusicConfig with updates applied.

        Example:
            >>> base = MusicConfig(brightness="medium")
            >>> update = MusicConfigUpdate(brightness=Step(+1))
            >>> new = update.apply_to(base)
            >>> new.brightness
            'bright'
        """
        updates: dict[str, Any] = {}

        for name, value in self:
            match value:
                case None:
                    continue
                case Step(delta=d):
                    levels = _extract_ordered_levels(self.model_fields[name].annotation)
                    if levels is None:
                        raise ValueError(f"Field '{name}' has Step but no ordered levels")
                    current = getattr(base, name)
                    if current is None:
                        raise ValueError(f"Cannot step '{name}': base value is None")
                    try:
                        idx = levels.index(current)
                    except ValueError:
                        raise ValueError(
                            f"Value '{current}' not in {name} levels: {levels}"
                        ) from None
                    updates[name] = levels[max(0, min(idx + d, len(levels) - 1))]
                case _:
                    updates[name] = value

        return base.model_copy(update=updates)


def _extract_ordered_levels(annotation: Any) -> list[Any] | None:
    """Extract ordered literal values from a type annotation like Optional[TempoLabel | Step].

    Returns None if the annotation doesn't contain an ordered Literal type.
    """
    # Unwrap Optional (Union with None)
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
    else:
        args = [annotation]

    # Look for a Literal type (not Step)
    for arg in args:
        if arg is Step:
            continue
        # Check if it's a Literal type
        literal_args = get_args(arg)
        if literal_args:
            return list(literal_args)

    return None


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
