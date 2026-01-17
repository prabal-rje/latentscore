"""Baseline config generators for evaluation comparison."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, get_args  # noqa: F401 (TypeVar used in generic)

from common.music_schema import (
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
    MusicConfigPrompt,
    MusicConfigPromptPayload,
    PadStyle,
    Palette,
    PaletteColor,
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
    WeightLabel,
)

# Extract valid values from Literal types with explicit type annotations.
# This allows random.choice() to return the correct type instead of Any.
TEMPO_VALUES: Sequence[TempoLabel] = get_args(TempoLabel)
BRIGHTNESS_VALUES: Sequence[BrightnessLabel] = get_args(BrightnessLabel)
MODE_VALUES: Sequence[ModeName] = get_args(ModeName)
ROOT_VALUES: Sequence[RootNote] = get_args(RootNote)
SPACE_VALUES: Sequence[SpaceLabel] = get_args(SpaceLabel)
MOTION_VALUES: Sequence[MotionLabel] = get_args(MotionLabel)
STEREO_VALUES: Sequence[StereoLabel] = get_args(StereoLabel)
ECHO_VALUES: Sequence[EchoLabel] = get_args(EchoLabel)
HUMAN_FEEL_VALUES: Sequence[HumanFeelLabel] = get_args(HumanFeelLabel)
DENSITY_VALUES: Sequence[DensityLevel] = get_args(DensityLevel)
BASS_STYLES: Sequence[BassStyle] = get_args(BassStyle)
PAD_STYLES: Sequence[PadStyle] = get_args(PadStyle)
MELODY_STYLES: Sequence[MelodyStyle] = get_args(MelodyStyle)
RHYTHM_STYLES: Sequence[RhythmStyle] = get_args(RhythmStyle)
TEXTURE_STYLES: Sequence[TextureStyle] = get_args(TextureStyle)
ACCENT_STYLES: Sequence[AccentStyle] = get_args(AccentStyle)
ATTACK_STYLES: Sequence[AttackStyle] = get_args(AttackStyle)
GRAIN_STYLES: Sequence[GrainStyle] = get_args(GrainStyle)
MELODY_ENGINES: Sequence[MelodyEngine] = get_args(MelodyEngine)
TENSION_CURVES: Sequence[TensionCurve] = get_args(TensionCurve)
HARMONY_STYLES: Sequence[HarmonyStyle] = get_args(HarmonyStyle)
CHORD_EXTENSIONS_VALUES: Sequence[ChordExtensions] = get_args(ChordExtensions)
PHRASE_LENGTH_BARS: Sequence[PhraseLengthBars] = get_args(PhraseLengthBars)
MELODY_DENSITY_VALUES: Sequence[MelodyDensityLabel] = get_args(MelodyDensityLabel)
SYNCOPATION_VALUES: Sequence[SyncopationLabel] = get_args(SyncopationLabel)
SWING_VALUES: Sequence[SwingLabel] = get_args(SwingLabel)
MOTIF_REPEAT_VALUES: Sequence[MotifRepeatLabel] = get_args(MotifRepeatLabel)
STEP_BIAS_VALUES: Sequence[StepBiasLabel] = get_args(StepBiasLabel)
CHROMATIC_VALUES: Sequence[ChromaticLabel] = get_args(ChromaticLabel)
CADENCE_VALUES: Sequence[CadenceLabel] = get_args(CadenceLabel)
CHORD_CHANGE_VALUES: Sequence[ChordChangeLabel] = get_args(ChordChangeLabel)
WEIGHT_VALUES: Sequence[WeightLabel] = get_args(WeightLabel)

# Hex colors for random palettes
HEX_COLORS = [
    "#1a237e",
    "#283593",
    "#303f9f",
    "#3949ab",
    "#3f51b5",  # blues
    "#e65100",
    "#ef6c00",
    "#f57c00",
    "#fb8c00",
    "#ff9800",  # oranges
    "#1b5e20",
    "#2e7d32",
    "#388e3c",
    "#43a047",
    "#4caf50",  # greens
    "#4a148c",
    "#6a1b9a",
    "#7b1fa2",
    "#8e24aa",
    "#9c27b0",  # purples
    "#ff6f00",
    "#ff8f00",
    "#ffa000",
    "#ffb300",
    "#ffc107",  # golds
    "#263238",
    "#37474f",
    "#455a64",
    "#546e7a",
    "#607d8b",  # grays
    "#b71c1c",
    "#c62828",
    "#d32f2f",
    "#e53935",
    "#f44336",  # reds
    "#006064",
    "#00838f",
    "#0097a7",
    "#00acc1",
    "#00bcd4",  # cyans
]


class ConfigBaseline(ABC):
    """Abstract base class for config baselines."""

    @abstractmethod
    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config from a vibe prompt."""
        pass


class RandomConfigBaseline(ConfigBaseline):
    """Baseline that generates random valid configs."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _random_palette(self) -> Palette:
        """Generate a random palette with exactly 5 colors."""
        colors = self.rng.sample(HEX_COLORS, k=5)
        weights = self.rng.sample(list(WEIGHT_VALUES), k=5)
        return Palette(colors=[PaletteColor(hex=c, weight=w) for c, w in zip(colors, weights)])

    def _random_config(self) -> MusicConfigPrompt:
        """Generate a random MusicConfigPrompt."""
        _MIN_OCTAVE = 1
        _MAX_OCTAVE_LOW = 4
        _MAX_OCTAVE_HIGH = 8
        min_oct = self.rng.randint(_MIN_OCTAVE, _MAX_OCTAVE_LOW)
        max_oct = self.rng.randint(min_oct + 1, _MAX_OCTAVE_HIGH)
        return MusicConfigPrompt(
            tempo=self.rng.choice(TEMPO_VALUES),
            root=self.rng.choice(ROOT_VALUES),
            mode=self.rng.choice(MODE_VALUES),
            brightness=self.rng.choice(BRIGHTNESS_VALUES),
            space=self.rng.choice(SPACE_VALUES),
            density=self.rng.choice(DENSITY_VALUES),
            bass=self.rng.choice(BASS_STYLES),
            pad=self.rng.choice(PAD_STYLES),
            melody=self.rng.choice(MELODY_STYLES),
            rhythm=self.rng.choice(RHYTHM_STYLES),
            texture=self.rng.choice(TEXTURE_STYLES),
            accent=self.rng.choice(ACCENT_STYLES),
            motion=self.rng.choice(MOTION_VALUES),
            attack=self.rng.choice(ATTACK_STYLES),
            stereo=self.rng.choice(STEREO_VALUES),
            depth=self.rng.choice([True, False]),
            echo=self.rng.choice(ECHO_VALUES),
            human=self.rng.choice(HUMAN_FEEL_VALUES),
            grain=self.rng.choice(GRAIN_STYLES),
            melody_engine=self.rng.choice(MELODY_ENGINES),
            phrase_len_bars=self.rng.choice(PHRASE_LENGTH_BARS),
            melody_density=self.rng.choice(MELODY_DENSITY_VALUES),
            syncopation=self.rng.choice(SYNCOPATION_VALUES),
            swing=self.rng.choice(SWING_VALUES),
            motif_repeat_prob=self.rng.choice(MOTIF_REPEAT_VALUES),
            step_bias=self.rng.choice(STEP_BIAS_VALUES),
            chromatic_prob=self.rng.choice(CHROMATIC_VALUES),
            cadence_strength=self.rng.choice(CADENCE_VALUES),
            register_min_oct=min_oct,
            register_max_oct=max_oct,
            tension_curve=self.rng.choice(TENSION_CURVES),
            harmony_style=self.rng.choice(HARMONY_STYLES),
            chord_change_bars=self.rng.choice(CHORD_CHANGE_VALUES),
            chord_extensions=self.rng.choice(CHORD_EXTENSIONS_VALUES),
        )

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a random MusicConfigPromptPayload (ignores vibe)."""
        _ = vibe  # Intentionally ignored
        return MusicConfigPromptPayload(
            justification="Random baseline config for evaluation purposes.",
            config=self._random_config(),
            palettes=[self._random_palette() for _ in range(3)],
        )


# Keyword mappings for rule-based baseline
TEMPO_KEYWORDS: dict[TempoLabel, list[str]] = {
    "very_slow": ["glacial", "frozen", "adagio", "crawl", "statue", "still"],
    "slow": ["slow", "languid", "leisurely", "drift", "gentle", "calm", "peaceful"],
    "fast": ["fast", "quick", "urgent", "rush", "chase", "rapid", "allegro"],
    "very_fast": ["frantic", "presto", "racing", "sprint", "breakneck", "furious"],
}

BRIGHTNESS_KEYWORDS: dict[BrightnessLabel, list[str]] = {
    "very_dark": ["pitch black", "void", "abyss", "lightless", "moonless"],
    "dark": ["dark", "dim", "shadow", "night", "fog", "gray", "overcast", "candle"],
    "bright": ["bright", "sunny", "light", "morning", "golden", "glow"],
    "very_bright": ["blinding", "dazzling", "radiant", "noon", "fluorescent", "gleaming"],
}

MODE_KEYWORDS: dict[ModeName, list[str]] = {
    "minor": [
        "sad",
        "melancholy",
        "grief",
        "loss",
        "mournful",
        "tragic",
        "funeral",
        "somber",
        "bittersweet",
        "minor",
    ],
    "major": [
        "happy",
        "joy",
        "triumph",
        "victory",
        "celebration",
        "hopeful",
        "uplifting",
        "wedding",
        "bright",
        "major",
        "cheerful",
    ],
    "dorian": ["jazz", "dorian", "funky", "groove", "smoky", "cool"],
    "mixolydian": ["folk", "celtic", "country", "mixolydian", "highlands"],
}

SPACE_KEYWORDS: dict[SpaceLabel, list[str]] = {
    "dry": ["dry", "close", "intimate", "dead"],
    "small": ["room", "indoor", "small"],
    "large": ["hall", "cathedral", "large", "vast"],
    "vast": ["space", "infinite", "endless", "cosmos", "universe"],
}


_T = TypeVar("_T", bound=str)


class RuleBasedBaseline(ConfigBaseline):
    """Baseline using keyword matching heuristics."""

    def __init__(self) -> None:
        self.random_baseline = RandomConfigBaseline()

    def _match_keywords(
        self,
        vibe: str,
        keyword_map: dict[_T, list[str]],
        default: _T,
    ) -> _T:
        """Find the best matching value based on keywords."""
        vibe_lower = vibe.lower()
        best_match = default
        best_count = 0

        for value, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in vibe_lower)
            if count > best_count:
                best_count = count
                best_match = value

        return best_match

    def _extract_numbers(self, vibe: str) -> list[int]:
        """Extract numbers from vibe text."""
        return [int(n) for n in re.findall(r"\d+", vibe)]

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config using keyword matching."""
        # Start with random config as base
        payload = self.random_baseline.generate(vibe)

        # Default values for keyword matching (typed as Literals)
        _DEFAULT_TEMPO: TempoLabel = "medium"
        _DEFAULT_BRIGHTNESS: BrightnessLabel = "medium"
        _DEFAULT_MODE: ModeName = "minor"
        _DEFAULT_SPACE: SpaceLabel = "medium"

        # Create a mutable copy of the config with overrides
        tempo = self._match_keywords(vibe, TEMPO_KEYWORDS, _DEFAULT_TEMPO)
        brightness = self._match_keywords(vibe, BRIGHTNESS_KEYWORDS, _DEFAULT_BRIGHTNESS)
        mode = self._match_keywords(vibe, MODE_KEYWORDS, _DEFAULT_MODE)
        space = self._match_keywords(vibe, SPACE_KEYWORDS, _DEFAULT_SPACE)

        # Adjust motion based on tempo
        motion: MotionLabel = "medium"
        if tempo in ["very_slow", "slow"]:
            motion = "slow"
        elif tempo in ["fast", "very_fast"]:
            motion = "fast"

        # Adjust rhythm based on tempo and keywords
        rhythm: RhythmStyle = payload.config.rhythm
        if "no drum" in vibe.lower() or "ambient" in vibe.lower():
            rhythm = "none"
        elif tempo == "very_slow":
            rhythm = "none"
        elif tempo == "slow":
            rhythm = "minimal"

        # Create new config with overrides
        new_config = MusicConfigPrompt(
            tempo=tempo,
            brightness=brightness,
            mode=mode,
            space=space,
            motion=motion,
            rhythm=rhythm,
            # Keep the rest from the random baseline
            root=payload.config.root,
            density=payload.config.density,
            bass=payload.config.bass,
            pad=payload.config.pad,
            melody=payload.config.melody,
            texture=payload.config.texture,
            accent=payload.config.accent,
            attack=payload.config.attack,
            stereo=payload.config.stereo,
            depth=payload.config.depth,
            echo=payload.config.echo,
            human=payload.config.human,
            grain=payload.config.grain,
            melody_engine=payload.config.melody_engine,
            phrase_len_bars=payload.config.phrase_len_bars,
            melody_density=payload.config.melody_density,
            syncopation=payload.config.syncopation,
            swing=payload.config.swing,
            motif_repeat_prob=payload.config.motif_repeat_prob,
            step_bias=payload.config.step_bias,
            chromatic_prob=payload.config.chromatic_prob,
            cadence_strength=payload.config.cadence_strength,
            register_min_oct=payload.config.register_min_oct,
            register_max_oct=payload.config.register_max_oct,
            tension_curve=payload.config.tension_curve,
            harmony_style=payload.config.harmony_style,
            chord_change_bars=payload.config.chord_change_bars,
            chord_extensions=payload.config.chord_extensions,
        )

        return MusicConfigPromptPayload(
            justification=f"Rule-based config for vibe: {vibe[:50]}...",
            config=new_config,
            palettes=payload.palettes,
        )


class ModeConfigBaseline(ConfigBaseline):
    """Baseline using most common values (mode) from training data."""

    def __init__(self) -> None:
        """Initialize with default mode values."""
        self.random_baseline = RandomConfigBaseline(seed=42)  # Fixed seed for consistency

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config using mode (most common) values (ignores vibe)."""
        _ = vibe  # Intentionally ignored
        # Start with random config for palettes and other fields
        payload = self.random_baseline.generate("")

        # Create config with "mode" (most common) values
        new_config = MusicConfigPrompt(
            tempo="medium",
            brightness="medium",
            space="medium",
            motion="medium",
            stereo="medium",
            echo="medium",
            human="natural",
            root="c",
            mode="minor",
            density=4,
            bass="drone",
            pad="ambient_drift",
            melody="contemplative",
            rhythm="minimal",
            texture="shimmer",
            accent="none",
            attack="medium",
            grain="warm",
            depth=True,
            melody_engine="procedural",
            phrase_len_bars=4,
            melody_density="medium",
            syncopation="light",
            swing="none",
            motif_repeat_prob="sometimes",
            step_bias="balanced",
            chromatic_prob="light",
            cadence_strength="medium",
            register_min_oct=3,
            register_max_oct=5,
            tension_curve="arc",
            harmony_style="ambient",
            chord_change_bars="slow",
            chord_extensions="triads",
        )

        return MusicConfigPromptPayload(
            justification="Mode baseline using most common values.",
            config=new_config,
            palettes=payload.palettes,
        )


# Registry of available baselines
BASELINES: dict[str, type[ConfigBaseline]] = {
    "random": RandomConfigBaseline,
    "rule_based": RuleBasedBaseline,
    "mode": ModeConfigBaseline,
}


def get_baseline(name: str) -> ConfigBaseline:
    """Get a baseline instance by name."""
    if name not in BASELINES:
        available = ", ".join(BASELINES.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    return BASELINES[name]()


def list_baselines() -> list[str]:
    """List available baseline names."""
    return list(BASELINES.keys())
