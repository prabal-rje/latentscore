"""Shared schema definitions for music config generation.

This module provides the single source of truth for MusicConfigPrompt,
MusicConfigPromptPayload, and palette schemas used by both latentscore
and data_work packages.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Field length limits
MAX_LONG_FIELD_CHARS = 1_000
MAX_TITLE_CHARS = 60
MAX_TITLE_WORDS = 6

# Core label types
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

# Layer styles
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

# Advanced melody/harmony types
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

# Palette types
WeightLabel = Literal["xs", "sm", "md", "lg", "xl", "xxl"]
PALETTE_WEIGHT_ORDER: dict[str, int] = {"xxl": 0, "xl": 1, "lg": 2, "md": 3, "sm": 4, "xs": 5}

# Palette field descriptions (shared between all consumers)
PALETTE_COLOR_DESC = "Hex color string in #RRGGBB format."
PALETTE_WEIGHT_DESC = "Relative weight label (xs, sm, md, lg, xl, xxl)."
PALETTE_DESC = "Ranked palette with exactly five colors."
PALETTES_DESC = "Three ranked palettes matching the vibe."

# Prompt field descriptions (shared between latentscore and data_work)
PROMPT_DESC: dict[str, str] = {
    "thinking": (
        "Explain the sonic reasoning for the choices. Mention vibe decomposition, sonic "
        "translation, coherence check, and which examples guided the selection."
    ),
    "title": "Short, readable title summarizing the vibe (<=6 words).",
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

PROMPT_REGISTER_MIN = 1
PROMPT_REGISTER_MAX = 8


class PaletteColor(BaseModel):
    """A single color with weight in a palette."""

    model_config = ConfigDict(extra="forbid")

    hex: str = Field(..., description=PALETTE_COLOR_DESC, pattern=r"^#[0-9A-Fa-f]{6}$")
    weight: WeightLabel = Field(..., description=PALETTE_WEIGHT_DESC)


class Palette(BaseModel):
    """A ranked palette with exactly five colors, auto-sorted by weight."""

    model_config = ConfigDict(extra="forbid")

    colors: list[PaletteColor] = Field(..., min_length=5, max_length=5, description=PALETTE_DESC)

    @model_validator(mode="after")
    def _sort_colors_by_weight(self) -> "Palette":
        self.colors = sorted(self.colors, key=lambda c: PALETTE_WEIGHT_ORDER[c.weight])
        return self


class MusicConfigPrompt(BaseModel):
    """Prompt-only schema matching MusicConfig without defaults."""

    model_config = ConfigDict(extra="forbid")

    tempo: TempoLabel = Field(..., description=PROMPT_DESC["tempo"])
    root: RootNote = Field(..., description=PROMPT_DESC["root"])
    mode: ModeName = Field(..., description=PROMPT_DESC["mode"])
    brightness: BrightnessLabel = Field(..., description=PROMPT_DESC["brightness"])
    space: SpaceLabel = Field(..., description=PROMPT_DESC["space"])
    density: DensityLevel = Field(..., ge=2, le=6, description=PROMPT_DESC["density"])

    bass: BassStyle = Field(..., description=PROMPT_DESC["bass"])
    pad: PadStyle = Field(..., description=PROMPT_DESC["pad"])
    melody: MelodyStyle = Field(..., description=PROMPT_DESC["melody"])
    rhythm: RhythmStyle = Field(..., description=PROMPT_DESC["rhythm"])
    texture: TextureStyle = Field(..., description=PROMPT_DESC["texture"])
    accent: AccentStyle = Field(..., description=PROMPT_DESC["accent"])

    motion: MotionLabel = Field(..., description=PROMPT_DESC["motion"])
    attack: AttackStyle = Field(..., description=PROMPT_DESC["attack"])
    stereo: StereoLabel = Field(..., description=PROMPT_DESC["stereo"])
    depth: bool = Field(..., description=PROMPT_DESC["depth"])
    echo: EchoLabel = Field(..., description=PROMPT_DESC["echo"])
    human: HumanFeelLabel = Field(..., description=PROMPT_DESC["human"])
    grain: GrainStyle = Field(..., description=PROMPT_DESC["grain"])

    melody_engine: MelodyEngine = Field(..., description=PROMPT_DESC["melody_engine"])
    phrase_len_bars: PhraseLengthBars = Field(..., description=PROMPT_DESC["phrase_len_bars"])
    melody_density: MelodyDensityLabel = Field(..., description=PROMPT_DESC["melody_density"])
    syncopation: SyncopationLabel = Field(..., description=PROMPT_DESC["syncopation"])
    swing: SwingLabel = Field(..., description=PROMPT_DESC["swing"])
    motif_repeat_prob: MotifRepeatLabel = Field(..., description=PROMPT_DESC["motif_repeat_prob"])
    step_bias: StepBiasLabel = Field(..., description=PROMPT_DESC["step_bias"])
    chromatic_prob: ChromaticLabel = Field(..., description=PROMPT_DESC["chromatic_prob"])
    cadence_strength: CadenceLabel = Field(..., description=PROMPT_DESC["cadence_strength"])
    register_min_oct: int = Field(
        ...,
        ge=PROMPT_REGISTER_MIN,
        le=PROMPT_REGISTER_MAX,
        description=PROMPT_DESC["register_min_oct"],
    )
    register_max_oct: int = Field(
        ...,
        ge=PROMPT_REGISTER_MIN,
        le=PROMPT_REGISTER_MAX,
        description=PROMPT_DESC["register_max_oct"],
    )
    tension_curve: TensionCurve = Field(..., description=PROMPT_DESC["tension_curve"])
    harmony_style: HarmonyStyle = Field(..., description=PROMPT_DESC["harmony_style"])
    chord_change_bars: ChordChangeLabel = Field(..., description=PROMPT_DESC["chord_change_bars"])
    chord_extensions: ChordExtensions = Field(..., description=PROMPT_DESC["chord_extensions"])


class MusicConfigPromptPayload(BaseModel):
    """LLM payload that includes a thinking field, config, and palettes."""

    model_config = ConfigDict(extra="forbid")

    thinking: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=PROMPT_DESC["thinking"],
    )
    title: str = Field(
        ...,
        max_length=MAX_TITLE_CHARS,
        min_length=1,
        description=PROMPT_DESC["title"],
    )
    config: MusicConfigPrompt = Field(..., description=PROMPT_DESC["config"])
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


def schema_signature() -> str:
    """Return the JSON schema as a compact string for prompts."""
    schema = MusicConfigPromptPayload.model_json_schema(by_alias=False)
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def schema_hash() -> str:
    """Return a stable hash of the schema for cache invalidation."""
    signature = schema_signature().encode("utf-8")
    return hashlib.sha256(signature).hexdigest()


def repair_palette_duplicates(palettes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair duplicate hex colors in palettes with deterministic jitter.

    For each palette, if there are duplicate hex values, apply a small RGB offset
    to make them unique while preserving visual similarity.
    """
    result: list[dict[str, Any]] = []

    for palette in palettes:
        colors = palette.get("colors", [])
        seen: dict[str, int] = {}
        repaired_colors: list[dict[str, Any]] = []

        for color in colors:
            hex_val = color.get("hex", "")
            weight = color.get("weight", "md")

            if hex_val in seen:
                # Apply deterministic jitter based on duplicate count
                dup_count = seen[hex_val]
                seen[hex_val] = dup_count + 1
                hex_val = _jitter_hex(hex_val, dup_count)
            else:
                seen[hex_val] = 1

            repaired_colors.append({"hex": hex_val, "weight": weight})

        result.append({"colors": repaired_colors})

    return result


def _jitter_hex(hex_color: str, offset_index: int) -> str:
    """Apply a small deterministic RGB offset to a hex color.

    The offset is based on the duplicate index to ensure unique values.
    """
    if not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color

    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return hex_color

    # Apply small offset (3 per duplicate, wrapping at boundaries)
    step = 3 * offset_index
    r = min(255, max(0, r + step))
    g = min(255, max(0, g + step))
    b = min(255, max(0, b + step))

    return f"#{r:02x}{g:02x}{b:02x}"
