"""Schema for generating music configuration payloads from vibe text."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_LONG_FIELD_CHARS = 1_000

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


class MusicConfigPrompt(BaseModel):
    """LLM-facing config for the synth engine (label-only)."""

    model_config = ConfigDict(extra="forbid")

    tempo: TempoLabel = Field(
        ...,
        description="Tempo label for the overall pace of the sound.",
    )
    root: RootNote = Field(
        ...,
        description="Root note for the harmonic center (pitch class).",
    )
    mode: ModeName = Field(
        ...,
        description="Scale mode describing the harmonic flavor (major/minor/etc).",
    )
    brightness: BrightnessLabel = Field(
        ...,
        description="Spectral brightness label for tone color.",
    )
    space: SpaceLabel = Field(
        ...,
        description="Perceived room size or reverb space label.",
    )
    density: DensityLevel = Field(
        ...,
        description="Layer density level (2-6) for how busy the texture feels.",
    )
    bass: BassStyle = Field(
        ...,
        description="Bass layer style (drone/pulsing/arp, etc.).",
    )
    pad: PadStyle = Field(
        ...,
        description="Pad layer style describing sustained harmonic texture.",
    )
    melody: MelodyStyle = Field(
        ...,
        description="Melody layer style for lead motifs or arpeggios.",
    )
    rhythm: RhythmStyle = Field(
        ...,
        description="Rhythmic layer style; use 'none' for no drums.",
    )
    texture: TextureStyle = Field(
        ...,
        description="Textural layer style such as shimmer, crackle, or noise wash.",
    )
    accent: AccentStyle = Field(
        ...,
        description="Accent or hit layer style; use 'none' for no accents.",
    )
    motion: MotionLabel = Field(
        ...,
        description="Motion label for modulation rate and movement intensity.",
    )
    attack: AttackStyle = Field(
        ...,
        description="Envelope attack style for transient sharpness.",
    )
    stereo: StereoLabel = Field(
        ...,
        description="Stereo width label from mono to ultra-wide.",
    )
    depth: bool = Field(
        ...,
        description="Whether to add depth modulation (true/false).",
    )
    echo: EchoLabel = Field(
        ...,
        description="Echo intensity label from none to infinite.",
    )
    human: HumanFeelLabel = Field(
        ...,
        description="Humanization label for timing/feel looseness.",
    )
    grain: GrainStyle = Field(
        ...,
        description="Grain style for noise/lo-fi texture.",
    )


class MusicConfigPromptPayload(BaseModel):
    """Top-level payload returned by the LLM for a vibe."""

    model_config = ConfigDict(extra="forbid")

    justification: str = Field(
        ...,
        max_length=MAX_LONG_FIELD_CHARS,
        description=(
            "Short reasoning linking the vibe to the chosen labels. "
            "Keep it concise (1-3 sentences)."
        ),
    )
    config: MusicConfigPrompt = Field(
        ...,
        description="The JSON config payload with label-only fields.",
    )


def schema_signature() -> str:
    schema = MusicConfigPromptPayload.model_json_schema()
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def schema_hash() -> str:
    signature = schema_signature().encode("utf-8")
    return hashlib.sha256(signature).hexdigest()
