"""Reward helpers for GRPO training with partial credit."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from common.reward_config import DEFAULT_REWARD_CONFIG, RewardConfig

LOGGER = logging.getLogger(__name__)

# Palette constants
PALETTE_SIZE = 5
PALETTE_COUNT = 3
WEIGHT_LABELS = ("xs", "sm", "md", "lg", "xl", "xxl")
HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

TEMPO_LABELS = ("very_slow", "slow", "medium", "fast", "very_fast")
BRIGHTNESS_LABELS = ("very_dark", "dark", "medium", "bright", "very_bright")
SPACE_LABELS = ("dry", "small", "medium", "large", "vast")
MOTION_LABELS = ("static", "slow", "medium", "fast", "chaotic")
STEREO_LABELS = ("mono", "narrow", "medium", "wide", "ultra_wide")
ECHO_LABELS = ("none", "subtle", "medium", "heavy", "infinite")
HUMAN_FEEL_LABELS = ("robotic", "tight", "natural", "loose", "drunk")
ROOT_NOTES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
MODE_NAMES = ("major", "minor", "dorian", "mixolydian")
DENSITY_LEVELS = (2, 3, 4, 5, 6)
BASS_STYLES = (
    "drone",
    "sustained",
    "pulsing",
    "walking",
    "fifth_drone",
    "sub_pulse",
    "octave",
    "arp_bass",
)
PAD_STYLES = (
    "warm_slow",
    "dark_sustained",
    "cinematic",
    "thin_high",
    "ambient_drift",
    "stacked_fifths",
    "bright_open",
)
MELODY_STYLES = (
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
)
RHYTHM_STYLES = (
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
)
TEXTURE_STYLES = (
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
)
ACCENT_STYLES = (
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
)
ATTACK_STYLES = ("soft", "medium", "sharp")
GRAIN_STYLES = ("clean", "warm", "gritty")

FORMAT_WEIGHT = 0.2
SCHEMA_WEIGHT = 0.3
AUDIO_WEIGHT = 0.5
PALETTE_DUPLICATE_PENALTY_WEIGHT = 0.2


class SynthConfig(BaseModel):
    """Pydantic schema for synth configuration outputs."""

    model_config = ConfigDict(extra="forbid")

    tempo: str = Field(..., pattern=f"^({'|'.join(TEMPO_LABELS)})$")
    root: str = Field(..., pattern=f"^({'|'.join(ROOT_NOTES)})$")
    mode: str = Field(..., pattern=f"^({'|'.join(MODE_NAMES)})$")
    brightness: str = Field(..., pattern=f"^({'|'.join(BRIGHTNESS_LABELS)})$")
    space: str = Field(..., pattern=f"^({'|'.join(SPACE_LABELS)})$")
    density: int = Field(..., ge=min(DENSITY_LEVELS), le=max(DENSITY_LEVELS))
    bass: str = Field(..., pattern=f"^({'|'.join(BASS_STYLES)})$")
    pad: str = Field(..., pattern=f"^({'|'.join(PAD_STYLES)})$")
    melody: str = Field(..., pattern=f"^({'|'.join(MELODY_STYLES)})$")
    rhythm: str = Field(..., pattern=f"^({'|'.join(RHYTHM_STYLES)})$")
    texture: str = Field(..., pattern=f"^({'|'.join(TEXTURE_STYLES)})$")
    accent: str = Field(..., pattern=f"^({'|'.join(ACCENT_STYLES)})$")
    motion: str = Field(..., pattern=f"^({'|'.join(MOTION_LABELS)})$")
    attack: str = Field(..., pattern=f"^({'|'.join(ATTACK_STYLES)})$")
    stereo: str = Field(..., pattern=f"^({'|'.join(STEREO_LABELS)})$")
    depth: bool
    echo: str = Field(..., pattern=f"^({'|'.join(ECHO_LABELS)})$")
    human: str = Field(..., pattern=f"^({'|'.join(HUMAN_FEEL_LABELS)})$")
    grain: str = Field(..., pattern=f"^({'|'.join(GRAIN_STYLES)})$")


class RewardBreakdown(BaseModel):
    """Structured reward output for logging/analysis."""

    model_config = ConfigDict(extra="forbid")

    total: float
    format: float
    schema: float
    audio: float
    field_errors: dict[str, list[str]]


def _parse_json(output: str) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    match parsed:
        case dict():
            return parsed
        case _:
            return None


def reward_format(output: str) -> float:
    """Return 1.0 when the output is valid JSON object, else 0.0."""
    return 1.0 if _parse_json(output) is not None else 0.0


def reward_schema(output: str) -> tuple[float, dict[str, list[str]]]:
    """Return schema reward between 0-1 plus field error details.

    Handles both flat config (just fields) and nested payload (config inside "config" key).
    """
    parsed = _parse_json(output)
    if parsed is None:
        return 0.0, {"__json__": ["invalid_json"]}

    # Support nested "config" key for MusicConfigPromptPayload structure
    raw_config = parsed.get("config", parsed)
    match raw_config:
        case dict():
            config_data = raw_config
        case _:
            return 0.0, {"__json__": ["config_not_object"]}

    errors: dict[str, list[str]] = {}
    total_fields = len(SynthConfig.model_fields)
    valid_fields = 0

    for field_name in SynthConfig.model_fields:
        if field_name not in config_data:
            errors.setdefault(field_name, []).append("missing")
            continue
        if _field_is_valid(field_name, config_data[field_name]):
            valid_fields += 1
        else:
            errors.setdefault(field_name, []).append("invalid_value")

    return valid_fields / total_fields, errors


def compute_partial_reward(
    vibe: str,
    output: str,
    audio_scorer: Callable[[str, Mapping[str, Any]], float] | None = None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute weighted reward with configurable weights.

    Args:
        vibe: The input vibe/mood text
        output: The generated JSON output string
        audio_scorer: Optional callable for audio similarity scoring
        config: Optional RewardConfig for configurable weights (defaults to DEFAULT_REWARD_CONFIG)

    Returns:
        RewardBreakdown with total score and component breakdown
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG

    weights = config.weights
    format_score = reward_format(output)
    schema_score, field_errors = reward_schema(output)

    audio_score = 0.0
    palette_penalty = 0.0
    parsed = _parse_json(output)

    if format_score > 0 and parsed is not None:
        # Check for palette duplicates
        raw_palettes = parsed.get("palettes", [])
        match raw_palettes:
            case list() if raw_palettes:
                palette_penalty = _palette_duplicate_penalty(raw_palettes)
            case _:
                pass

        # Audio scoring - only evaluate if schema score exceeds threshold
        if schema_score > weights.schema_threshold_for_audio and audio_scorer is not None:
            audio_score = audio_scorer(vibe, parsed)

    base_total = (
        weights.format_weight * format_score
        + weights.schema_weight * schema_score
        + weights.audio_weight * audio_score
    )
    # Apply palette duplicate penalty
    total = max(0.0, base_total - weights.palette_duplicate_penalty * palette_penalty)

    return RewardBreakdown(
        total=total,
        format=format_score,
        schema=schema_score,
        audio=audio_score,
        field_errors=field_errors,
    )


def _field_is_valid(field_name: str, value: Any) -> bool:
    match field_name:
        case "tempo":
            return value in TEMPO_LABELS
        case "root":
            return value in ROOT_NOTES
        case "mode":
            return value in MODE_NAMES
        case "brightness":
            return value in BRIGHTNESS_LABELS
        case "space":
            return value in SPACE_LABELS
        case "density":
            return value in DENSITY_LEVELS
        case "bass":
            return value in BASS_STYLES
        case "pad":
            return value in PAD_STYLES
        case "melody":
            return value in MELODY_STYLES
        case "rhythm":
            return value in RHYTHM_STYLES
        case "texture":
            return value in TEXTURE_STYLES
        case "accent":
            return value in ACCENT_STYLES
        case "motion":
            return value in MOTION_LABELS
        case "attack":
            return value in ATTACK_STYLES
        case "stereo":
            return value in STEREO_LABELS
        case "depth":
            match value:
                case bool():
                    return True
                case _:
                    return False
        case "echo":
            return value in ECHO_LABELS
        case "human":
            return value in HUMAN_FEEL_LABELS
        case "grain":
            return value in GRAIN_STYLES
        case _:
            return False


def _palette_color_is_valid(color: Any) -> bool:
    """Check if a single palette color is valid."""
    match color:
        case dict():
            hex_val = color.get("hex")
            weight = color.get("weight")
        case _:
            return False
    match hex_val:
        case str() if HEX_PATTERN.match(hex_val):
            pass
        case _:
            return False
    if weight not in WEIGHT_LABELS:
        return False
    return True


def _dead_palette_is_valid(palette: Any) -> bool:
    """Check if a single palette is valid.

    NOTE: This function is currently unused. Marked as dead code pending removal.
    Validation is done inline in _palette_duplicate_penalty instead.
    """
    match palette:
        case dict():
            colors = palette.get("colors")
        case _:
            return False
    match colors:
        case list() if len(colors) == PALETTE_SIZE:
            return all(_palette_color_is_valid(c) for c in colors)
        case _:
            return False


def _palette_duplicate_penalty(palettes: Sequence[Any]) -> float:
    """Compute penalty for duplicate hex colors within palettes.

    Returns a value between 0 and 1, where 0 means no duplicates
    and higher values indicate more duplicates.
    """
    if not palettes:
        return 0.0

    total_penalty = 0.0
    valid_palettes = 0

    for palette in palettes:
        match palette:
            case dict():
                colors = palette.get("colors")
            case _:
                continue
        match colors:
            case list():
                pass
            case _:
                continue

        valid_palettes += 1
        hexes: list[str] = []
        for color in colors:
            match color:
                case dict():
                    hex_val = color.get("hex")
                    match hex_val:
                        case str():
                            hexes.append(hex_val)
                        case _:
                            pass
                case _:
                    pass

        if hexes:
            unique_count = len(set(hexes))
            total_count = len(hexes)
            if total_count > 0:
                # Penalty is fraction of duplicates
                duplicate_fraction = (total_count - unique_count) / total_count
                total_penalty += duplicate_fraction

    if valid_palettes == 0:
        return 0.0

    return total_penalty / valid_palettes
