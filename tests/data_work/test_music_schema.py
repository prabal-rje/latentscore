import json

import pytest
from pydantic import ValidationError

from data_work.lib.music_schema import (
    MusicConfigPromptPayload,
    repair_palette_duplicates,
    schema_hash,
    schema_signature,
)


def test_music_schema_validates() -> None:
    payload = {
        "thinking": "Calm, slow, and spacious to match a quiet midnight mood.",
        "title": "Quiet midnight hush",
        "config": {
            "tempo": "slow",
            "root": "d",
            "mode": "minor",
            "brightness": "dark",
            "space": "large",
            "density": 3,
            "bass": "drone",
            "pad": "ambient_drift",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "wide",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "warm",
            "melody_engine": "pattern",
            "phrase_len_bars": 4,
            "melody_density": "medium",
            "syncopation": "light",
            "swing": "none",
            "motif_repeat_prob": "sometimes",
            "step_bias": "balanced",
            "chromatic_prob": "none",
            "cadence_strength": "medium",
            "register_min_oct": 4,
            "register_max_oct": 6,
            "tension_curve": "arc",
            "harmony_style": "auto",
            "chord_change_bars": "medium",
            "chord_extensions": "triads",
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "sm"},
                    {"hex": "#444444", "weight": "xl"},
                    {"hex": "#333333", "weight": "lg"},
                    {"hex": "#222222", "weight": "md"},
                    {"hex": "#000000", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#aa0000", "weight": "xl"},
                    {"hex": "#00aa00", "weight": "lg"},
                    {"hex": "#0000aa", "weight": "md"},
                    {"hex": "#aaaa00", "weight": "sm"},
                    {"hex": "#00aaaa", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#123456", "weight": "xl"},
                    {"hex": "#654321", "weight": "lg"},
                    {"hex": "#abcdef", "weight": "md"},
                    {"hex": "#fedcba", "weight": "sm"},
                    {"hex": "#0f0f0f", "weight": "xs"},
                ]
            },
        ],
    }
    parsed = MusicConfigPromptPayload.model_validate(payload)
    assert parsed.config.tempo == "slow"
    assert parsed.config.density == 3
    assert parsed.thinking.startswith("Calm, slow")
    # Verify auto-sorting by weight (xl -> lg -> md -> sm -> xs)
    assert [c.weight for c in parsed.palettes[0].colors] == ["xl", "lg", "md", "sm", "xs"]


def test_music_schema_requires_title() -> None:
    payload = {
        "thinking": "Calm, slow, and spacious to match a quiet midnight mood.",
        "config": {
            "tempo": "slow",
            "root": "d",
            "mode": "minor",
            "brightness": "dark",
            "space": "large",
            "density": 3,
            "bass": "drone",
            "pad": "ambient_drift",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "wide",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "warm",
            "melody_engine": "pattern",
            "phrase_len_bars": 4,
            "melody_density": "medium",
            "syncopation": "light",
            "swing": "none",
            "motif_repeat_prob": "sometimes",
            "step_bias": "balanced",
            "chromatic_prob": "none",
            "cadence_strength": "medium",
            "register_min_oct": 4,
            "register_max_oct": 6,
            "tension_curve": "arc",
            "harmony_style": "auto",
            "chord_change_bars": "medium",
            "chord_extensions": "triads",
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "sm"},
                    {"hex": "#444444", "weight": "xl"},
                    {"hex": "#333333", "weight": "lg"},
                    {"hex": "#222222", "weight": "md"},
                    {"hex": "#000000", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#aa0000", "weight": "xl"},
                    {"hex": "#00aa00", "weight": "lg"},
                    {"hex": "#0000aa", "weight": "md"},
                    {"hex": "#aaaa00", "weight": "sm"},
                    {"hex": "#00aaaa", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#123456", "weight": "xl"},
                    {"hex": "#654321", "weight": "lg"},
                    {"hex": "#abcdef", "weight": "md"},
                    {"hex": "#fedcba", "weight": "sm"},
                    {"hex": "#0f0f0f", "weight": "xs"},
                ]
            },
        ],
    }
    with pytest.raises(ValidationError):
        MusicConfigPromptPayload.model_validate(payload)


def test_music_schema_rejects_long_title() -> None:
    payload = {
        "thinking": "Calm, slow, and spacious to match a quiet midnight mood.",
        "title": "one two three four five six seven",
        "config": {
            "tempo": "slow",
            "root": "d",
            "mode": "minor",
            "brightness": "dark",
            "space": "large",
            "density": 3,
            "bass": "drone",
            "pad": "ambient_drift",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "wide",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "warm",
            "melody_engine": "pattern",
            "phrase_len_bars": 4,
            "melody_density": "medium",
            "syncopation": "light",
            "swing": "none",
            "motif_repeat_prob": "sometimes",
            "step_bias": "balanced",
            "chromatic_prob": "none",
            "cadence_strength": "medium",
            "register_min_oct": 4,
            "register_max_oct": 6,
            "tension_curve": "arc",
            "harmony_style": "auto",
            "chord_change_bars": "medium",
            "chord_extensions": "triads",
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "sm"},
                    {"hex": "#444444", "weight": "xl"},
                    {"hex": "#333333", "weight": "lg"},
                    {"hex": "#222222", "weight": "md"},
                    {"hex": "#000000", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#aa0000", "weight": "xl"},
                    {"hex": "#00aa00", "weight": "lg"},
                    {"hex": "#0000aa", "weight": "md"},
                    {"hex": "#aaaa00", "weight": "sm"},
                    {"hex": "#00aaaa", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#123456", "weight": "xl"},
                    {"hex": "#654321", "weight": "lg"},
                    {"hex": "#abcdef", "weight": "md"},
                    {"hex": "#fedcba", "weight": "sm"},
                    {"hex": "#0f0f0f", "weight": "xs"},
                ]
            },
        ],
    }
    with pytest.raises(ValidationError):
        MusicConfigPromptPayload.model_validate(payload)


def test_music_schema_rejects_justification_alias() -> None:
    payload = {
        "justification": "Legacy field should fail.",
        "title": "Legacy config",
        "config": {
            "tempo": "slow",
            "root": "d",
            "mode": "minor",
            "brightness": "dark",
            "space": "large",
            "density": 3,
            "bass": "drone",
            "pad": "ambient_drift",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "wide",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "warm",
            "melody_engine": "pattern",
            "phrase_len_bars": 4,
            "melody_density": "medium",
            "syncopation": "light",
            "swing": "none",
            "motif_repeat_prob": "sometimes",
            "step_bias": "balanced",
            "chromatic_prob": "none",
            "cadence_strength": "medium",
            "register_min_oct": 4,
            "register_max_oct": 6,
            "tension_curve": "arc",
            "harmony_style": "auto",
            "chord_change_bars": "medium",
            "chord_extensions": "triads",
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "sm"},
                    {"hex": "#444444", "weight": "xl"},
                    {"hex": "#333333", "weight": "lg"},
                    {"hex": "#222222", "weight": "md"},
                    {"hex": "#000000", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#aa0000", "weight": "xl"},
                    {"hex": "#00aa00", "weight": "lg"},
                    {"hex": "#0000aa", "weight": "md"},
                    {"hex": "#aaaa00", "weight": "sm"},
                    {"hex": "#00aaaa", "weight": "xs"},
                ]
            },
            {
                "colors": [
                    {"hex": "#123456", "weight": "xl"},
                    {"hex": "#654321", "weight": "lg"},
                    {"hex": "#abcdef", "weight": "md"},
                    {"hex": "#fedcba", "weight": "sm"},
                    {"hex": "#0f0f0f", "weight": "xs"},
                ]
            },
        ],
    }
    with pytest.raises(ValidationError):
        MusicConfigPromptPayload.model_validate(payload)


def test_schema_signature_uses_thinking_key() -> None:
    schema = json.loads(schema_signature())
    assert "thinking" in schema["properties"]
    assert "title" in schema["properties"]
    assert "justification" not in schema["properties"]


def test_music_schema_hash_stable() -> None:
    assert schema_hash() == schema_hash()


def test_repair_palette_duplicates_jitters_hexes() -> None:
    palettes = [
        {
            "colors": [
                {"hex": "#111111", "weight": "xl"},
                {"hex": "#111111", "weight": "lg"},
                {"hex": "#111111", "weight": "md"},
                {"hex": "#222222", "weight": "sm"},
                {"hex": "#333333", "weight": "xs"},
            ]
        }
    ] * 3
    repaired = repair_palette_duplicates(palettes)
    hexes = {color["hex"] for color in repaired[0]["colors"]}
    assert len(hexes) == 5
