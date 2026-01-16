"""Tests for reward helpers including palette duplicate penalty."""

import json

from data_work.lib.rewards import (
    FORMAT_WEIGHT,
    PALETTE_DUPLICATE_PENALTY_WEIGHT,
    SCHEMA_WEIGHT,
    compute_partial_reward,
)


def test_palette_duplicate_penalty_applies() -> None:
    """Verify that duplicate hex colors within palettes reduce the reward."""
    payload = {
        "justification": "ok",
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
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "xl"},
                    {"hex": "#111111", "weight": "lg"},
                    {"hex": "#111111", "weight": "md"},
                    {"hex": "#222222", "weight": "sm"},
                    {"hex": "#333333", "weight": "xs"},
                ]
            }
        ]
        * 3,
    }
    breakdown = compute_partial_reward(vibe="test", output=json.dumps(payload))

    # With 3 duplicates out of 5 colors per palette, penalty fraction is 2/5 = 0.4
    # Base total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0 = 0.2 + 0.3 = 0.5
    # Expected = max(0.0, 0.5 - 0.2 * 0.4) = 0.5 - 0.08 = 0.42
    base_total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0
    expected = max(0.0, base_total - PALETTE_DUPLICATE_PENALTY_WEIGHT * ((5 - 3) / 5))
    assert abs(breakdown.total - expected) < 0.01


def test_no_penalty_for_unique_colors() -> None:
    """Verify no penalty when all colors are unique."""
    payload = {
        "justification": "ok",
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
        },
        "palettes": [
            {
                "colors": [
                    {"hex": "#111111", "weight": "xl"},
                    {"hex": "#222222", "weight": "lg"},
                    {"hex": "#333333", "weight": "md"},
                    {"hex": "#444444", "weight": "sm"},
                    {"hex": "#555555", "weight": "xs"},
                ]
            }
        ]
        * 3,
    }
    breakdown = compute_partial_reward(vibe="test", output=json.dumps(payload))

    # No duplicates, so no penalty
    base_total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0
    assert abs(breakdown.total - base_total) < 0.01
