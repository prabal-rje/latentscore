"""Tests for reward helpers including palette duplicate penalty."""

import json

from common.reward_config import DEFAULT_REWARD_CONFIG, RewardConfig, RewardWeights
from data_work.lib.rewards import compute_partial_reward

# Get weights from default config for backward compatibility
FORMAT_WEIGHT = DEFAULT_REWARD_CONFIG.weights.format_weight
SCHEMA_WEIGHT = DEFAULT_REWARD_CONFIG.weights.schema_weight
PALETTE_DUPLICATE_PENALTY_WEIGHT = DEFAULT_REWARD_CONFIG.weights.palette_duplicate_penalty


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


def test_compute_partial_reward_custom_weights() -> None:
    """Verify custom reward weights are applied."""
    valid_payload = {
        "justification": "test",
        "config": {
            "tempo": "medium",
            "root": "c",
            "mode": "major",
            "brightness": "medium",
            "space": "medium",
            "density": 4,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "minimal",
            "rhythm": "none",
            "texture": "none",
            "accent": "none",
            "motion": "slow",
            "attack": "soft",
            "stereo": "medium",
            "depth": True,
            "echo": "subtle",
            "human": "natural",
            "grain": "clean",
        },
        "palettes": [],
    }
    valid_json = json.dumps(valid_payload)

    # Default weights: format=0.2, schema=0.3, audio=0.5
    default_result = compute_partial_reward("test vibe", valid_json)

    # Custom weights: heavy on format (0.8), light on schema (0.2), no audio
    custom_weights = RewardWeights(
        format_weight=0.8,
        schema_weight=0.2,
        audio_weight=0.0,
    )
    custom_config = RewardConfig(weights=custom_weights)
    custom_result = compute_partial_reward("test vibe", valid_json, config=custom_config)

    # Both should have format=1.0 and schema=1.0 (valid JSON, all fields valid)
    assert default_result.format == custom_result.format == 1.0
    assert default_result.schema == custom_result.schema == 1.0

    # Default total = 0.2 * 1.0 + 0.3 * 1.0 + 0.5 * 0.0 = 0.5
    assert abs(default_result.total - 0.5) < 0.01

    # Custom total = 0.8 * 1.0 + 0.2 * 1.0 + 0.0 * 0.0 = 1.0
    assert abs(custom_result.total - 1.0) < 0.01
