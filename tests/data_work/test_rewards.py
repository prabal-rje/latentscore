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
        "thinking": "ok",
        "title": "Test vibe",
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
    reward_config = RewardConfig(
        weights=RewardWeights(
            format_weight=FORMAT_WEIGHT,
            schema_weight=SCHEMA_WEIGHT,
            audio_weight=0.0,
            title_similarity_weight=0.0,
            title_length_penalty_weight=0.0,
            palette_duplicate_penalty=PALETTE_DUPLICATE_PENALTY_WEIGHT,
        )
    )
    breakdown = compute_partial_reward(
        vibe="test", output=json.dumps(payload), config=reward_config
    )

    # With 3 duplicates out of 5 colors per palette, penalty fraction is 2/5 = 0.4
    # Base total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0 = 0.2 + 0.3 = 0.5
    # Expected = max(0.0, 0.5 - 0.2 * 0.4) = 0.5 - 0.08 = 0.42
    base_total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0
    expected = max(0.0, base_total - PALETTE_DUPLICATE_PENALTY_WEIGHT * ((5 - 3) / 5))
    assert abs(breakdown.total - expected) < 0.01


def test_title_similarity_and_length_penalty() -> None:
    payload = {
        "thinking": "ok",
        "title": "neon rain city",
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
        "palettes": [],
    }
    breakdown = compute_partial_reward(vibe="neon rain", output=json.dumps(payload))
    assert breakdown.title_similarity is not None
    assert breakdown.title_similarity > 0.0
    assert breakdown.title_length_penalty == 0.0


def test_no_penalty_for_unique_colors() -> None:
    """Verify no penalty when all colors are unique."""
    payload = {
        "thinking": "ok",
        "title": "Test vibe",
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
    reward_config = RewardConfig(
        weights=RewardWeights(
            format_weight=FORMAT_WEIGHT,
            schema_weight=SCHEMA_WEIGHT,
            audio_weight=0.0,
            title_similarity_weight=0.0,
            title_length_penalty_weight=0.0,
            palette_duplicate_penalty=PALETTE_DUPLICATE_PENALTY_WEIGHT,
        )
    )
    breakdown = compute_partial_reward(
        vibe="test", output=json.dumps(payload), config=reward_config
    )

    # No duplicates, so no penalty
    base_total = FORMAT_WEIGHT * 1.0 + SCHEMA_WEIGHT * 1.0
    assert abs(breakdown.total - base_total) < 0.01


def test_compute_partial_reward_custom_weights() -> None:
    """Verify custom reward weights are applied."""
    valid_payload = {
        "thinking": "test",
        "title": "Test vibe",
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

    # Default weights from config
    default_result = compute_partial_reward("test vibe", valid_json)

    # Custom weights: heavy on format (0.8), light on schema (0.2), no audio/title
    custom_weights = RewardWeights(
        format_weight=0.8,
        schema_weight=0.2,
        audio_weight=0.0,
        title_similarity_weight=0.0,
        title_length_penalty_weight=0.0,
    )
    custom_config = RewardConfig(weights=custom_weights)
    custom_result = compute_partial_reward("test vibe", valid_json, config=custom_config)

    # Both should have format=1.0 and schema=1.0 (valid JSON, all fields valid)
    assert default_result.format == custom_result.format == 1.0
    assert default_result.schema == custom_result.schema == 1.0

    expected_default = (
        DEFAULT_REWARD_CONFIG.weights.format_weight * default_result.format
        + DEFAULT_REWARD_CONFIG.weights.schema_weight * default_result.schema
        + DEFAULT_REWARD_CONFIG.weights.audio_weight * default_result.audio
        + DEFAULT_REWARD_CONFIG.weights.title_similarity_weight * default_result.title_similarity
        - DEFAULT_REWARD_CONFIG.weights.title_length_penalty_weight
        * default_result.title_length_penalty
    )
    assert abs(default_result.total - expected_default) < 0.01

    expected_custom = (
        custom_weights.format_weight * custom_result.format
        + custom_weights.schema_weight * custom_result.schema
        + custom_weights.audio_weight * custom_result.audio
        + custom_weights.title_similarity_weight * custom_result.title_similarity
        - custom_weights.title_length_penalty_weight * custom_result.title_length_penalty
    )
    assert abs(custom_result.total - expected_custom) < 0.01
