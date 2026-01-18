"""Test custom scorer for 02c_score_configs.py testing."""

from typing import Any


def simple_score(vibe: str, config: dict[str, Any]) -> dict[str, Any]:
    """Simple test scorer that checks basic config structure.

    Args:
        vibe: The vibe text
        config: The config dict

    Returns:
        Dict with internal scores and final_score
    """
    has_thinking = "thinking" in config
    has_config = "config" in config
    has_palettes = "palettes" in config

    # Count how many required keys are present
    score = sum([has_thinking, has_config, has_palettes]) / 3.0

    return {
        "has_thinking": int(has_thinking),
        "has_config": int(has_config),
        "has_palettes": int(has_palettes),
        "final_score": score,
    }
