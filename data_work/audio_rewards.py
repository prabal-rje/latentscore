"""Audio reward helpers for GRPO training."""

from __future__ import annotations

import os
from typing import Any, Mapping

from data_work.lib.clap_scorer import ClapScorer, score_config

_SCORER: ClapScorer | None = None


def _get_scorer() -> ClapScorer:
    """Lazy-load and cache the CLAP scorer."""
    global _SCORER
    if _SCORER is None:
        _SCORER = ClapScorer()
    return _SCORER


def score_clap(vibe: str, config: Mapping[str, Any]) -> float:
    """Return CLAP reward score for a vibe + config pair.

    AUDIO_REWARD_DURATION env var controls audio synthesis length (seconds).
    """
    duration = float(os.environ.get("AUDIO_REWARD_DURATION", "30.0"))
    scorer = _get_scorer()
    result = score_config(
        vibe=vibe,
        config=dict(config),
        scorer=scorer,
        duration=duration,
    )
    return float(result.final_reward)
