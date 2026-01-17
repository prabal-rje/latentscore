"""CLAP scoring helpers for evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble


class ClapScore(BaseModel):
    """CLAP scoring result for a single vibe-audio pair."""

    model_config = ConfigDict(extra="forbid")

    audio_text_similarity: float
    audio_bad_similarity: float
    text_bad_similarity: float
    excess_badness: float
    penalty: float
    raw_score: float
    final_reward: float


class ClapScorerProtocol(Protocol):
    """Protocol for CLAP scorers that can score audio against text."""

    def score(self, vibe: str, audio_path: str) -> ClapScore:
        """Score audio against vibe text using CLAP embeddings."""
        ...


def score_config(
    *,
    vibe: str,
    config: dict[str, Any],
    scorer: ClapScorerProtocol,
    duration: float = 12.0,
) -> ClapScore:
    """Score a config against a vibe using CLAP embeddings.

    Args:
        vibe: The vibe text prompt.
        config: A config dict (may be nested MusicConfigPromptPayload format).
        scorer: An initialized ClapScorer instance.
        duration: Audio duration in seconds.

    Returns:
        ClapScore with all scoring metrics.
    """
    # Handle nested config format (MusicConfigPromptPayload has config.config)
    nested = config.get("config")
    if isinstance(nested, dict):
        assert isinstance(nested, dict)
        config_dict = cast(dict[str, Any], nested)
    else:
        config_dict = config

    # Try MusicConfigPrompt first (string labels like "sparse", "light")
    # then convert to MusicConfig via to_config() which handles the label->float mapping
    music_config: MusicConfig
    try:
        prompt_config: MusicConfigPrompt = MusicConfigPrompt.model_validate(config_dict)
        music_config = prompt_config.to_config()
    except Exception:
        # Fallback: maybe it's already in MusicConfig format (numeric values)
        music_config = MusicConfig.model_validate(config_dict)

    internal = music_config.to_internal()
    audio = assemble(internal, duration=duration)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "sample.wav"
        write_wav(audio_path, audio)
        return scorer.score(vibe, str(audio_path))
