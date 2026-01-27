"""CLAP scoring helpers for evaluation."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

# Import ScoreResult protocol for type checking
from data_work.lib.scoring_types import ScoreResult
from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble


class ClapScore(BaseModel):
    """CLAP scoring result for a single vibe-audio pair.

    Implements ScoreResult protocol via final_score property.

    Formula:
        excess_badness = audio_bad_sim - text_bad_sim
        penalty = max(0, excess_badness)  # [0, 2], zero if audio isn't "bad"
        final_score = audio_text_similarity - penalty

    Range: [-3, 1] (higher = better)
        Best:  audio_text_sim=1.0, penalty=0 → 1.0
        Worst: audio_text_sim=-1.0, penalty=2.0 → -3.0
    """

    model_config = ConfigDict(extra="forbid")

    audio_text_similarity: float
    audio_bad_similarity: float
    text_bad_similarity: float
    excess_badness: float
    penalty: float
    raw_score: float
    final_reward: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def final_score(self) -> float:
        """Return final_reward as the canonical final_score.

        This implements the ScoreResult protocol requirement.
        """
        return self.final_reward


# Runtime check that ClapScore implements ScoreResult
def _check_protocol() -> None:
    """Verify ClapScore implements ScoreResult at import time."""
    assert isinstance(
        ClapScore(
            audio_text_similarity=0.0,
            audio_bad_similarity=0.0,
            text_bad_similarity=0.0,
            excess_badness=0.0,
            penalty=0.0,
            raw_score=0.0,
            final_reward=0.0,
        ),
        ScoreResult,
    ), "ClapScore must implement ScoreResult protocol"


_check_protocol()


class ClapScorerProtocol(Protocol):
    """Protocol for CLAP scorers that can score audio against text."""

    def score(self, vibe: str, audio_path: str) -> ClapScore:
        """Score audio against vibe text using CLAP embeddings."""
        ...


BAD_CONCEPTS = [
    "bad",
    "terrible",
    "awful",
    "discordant",
    "unharmonious",
    "cacophony",
    "noise",
    "unpleasant",
    "harsh",
    "grating",
    "off-key",
    "out of tune",
    "distorted badly",
    "broken audio",
    "annoying sound",
    "painful to listen to",
    "low quality audio",
]


class ClapScorer:
    def __init__(self) -> None:
        try:
            import laion_clap  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit(
                "laion_clap is required. Install via data_work/requirements.txt."
            ) from exc

        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()
        self._bad_embedding: np.ndarray | None = None

    def _get_bad_embedding(self) -> np.ndarray:
        if self._bad_embedding is None:
            embeddings = self._model.get_text_embedding(BAD_CONCEPTS)
            self._bad_embedding = embeddings.mean(axis=0, keepdims=True)
        return self._bad_embedding

    @staticmethod
    def _softplus(value: float, beta: float = 1.0) -> float:
        return (1.0 / beta) * math.log(1.0 + math.exp(beta * value))

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        numerator = float(a @ b.T)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return numerator / denom

    def score(self, vibe: str, audio_path: str) -> ClapScore:
        """Score audio against vibe text using CLAP embeddings."""
        text_embed = self._model.get_text_embedding([vibe])
        audio_embed = self._model.get_audio_embedding_from_filelist([audio_path])
        bad_embed = self._get_bad_embedding()

        audio_text_sim = self._cosine_sim(audio_embed, text_embed)
        audio_bad_sim = self._cosine_sim(audio_embed, bad_embed)
        text_bad_sim = self._cosine_sim(text_embed, bad_embed)
        excess_badness = audio_bad_sim - text_bad_sim
        # Penalize only if audio sounds "bad" beyond what vibe implies
        penalty = max(0.0, excess_badness)
        # Direct formula: similarity minus badness penalty
        raw_score = audio_text_sim - penalty
        # For GRPO/Best-of-N, only relative ordering matters
        reward = raw_score

        return ClapScore(
            audio_text_similarity=float(audio_text_sim),
            audio_bad_similarity=float(audio_bad_sim),
            text_bad_similarity=float(text_bad_sim),
            excess_badness=float(excess_badness),
            penalty=float(penalty),
            raw_score=float(raw_score),
            final_reward=reward,
        )


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
