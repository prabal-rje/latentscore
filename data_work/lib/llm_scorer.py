"""LLM-based audio scoring using multimodal models (Gemini, Voxtral)."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_work.lib.clap_scorer import ClapScore, ClapScorerProtocol
from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble

LOGGER = logging.getLogger("data_work.llm_scorer")

# Scoring prompt template
DEFAULT_SCORING_PROMPT = """You are an expert music critic evaluating ambient/electronic music.

Listen to the audio and evaluate how well it matches the intended vibe/mood description.

**Intended Vibe:** {vibe}

Score the audio on these dimensions (each 0.0 to 1.0):

1. **vibe_match** (0.0-1.0): How well does the audio capture the intended vibe/mood?
   - 0.0 = Completely mismatched, wrong mood entirely
   - 0.5 = Partially captures the vibe, some elements work
   - 1.0 = Perfectly captures the intended vibe/mood

2. **audio_quality** (0.0-1.0): Technical quality of the audio
   - 0.0 = Harsh, grating, painful to listen to
   - 0.5 = Acceptable quality, some rough edges
   - 1.0 = Clean, well-produced, pleasant to listen to

3. **coherence** (0.0-1.0): How coherent and intentional does the music sound?
   - 0.0 = Random noise, no musical structure
   - 0.5 = Some structure but disjointed
   - 1.0 = Coherent, intentional musical piece

4. **creativity** (0.0-1.0): How creative/interesting is the interpretation?
   - 0.0 = Generic, boring, predictable
   - 0.5 = Decent interpretation, expected choices
   - 1.0 = Creative, surprising, engaging interpretation

Provide a brief justification for your scores."""


class LLMScoreResult(BaseModel):
    """Structured output for LLM-based audio scoring."""

    model_config = ConfigDict(extra="forbid")

    vibe_match: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the audio captures the intended vibe (0.0-1.0)",
    )
    audio_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Technical quality of the audio (0.0-1.0)",
    )
    coherence: float = Field(
        ge=0.0,
        le=1.0,
        description="How coherent and intentional the music sounds (0.0-1.0)",
    )
    creativity: float = Field(
        ge=0.0,
        le=1.0,
        description="How creative/interesting the interpretation is (0.0-1.0)",
    )
    justification: str = Field(
        max_length=500,
        description="Brief explanation of the scores",
    )


def _encode_audio_base64(audio_path: str) -> str:
    """Encode audio file as base64 for LLM input."""
    with open(audio_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _build_audio_message(
    vibe: str,
    audio_path: str,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Build messages with audio content for multimodal LLM."""
    audio_base64 = _encode_audio_base64(audio_path)
    prompt = prompt_template.format(vibe=vibe)

    # Format for Gemini/LiteLLM multimodal input
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_base64}",
                    },
                },
            ],
        }
    ]


def _llm_score_to_clap_score(result: LLMScoreResult) -> ClapScore:
    """Convert LLM score result to ClapScore for compatibility."""
    # Map LLM scores to ClapScore fields:
    # - vibe_match -> audio_text_similarity (main similarity metric)
    # - audio_quality penalty -> excess_badness (inverted)
    # - Compute final_reward as weighted average

    audio_text_sim = result.vibe_match
    audio_quality = result.audio_quality

    # Badness is inverse of quality
    audio_bad_sim = 1.0 - audio_quality
    text_bad_sim = 0.0  # Assume vibe text isn't inherently "bad"
    excess_badness = max(0.0, audio_bad_sim - 0.3)  # Penalty if quality < 0.7
    penalty = excess_badness * 0.5

    # Weighted final score
    weights = {
        "vibe_match": 0.5,
        "audio_quality": 0.2,
        "coherence": 0.2,
        "creativity": 0.1,
    }
    raw_score = (
        result.vibe_match * weights["vibe_match"]
        + result.audio_quality * weights["audio_quality"]
        + result.coherence * weights["coherence"]
        + result.creativity * weights["creativity"]
    )
    final_reward = max(0.0, min(1.0, raw_score - penalty))

    return ClapScore(
        audio_text_similarity=audio_text_sim,
        audio_bad_similarity=audio_bad_sim,
        text_bad_similarity=text_bad_sim,
        excess_badness=excess_badness,
        penalty=penalty,
        raw_score=raw_score,
        final_reward=final_reward,
    )


class LLMScorer:
    """Score audio-vibe alignment using multimodal LLMs (Gemini, Voxtral)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        prompt_template: str = DEFAULT_SCORING_PROMPT,
        temperature: float = 0.0,
    ) -> None:
        """Initialize LLM scorer.

        Args:
            model: LiteLLM model identifier (e.g., "gemini/gemini-2.0-flash",
                   "mistral/voxtral-small-latest", "openrouter/mistral/voxtral-small")
            api_key: API key for the provider
            api_base: Optional API base URL override
            prompt_template: Scoring prompt template with {vibe} placeholder
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._prompt_template = prompt_template
        self._temperature = temperature

    def score(self, vibe: str, audio_path: str) -> ClapScore:
        """Score audio against vibe text using LLM.

        Args:
            vibe: The vibe/mood description text
            audio_path: Path to the audio file (WAV)

        Returns:
            ClapScore with scoring metrics (compatible with CLAP interface)
        """
        import asyncio

        return asyncio.run(self.score_async(vibe, audio_path))

    async def score_async(self, vibe: str, audio_path: str) -> ClapScore:
        """Async version of score()."""
        from data_work.lib.llm_client import litellm_structured_completion

        messages = _build_audio_message(vibe, audio_path, self._prompt_template)

        result = await litellm_structured_completion(
            model=self._model,
            messages=messages,
            response_model=LLMScoreResult,
            api_key=self._api_key,
            api_base=self._api_base,
            model_kwargs={},
            temperature=self._temperature,
        )

        LOGGER.debug(
            "LLM score for '%s': vibe=%.2f quality=%.2f coherence=%.2f creativity=%.2f",
            vibe[:30],
            result.vibe_match,
            result.audio_quality,
            result.coherence,
            result.creativity,
        )

        return _llm_score_to_clap_score(result)

    def score_detailed(self, vibe: str, audio_path: str) -> LLMScoreResult:
        """Get detailed scoring result (not just ClapScore).

        Use this to get the full LLMScoreResult with justification.
        """
        import asyncio

        return asyncio.run(self.score_detailed_async(vibe, audio_path))

    async def score_detailed_async(self, vibe: str, audio_path: str) -> LLMScoreResult:
        """Async version of score_detailed()."""
        from data_work.lib.llm_client import litellm_structured_completion

        messages = _build_audio_message(vibe, audio_path, self._prompt_template)

        return await litellm_structured_completion(
            model=self._model,
            messages=messages,
            response_model=LLMScoreResult,
            api_key=self._api_key,
            api_base=self._api_base,
            model_kwargs={},
            temperature=self._temperature,
        )


def _config_to_audio_path(
    config: dict[str, Any],
    duration: float,
    tmpdir: str,
) -> str:
    """Convert config dict to audio file, return path."""
    # Handle nested config format
    nested = config.get("config")
    if isinstance(nested, dict):
        config_dict = nested
    else:
        config_dict = config

    # Convert to MusicConfig (handle string labels)
    music_config: MusicConfig
    try:
        prompt_config: MusicConfigPrompt = MusicConfigPrompt.model_validate(config_dict)
        music_config = prompt_config.to_config()
    except Exception:
        music_config = MusicConfig.model_validate(config_dict)

    internal = music_config.to_internal()
    audio = assemble(internal, duration=duration)

    audio_path = Path(tmpdir) / "sample.wav"
    write_wav(audio_path, audio)
    return str(audio_path)


def score_config_with_llm(
    *,
    vibe: str,
    config: dict[str, Any],
    scorer: LLMScorer,
    duration: float = 12.0,
) -> ClapScore:
    """Score a config against a vibe using LLM-based scoring.

    Args:
        vibe: The vibe text prompt.
        config: A config dict (may be nested MusicConfigPromptPayload format).
        scorer: An initialized LLMScorer instance.
        duration: Audio duration in seconds.

    Returns:
        ClapScore with all scoring metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _config_to_audio_path(config, duration, tmpdir)
        return scorer.score(vibe, audio_path)


def score_config_with_llm_detailed(
    *,
    vibe: str,
    config: dict[str, Any],
    scorer: LLMScorer,
    duration: float = 12.0,
) -> LLMScoreResult:
    """Score a config against a vibe using LLM-based scoring, return detailed result.

    Args:
        vibe: The vibe text prompt.
        config: A config dict (may be nested MusicConfigPromptPayload format).
        scorer: An initialized LLMScorer instance.
        duration: Audio duration in seconds.

    Returns:
        LLMScoreResult with all detailed scores and justification.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _config_to_audio_path(config, duration, tmpdir)
        return scorer.score_detailed(vibe, audio_path)


async def score_config_with_llm_detailed_async(
    *,
    vibe: str,
    config: dict[str, Any],
    scorer: LLMScorer,
    duration: float = 12.0,
) -> LLMScoreResult:
    """Async version of score_config_with_llm_detailed()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _config_to_audio_path(config, duration, tmpdir)
        return await scorer.score_detailed_async(vibe, audio_path)


# Type alias for scorer protocol compliance
LLMScorerProtocol = ClapScorerProtocol


def get_llm_scorer(
    model: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> LLMScorer:
    """Factory function to create an LLM scorer.

    Supported models (multimodal with audio input):
    - Gemini 3 Flash Preview: "gemini/gemini-3-flash-preview"
    - Gemini 3 Pro Preview: "gemini/gemini-3-pro-preview"
    - Voxtral Small (24B): "mistral/voxtral-small-latest"
    - Voxtral Mini (3B): "mistral/voxtral-mini-latest"

    Args:
        model: LiteLLM model identifier
        api_key: API key (or set via environment: GEMINI_API_KEY, MISTRAL_API_KEY)
        api_base: Optional API base URL

    Returns:
        Configured LLMScorer instance
    """
    return LLMScorer(model=model, api_key=api_key, api_base=api_base)
