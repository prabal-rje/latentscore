"""Configurable reward weights and scoring parameters for GRPO training."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ScorerProtocol(Protocol):
    """Protocol for custom reward scorers.

    Implement this to create custom reward functions that can be
    plugged into GRPO training instead of the default CLAP scorer.
    """

    def __call__(self, vibe: str, config: Mapping[str, Any]) -> float:
        """Score a generated config against the input vibe.

        Args:
            vibe: The input vibe/mood text
            config: The generated config as a parsed dict

        Returns:
            Score between 0.0 and 1.0
        """
        ...


class RewardWeights(BaseModel):
    """Weights for combining reward components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for JSON format validity (0.0-1.0)",
    )
    schema_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for schema field correctness (0.0-1.0)",
    )
    audio_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for audio/CLAP similarity score (0.0-1.0)",
    )
    title_similarity_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for title-vibe similarity score (0.0-1.0)",
    )
    title_length_penalty_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Penalty weight for title length overage (0.0-1.0)",
    )
    palette_duplicate_penalty: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Penalty weight for duplicate colors in palettes (0.0-1.0)",
    )
    schema_threshold_for_audio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum schema score required before evaluating audio scorer",
    )


class ClapScorerConfig(BaseModel):
    """Configuration for CLAP-based audio scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    softplus_beta_similarity: float = Field(
        default=2.0,
        gt=0.0,
        description="Softplus beta for text-audio similarity scaling",
    )
    softplus_beta_badness: float = Field(
        default=5.0,
        gt=0.0,
        description="Softplus beta for badness penalty scaling",
    )
    badness_penalty_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for badness penalty in raw score",
    )
    exponential_shift: float = Field(
        default=-1.0,
        description="Shift applied before exponential in final score",
    )
    score_clip_min: float = Field(
        default=0.0,
        description="Minimum clipped score",
    )
    score_clip_max: float = Field(
        default=1.0,
        description="Maximum clipped score",
    )


class RewardConfig(BaseModel):
    """Complete reward configuration for GRPO training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: RewardWeights = Field(default_factory=RewardWeights)
    clap_config: ClapScorerConfig = Field(default_factory=ClapScorerConfig)
    reward_type: str = Field(
        default="clap",
        pattern="^(clap|schema_only|custom)$",
        description="Reward function type: 'clap', 'schema_only', or 'custom'",
    )
    custom_scorer_path: str | None = Field(
        default=None,
        description="Python path to custom scorer (e.g., 'my_module:my_scorer')",
    )


# Default configuration instance
DEFAULT_REWARD_CONFIG = RewardConfig()
