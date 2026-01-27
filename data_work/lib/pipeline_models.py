"""Typed Pydantic models for data_work pipeline rows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

JsonDict = dict[str, JsonValue]

VibeScope = Literal["scene", "character", "raw_text"]
VibeLevel = Literal["xl", "lg", "m", "sm", "xs"]


class BaseRecord(BaseModel):
    """Input record from 01_download_base_data."""

    model_config = ConfigDict(extra="forbid")

    created: str
    metadata: JsonDict
    dataset: str
    id_in_dataset: str | int = Field(alias="id_in_dataset")
    text: str


class VibeRow(BaseModel):
    """Flattened vibe row emitted by 02a_extract_vibes."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    id_in_dataset: str | int
    split: str
    vibe_index: int
    text_page: int
    vibe_scope: VibeScope
    character_name: str | None
    vibe_level: VibeLevel
    vibe_original: str
    vibe_noisy: str
    tags_original: list[str]
    tags_noisy: list[str]
    vibe_model: str


class ValidationScores(BaseModel):
    """Per-candidate validation flags (not quality scores)."""

    model_config = ConfigDict(extra="forbid")

    format_valid: list[int]
    schema_valid: list[int]
    palette_valid: list[int]


class ConfigGenerationRow(VibeRow):
    """Row emitted by 02b_generate_configs."""

    config_model: str
    config_candidates: list[JsonDict | None]
    validation_scores: ValidationScores
    best_index: int
    config_payload: JsonDict | None
    config_error: str | None = None


class ScoredRow(ConfigGenerationRow):
    """Row emitted by 02c_score_configs.

    Note: 02c re-selects best_index based on CLAP scores (highest score among valid candidates),
    overriding the validation-only selection from 02b. The config_payload is updated to match.
    """

    # Per-candidate CLAP scores: {"clap": [0.52, 0.61, None, 0.48, 0.55]}
    # None indicates invalid candidate or scoring error
    candidate_scores: dict[str, list[float | None]] = Field(default_factory=dict)
    # Detailed scores for the selected winner (unchanged from before)
    scores_external: dict[str, JsonDict] = Field(default_factory=dict)
