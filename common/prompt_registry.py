"""Prompt registry for versioned, configurable prompts.

Prompts can be swapped for ablation studies without code changes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from common.music_schema import schema_signature as music_schema_signature
from common.prompts import (
    PALETTE_REQUIREMENTS,
    ROLE_SOUND_DESIGNER,
)


class PromptVersion(BaseModel):
    """A versioned prompt with metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Unique prompt identifier")
    version: str = Field(..., description="Semantic version string")
    description: str = Field(..., description="What this prompt is for")
    template: str = Field(..., description="The prompt template text")


class PromptConfig(BaseModel):
    """Configuration for which prompts to use in training/inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vibe_extraction: str = Field(
        default="vibe_v1",
        description="Prompt version for vibe extraction from text",
    )
    config_generation: str = Field(
        default="config_v1",
        description="Prompt version for config generation from vibes",
    )
    system_prompt: str = Field(
        default="system_v1",
        description="System prompt version for training",
    )


# Prompt registry
_PROMPT_REGISTRY: dict[str, PromptVersion] = {}


def register_prompt(prompt: PromptVersion) -> None:
    """Register a prompt version."""
    _PROMPT_REGISTRY[prompt.name] = prompt


def get_prompt(name: str) -> PromptVersion:
    """Get a registered prompt by name."""
    if name not in _PROMPT_REGISTRY:
        available = list(_PROMPT_REGISTRY.keys())
        raise KeyError(f"Prompt '{name}' not registered. Available: {available}")
    return _PROMPT_REGISTRY[name]


def list_prompts() -> list[str]:
    """List all registered prompt names."""
    return list(_PROMPT_REGISTRY.keys())


# --- Register Default Prompts ---

# Vibe extraction v1 (single-page enforced)
register_prompt(
    PromptVersion(
        name="vibe_v1",
        version="1.0.0",
        description="Vibe extraction with single-page enforcement",
        template="""\
You are an expert data labeler. Read the input text with (Page N) markers and \
return ONLY valid JSON that matches the schema exactly. Follow these rules strictly:
- vibe_index starts at 0 and increments by 1 in output order.
- text_page is a SINGLE integer (0-based page number). You MUST process ONE page at a time. \
Page ranges are FORBIDDEN. If content spans pages, create SEPARATE vibe objects for each page.
- character_name uses real names when present; otherwise use labels like 'anonymous', 'stranger 1', or '3rd person'.
- character_perceived_vibes and scene_vibes use the 5-level descriptor ladder.
- tags are atomic concepts (1-3 words each), lowercase when possible.
- Keep every string concise: <=1000 chars for vibe fields, <=100 chars for short fields.

Schema:
{schema}

Return JSON only. No prose, no extra keys. ONE page per vibe object.""",
    )
)

# Config generation v1
register_prompt(
    PromptVersion(
        name="config_v1",
        version="1.0.0",
        description="Config generation with palette requirements",
        template=f"""\
{ROLE_SOUND_DESIGNER}
Convert the given vibe text into a config payload that matches the schema exactly.

Rules:
- Return ONLY valid JSON matching the schema; no extra keys.
- Use only the allowed label values from the schema enums.
- Keep the justification concise (1-3 sentences, <=1000 chars).
- Prefer ambient textures (pads, drones, subtle rhythm); avoid vocals or realistic instruments.

{PALETTE_REQUIREMENTS}

Schema:
{{schema}}

Return JSON only. No prose, no markdown.""",
    )
)

# System prompt v1
register_prompt(
    PromptVersion(
        name="system_v1",
        version="1.0.0",
        description="Default system prompt for SFT training",
        template="""\
You are a synthesizer configuration assistant. Given a vibe description, \
output a JSON configuration for an ambient/electronic synthesizer. \
Output ONLY valid JSON with no explanation.""",
    )
)


def render_prompt(name: str, **kwargs: str) -> str:
    """Render a registered prompt with variable substitution.

    Args:
        name: Registered prompt name
        **kwargs: Variables to substitute in the template

    Returns:
        Rendered prompt string

    Example:
        >>> render_prompt("vibe_v1", schema=my_schema)
    """
    prompt = get_prompt(name)
    return prompt.template.format(**kwargs)


def render_config_prompt(name: str = "config_v1") -> str:
    """Render a config generation prompt with the current schema.

    Convenience function that automatically includes the music schema.
    """
    schema = music_schema_signature()
    return render_prompt(name, schema=schema)
