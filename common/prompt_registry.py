"""Prompt registry for versioned, configurable prompts.

Prompts can be swapped for ablation studies without code changes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from common.music_schema import schema_signature as music_schema_signature
from common.prompts import CONFIG_GENERATION_PROMPT_TEMPLATE, VIBE_EXTRACTION_PROMPT_TEMPLATE


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
        version="1.1.0",
        description="Vibe extraction with single-page enforcement",
        template=VIBE_EXTRACTION_PROMPT_TEMPLATE,
    )
)

# Config generation v1
register_prompt(
    PromptVersion(
        name="config_v1",
        version="1.1.0",
        description="Unified config generation prompt",
        template=CONFIG_GENERATION_PROMPT_TEMPLATE,
    )
)

# System prompt v1
register_prompt(
    PromptVersion(
        name="system_v1",
        version="1.1.0",
        description="Default system prompt for SFT training",
        template=CONFIG_GENERATION_PROMPT_TEMPLATE,
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
