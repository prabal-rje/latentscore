"""Shared prompt constants for music config generation.

These prompts are used consistently across training (data_work) and inference (latentscore).
Import from here to ensure training and inference use identical prompts.
"""

from __future__ import annotations

from common.music_schema import schema_signature

# --- Core Role Descriptions ---

ROLE_MUSIC_EXPERT = "You are a world-class music synthesis expert with deep music theory knowledge."

ROLE_SOUND_DESIGNER = (
    "You are an expert sound designer generating JSON configs for a deterministic "
    "ambient/electronic synth engine."
)


# --- Task Instructions ---

TASK_GENERATE_CONFIG = "Given a vibe/mood description, generate ONE JSON payload with justification, config, and palettes."

TASK_JUSTIFICATION_FIRST = "Place justification before config in the JSON object."

TASK_USE_EXAMPLES = "Use the source examples as guidance for value choices."


# --- Palette Requirements ---

PALETTE_REQUIREMENTS = "\n".join(
    [
        "Palette requirements:",
        "- Include exactly 3 palettes, each with exactly 5 colors.",
        "- Each color needs hex (#RRGGBB) and weight (xs, sm, md, lg, xl, xxl).",
        "- Order colors by weight descending (xxl -> xl -> lg -> md -> sm -> xs).",
        "- Palettes should visually match the vibe/mood.",
    ]
)


# --- Output Requirements ---

OUTPUT_JSON_ONLY = "Return only JSON (no markdown, no explanations)."

OUTPUT_MATCH_SCHEMA = "Return ONLY valid JSON matching the schema; no extra keys."

OUTPUT_USE_ENUMS = "Use only the allowed label values from the schema enums."

OUTPUT_USE_EXAMPLE_KEYS = "Use only the keys shown in the examples."


# --- Style Guidance ---

STYLE_AMBIENT = (
    "Prefer ambient textures (pads, drones, subtle rhythm); avoid vocals or realistic instruments."
)

STYLE_JUSTIFICATION_CONCISE = "Keep the justification concise (1-3 sentences, <=1000 chars)."


# --- Composite Prompts ---


def build_training_prompt() -> str:
    """Build the full system prompt for training data generation.

    This prompt includes the schema signature for strict validation.
    Used by data_work pipeline during training data creation.
    """
    schema = schema_signature()
    return "\n".join(
        [
            ROLE_SOUND_DESIGNER,
            "Convert the given vibe text into a config payload that matches the schema exactly.",
            "",
            "Rules:",
            f"- {OUTPUT_MATCH_SCHEMA}",
            f"- {OUTPUT_USE_ENUMS}",
            f"- {STYLE_JUSTIFICATION_CONCISE}",
            f"- {STYLE_AMBIENT}",
            "",
            PALETTE_REQUIREMENTS,
            "",
            f"Schema:\n{schema}",
            "",
            f"{OUTPUT_JSON_ONLY}",
        ]
    )


def build_inference_prompt() -> str:
    """Build the system prompt for inference (external LLM calls).

    This is the prompt used during inference via LiteLLM.
    Matches training prompt style for consistency.
    """
    return "\n".join(
        [
            f"Role: {ROLE_MUSIC_EXPERT}",
            "",
            "Task:",
            f"- {TASK_GENERATE_CONFIG}",
            f"- {TASK_JUSTIFICATION_FIRST}",
            f"- {TASK_USE_EXAMPLES}",
            "",
            PALETTE_REQUIREMENTS,
            "",
            "Output requirements:",
            f"- {OUTPUT_JSON_ONLY}",
            f"- {OUTPUT_USE_EXAMPLE_KEYS}",
            "",
        ]
    )


def build_expressive_prompt() -> str:
    """Build the system prompt for expressive/local model inference.

    Used by the local MLX model for on-device generation.
    """
    return "\n".join(
        [
            f"Role: {ROLE_MUSIC_EXPERT}",
            "",
            "Task:",
            "- Given a vibe/mood description, generate ONE MusicConfig JSON object.",
            f"- {TASK_USE_EXAMPLES}",
            "",
            "Output requirements:",
            f"- {OUTPUT_JSON_ONLY}",
            f"- {OUTPUT_USE_EXAMPLE_KEYS}",
            "",
        ]
    )


# Aliases for backwards compatibility
build_music_prompt = build_training_prompt
build_litellm_prompt = build_inference_prompt
