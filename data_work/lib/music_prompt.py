"""Prompt builder for music config generation."""

from __future__ import annotations

from data_work.lib.music_schema import schema_signature


def build_music_prompt() -> str:
    schema = schema_signature()
    return (
        "You are an expert sound designer generating JSON configs for a deterministic "
        "ambient/electronic synth engine. Convert the given vibe text into a config payload "
        "that matches the schema exactly. Follow these rules strictly:\n"
        "- Return ONLY valid JSON matching the schema; no extra keys.\n"
        "- Use only the allowed label values from the schema enums.\n"
        "- Keep the justification concise (1-3 sentences, <=1000 chars).\n"
        "- Prefer ambient textures (pads, drones, subtle rhythm); avoid vocals or realistic instruments.\n"
        f"\nSchema:\n{schema}\n"
        "\nReturn JSON only. No prose, no markdown."
    )
