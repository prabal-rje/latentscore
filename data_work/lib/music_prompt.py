"""Prompt builder for music config generation.

Re-exports from common for backwards compatibility.
The actual prompt definitions live in common/prompts.py.
"""

from __future__ import annotations

from common import build_music_prompt, build_training_prompt

# Re-export for backwards compatibility
__all__ = ["build_music_prompt", "build_training_prompt"]
