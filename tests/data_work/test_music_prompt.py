"""Tests for data_work music prompt builder."""

from common.prompts import build_config_generation_prompt
from data_work.lib.music_prompt import build_music_prompt


def test_music_prompt_requires_palettes() -> None:
    prompt = build_music_prompt()
    assert "palettes" in prompt
    assert "3" in prompt
    assert "5" in prompt


def test_music_prompt_includes_schema() -> None:
    prompt = build_music_prompt()
    assert "<schema>" in prompt
    assert "</schema>" in prompt
    assert "thinking" in prompt
    assert "config" in prompt


def test_music_prompt_has_no_examples() -> None:
    prompt = build_music_prompt()
    assert "Example 1" not in prompt
    assert "Few-shot examples" not in prompt


def test_music_prompt_batch_suffix_only_when_enabled() -> None:
    base_prompt = build_music_prompt()
    batch_prompt = build_config_generation_prompt(batch=True)
    assert "<batch_response>" not in base_prompt
    assert "<batch_response>" in batch_prompt
