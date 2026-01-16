"""Tests for data_work music prompt builder."""

from data_work.lib.music_prompt import build_music_prompt


def test_music_prompt_requires_palettes() -> None:
    prompt = build_music_prompt()
    assert "palettes" in prompt
    assert "3" in prompt
    assert "5" in prompt


def test_music_prompt_includes_schema() -> None:
    prompt = build_music_prompt()
    assert "Schema:" in prompt
    assert "justification" in prompt
    assert "config" in prompt
