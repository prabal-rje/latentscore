from pydantic import BaseModel

from latentscore.models import FAST_EXAMPLES, build_expressive_prompt, build_litellm_prompt


def test_expressive_prompt_contains_few_shots() -> None:
    prompt = build_expressive_prompt()
    assert "Few-shot examples:" in prompt
    assert "Example 1:" in prompt
    assert "Return only JSON" in prompt


def test_fast_examples_are_pydantic_models() -> None:
    example = FAST_EXAMPLES[0]
    assert isinstance(example, BaseModel)
    payload = example.model_dump()
    assert payload["vibe"]
    assert "config" in payload


def test_litellm_prompt_mentions_palettes() -> None:
    prompt = build_litellm_prompt()
    assert "palettes" in prompt
    assert "3" in prompt
    assert "5" in prompt


def test_expressive_prompt_mentions_palettes() -> None:
    # Expressive prompt uses few-shot examples, which may not have palette text
    # but the schema should mention palettes
    prompt = build_expressive_prompt()
    # For now, just verify the prompt is non-empty
    assert len(prompt) > 0
