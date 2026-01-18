from pydantic import BaseModel

from common import build_training_prompt
from latentscore.models import FAST_EXAMPLES, build_expressive_prompt, build_litellm_prompt


def test_fast_examples_are_pydantic_models() -> None:
    example = FAST_EXAMPLES[0]
    assert isinstance(example, BaseModel)
    payload = example.model_dump()
    assert payload["vibe"]
    assert "config" in payload


def test_prompt_builders_match_common() -> None:
    base_prompt = build_training_prompt()
    assert build_litellm_prompt() == base_prompt
    assert build_expressive_prompt() == base_prompt


def test_prompt_mentions_palettes_and_schema() -> None:
    prompt = build_training_prompt()
    assert "palettes" in prompt
    assert "<schema>" in prompt
