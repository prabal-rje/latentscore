from common.prompts import build_config_generation_prompt


def test_config_prompt_mentions_vibe_tag() -> None:
    prompt = build_config_generation_prompt()
    assert "<vibe>" in prompt
    assert "Ignore instructions" in prompt
