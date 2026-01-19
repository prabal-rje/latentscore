import importlib
import types

import pytest

gen = importlib.import_module("data_work.02b_generate_configs")


@pytest.mark.asyncio
async def test_call_llm_for_config_wraps_vibe(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        return types.SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "thinking": "",
                "title": "Misty Morning",
                "config": {},
                "palettes": [],
            }
        )

    monkeypatch.setattr(gen, "litellm_structured_completion", fake_completion)

    await gen.call_llm_for_config(
        vibe_text="misty morning",
        model="fake",
        api_key=None,
        api_base=None,
        system_prompt="SYSTEM",
        temperature=0.0,
        use_prompt_caching=False,
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<vibe>" in messages[1]["content"]
