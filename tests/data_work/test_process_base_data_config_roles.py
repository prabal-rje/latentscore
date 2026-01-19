import importlib
import types

import pytest


@pytest.mark.asyncio
async def test_call_llm_for_config_uses_vibe_tag_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("data_work.02b_generate_configs")
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        return types.SimpleNamespace(
            model_dump=lambda *args, **_kwargs: {
                "thinking": "",
                "title": "Foggy Night",
                "config": {},
                "palettes": [],
            }
        )

    monkeypatch.setattr(module, "litellm_structured_completion", fake_completion)

    await module.call_llm_for_config(
        vibe_text="foggy night",
        model="fake",
        api_key=None,
        api_base=None,
        system_prompt="SYSTEM",
        temperature=0.2,
        use_prompt_caching=True,
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<vibe>" in messages[1]["content"]
    assert captured["cache_control_injection_points"] == [{"location": "message", "role": "system"}]
