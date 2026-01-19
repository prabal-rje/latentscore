import importlib
import types

import pytest

gen = importlib.import_module("data_work.02b_generate_configs")


@pytest.mark.asyncio
async def test_batch_call_uses_batch_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        model = types.SimpleNamespace(
            model_dump=lambda: {
                "generated_config_0": {"thinking": "", "config": {}, "palettes": []}
            }
        )
        return model

    monkeypatch.setattr(gen, "litellm_structured_completion", fake_completion)
    monkeypatch.setattr(
        gen,
        "parse_batch_response",
        lambda _response, _size: [types.SimpleNamespace(model_dump=lambda **_kwargs: {})],
    )

    await gen.call_music_config_batch(
        vibe_texts=["misty morning"],
        model="fake",
        api_key=None,
        api_base=None,
        system_prompt="SYSTEM\n\n<batch_response>stub</batch_response>",
        model_kwargs={},
        use_prompt_caching=True,
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert "<batch_response>" in messages[0]["content"]


@pytest.mark.asyncio
async def test_batch_call_enables_prompt_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        model = types.SimpleNamespace(
            model_dump=lambda: {
                "generated_config_0": {"thinking": "", "config": {}, "palettes": []}
            }
        )
        return model

    monkeypatch.setattr(gen, "litellm_structured_completion", fake_completion)
    monkeypatch.setattr(
        gen,
        "parse_batch_response",
        lambda _response, _size: [types.SimpleNamespace(model_dump=lambda **_kwargs: {})],
    )

    await gen.call_music_config_batch(
        vibe_texts=["fog"],
        model="fake",
        api_key=None,
        api_base=None,
        system_prompt="SYSTEM\n\n<batch_response>stub</batch_response>",
        model_kwargs={},
        use_prompt_caching=True,
    )

    cache_points = captured["cache_control_injection_points"]
    assert cache_points[0]["role"] == "system"
