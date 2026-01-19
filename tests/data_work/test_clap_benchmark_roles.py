import importlib
import types

import pytest

module = importlib.import_module("data_work.04_clap_benchmark")


@pytest.mark.asyncio
async def test_clap_benchmark_wraps_vibe(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs: object):
        captured.update(kwargs)
        return types.SimpleNamespace()

    monkeypatch.setattr(module, "litellm_structured_completion", fake_completion)

    await module._generate_litellm_payload(
        vibe="quiet dusk",
        model="fake",
        api_key=None,
        api_base=None,
        model_kwargs={},
        system_prompt="SYSTEM",
    )

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<vibe>" in messages[1]["content"]
