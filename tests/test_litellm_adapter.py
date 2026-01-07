from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from latentscore.errors import ConfigGenerateError
from latentscore.providers.litellm import LiteLLMAdapter, _safe_async_cleanup


@pytest.mark.asyncio
async def test_litellm_adapter_enforces_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        message = SimpleNamespace(content="{}")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash")
    await adapter.generate("warm sunrise")

    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_litellm_adapter_non_json_error_includes_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: object) -> object:
        message = SimpleNamespace(content="not json output")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash")

    with pytest.raises(ConfigGenerateError) as excinfo:
        await adapter.generate("warm sunrise")

    assert "not json output" in str(excinfo.value)


def test_litellm_adapter_aclose_awaits_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aclose = AsyncMock()
    monkeypatch.setattr(litellm, "aclose", aclose, raising=False)

    adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash")
    asyncio.run(adapter.aclose())

    aclose.assert_awaited_once()


def test_safe_async_cleanup_uses_asyncio_run_for_closed_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.close()
    ran = {"value": False}

    async def cleanup() -> None:
        ran["value"] = True

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            _safe_async_cleanup(cleanup)
    finally:
        asyncio.set_event_loop(None)

    assert ran["value"] is True
    assert not [warn for warn in captured if issubclass(warn.category, RuntimeWarning)]
