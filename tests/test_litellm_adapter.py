from __future__ import annotations

import asyncio
import logging
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from latentscore import MusicConfig
from latentscore.errors import ConfigGenerateError
from latentscore.models import build_expressive_prompt
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

    adapter = LiteLLMAdapter(model="gemini/gemini-3-flash-preview")
    await adapter.generate("warm sunrise")

    assert captured["response_format"] is MusicConfig
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == build_expressive_prompt()


@pytest.mark.asyncio
async def test_litellm_kwargs_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        message = SimpleNamespace(content="{}")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = LiteLLMAdapter(
        model="external:gemini/gemini-3-flash-preview",
        litellm_kwargs={"timeout": 42, "temperature": 0.2},
    )
    await adapter.generate("warm sunrise")

    assert captured["timeout"] == 42
    assert captured["temperature"] == 0.2


def test_litellm_adapter_reuses_loop_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_refs: list[asyncio.AbstractEventLoop] = []

    async def fake_acompletion(**kwargs: object) -> object:
        _ = kwargs
        loop_refs.append(asyncio.get_running_loop())
        message = SimpleNamespace(content="{}")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = LiteLLMAdapter(model="gemini/gemini-3-flash-preview")
    asyncio.run(adapter.generate("warm sunrise"))
    asyncio.run(adapter.generate("late night neon"))

    assert len(loop_refs) == 2
    assert loop_refs[0] is loop_refs[1]


@pytest.mark.asyncio
async def test_litellm_adapter_non_json_error_includes_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: object) -> object:
        message = SimpleNamespace(content="not json output")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    adapter = LiteLLMAdapter(model="gemini/gemini-3-flash-preview")

    with pytest.raises(ConfigGenerateError) as excinfo:
        await adapter.generate("warm sunrise")

    assert "not json output" in str(excinfo.value)


@pytest.mark.asyncio
async def test_litellm_adapter_repairs_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: object) -> object:
        message = SimpleNamespace(content='{"tempo":"slow",}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    def fake_repair_json(content: str) -> str:
        _ = content
        return '{"tempo":"slow"}'

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        "latentscore.providers.litellm.repair_json",
        fake_repair_json,
    )

    adapter = LiteLLMAdapter(model="external:gemini/gemini-3-flash-preview")
    config = await adapter.generate("warm sunrise")

    assert config.tempo == "slow"


def test_litellm_adapter_aclose_awaits_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aclose = AsyncMock()
    monkeypatch.setattr(litellm, "aclose", aclose, raising=False)

    adapter = LiteLLMAdapter(model="gemini/gemini-3-flash-preview")
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


def test_safe_async_cleanup_logs_failures(caplog: pytest.LogCaptureFixture) -> None:
    async def cleanup() -> None:
        raise RuntimeError("boom")

    caplog.set_level(logging.WARNING)
    _safe_async_cleanup(cleanup)

    assert any("boom" in record.getMessage() for record in caplog.records)
