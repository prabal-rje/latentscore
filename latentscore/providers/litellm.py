from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from pydantic import ValidationError

from ..config import MusicConfig
from ..errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError

_DEFAULT_TEMPERATURE = 0.2
_JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
_atexit_guard_registered = False
_safe_get_event_loop_installed = False


def _safe_async_cleanup(
    cleanup_coro: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        try:
            loop.create_task(cleanup_coro())
        except Exception:
            return
        return

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is None or loop.is_closed():
        try:
            asyncio.run(cleanup_coro())
        except Exception:
            return
        return

    try:
        loop.run_until_complete(cleanup_coro())
    except Exception:
        return


def _install_safe_get_event_loop() -> None:
    global _safe_get_event_loop_installed
    if _safe_get_event_loop_installed:
        return
    _safe_get_event_loop_installed = True

    original_get_event_loop = asyncio.get_event_loop

    def safe_get_event_loop() -> asyncio.AbstractEventLoop:
        try:
            loop = original_get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    asyncio.get_event_loop = safe_get_event_loop  # type: ignore[assignment]

    try:
        from litellm.llms.custom_httpx.async_client_cleanup import (
            close_litellm_async_clients,
        )
    except Exception:
        return
    _safe_async_cleanup(close_litellm_async_clients)


def _register_safe_get_event_loop_atexit() -> None:
    global _atexit_guard_registered
    if _atexit_guard_registered:
        return
    _atexit_guard_registered = True
    import atexit

    atexit.register(_install_safe_get_event_loop)


def _extract_json_payload(content: str) -> str | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1]


def _content_snippet(content: str, limit: int = 200) -> str:
    cleaned = content.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


class LiteLLMAdapter:
    """LiteLLM model wrapper implementing the async model protocol."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        system_prompt: str | None = None,
        response_format: dict[str, str] | None = _JSON_RESPONSE_FORMAT,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._response_format = response_format

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        loop.create_task(self.aclose())

    async def aclose(self) -> None:
        try:
            import litellm  # type: ignore[import]  # Optional dependency at runtime.
        except ImportError:
            return

        _register_safe_get_event_loop_atexit()
        close_fn: Any = getattr(litellm, "aclose", None)
        if close_fn is None:
            close_fn = getattr(litellm, "close_litellm_async_clients", None)
        if close_fn is None:
            return
        await close_fn()

    async def generate(self, vibe: str) -> MusicConfig:
        try:
            from litellm import acompletion  # type: ignore[import]
        except ImportError as exc:
            raise ModelNotAvailableError("litellm is not installed") from exc

        _register_safe_get_event_loop_atexit()
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append(
            {
                "role": "user",
                "content": f"Generate config for: {vibe}\nReturn only JSON.",
            }
        )

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._response_format is not None:
            request["response_format"] = self._response_format
        if self._api_key:
            request["api_key"] = self._api_key

        try:
            # LiteLLM response typing is not exported; keep runtime checks below.
            response: Any = await acompletion(  # type: ignore[reportUnknownVariableType]
                **request
            )
        except Exception as exc:  # pragma: no cover - provider errors
            raise LLMInferenceError(str(exc)) from exc

        content = ""
        try:
            assert response is not None
            if not hasattr(response, "choices"):
                raise ConfigGenerateError("LiteLLM response missing choices")
            raw_content = response.choices[0].message.content
            if not isinstance(raw_content, str) or not raw_content.strip():
                raise ConfigGenerateError("LiteLLM returned empty content")
            content = raw_content.strip()
            return MusicConfig.model_validate_json(content)
        except ValidationError as exc:
            extracted = _extract_json_payload(content)
            if extracted:
                try:
                    return MusicConfig.model_validate_json(extracted)
                except ValidationError:
                    pass
            snippet = _content_snippet(content) or "<empty>"
            raise ConfigGenerateError(f"LiteLLM returned non-JSON content: {snippet}") from exc
