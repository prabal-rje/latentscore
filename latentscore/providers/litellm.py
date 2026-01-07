from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine
from pydantic import BaseModel, ValidationError
import json

from ..config import MusicConfig
from ..errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError
from ..models import build_expressive_prompt

_DEFAULT_TEMPERATURE = 0.0
# _JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object", "enforce_validations": "true", "strict": "true"}
_atexit_guard_registered = False
_safe_get_event_loop_installed = False
_LOGGER = logging.getLogger("latentscore.providers.litellm")

def _safe_async_cleanup(
    cleanup_coro: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _LOGGER.info("LiteLLM cleanup running outside an event loop.")
        loop = None

    if loop is not None:
        try:
            loop.create_task(cleanup_coro())
        except Exception as exc:
            _LOGGER.warning("LiteLLM async cleanup task failed: %s", exc, exc_info=True)
        return

    try:
        asyncio.run(cleanup_coro())
    except Exception as exc:
        _LOGGER.warning("LiteLLM async cleanup failed: %s", exc, exc_info=True)


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
            _LOGGER.info("LiteLLM cleanup creating a new event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        if loop.is_closed():
            _LOGGER.info("LiteLLM cleanup replacing a closed event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    asyncio.get_event_loop = safe_get_event_loop  # type: ignore[assignment]

    try:
        from litellm.llms.custom_httpx.async_client_cleanup import (
            close_litellm_async_clients,
        ) 
    except Exception as exc:
        _LOGGER.info("LiteLLM cleanup import unavailable: %s", exc)
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


class _LiteLLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    response_format: BaseModel | type[BaseModel] | None = None
    api_key: str | None = None
    


class LiteLLMAdapter:
    """LiteLLM model wrapper implementing the async model protocol."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        response_format: BaseModel | type[BaseModel] | None = MusicConfig,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._base_prompt = build_expressive_prompt()
        self._response_format = response_format
        self._reasoning_effort = reasoning_effort

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOGGER.info("LiteLLM close running outside event loop.")
            asyncio.run(self.aclose())
            return
        loop.create_task(self.aclose())

    async def aclose(self) -> None:
        try:
            import litellm  # type: ignore[import]  # Optional dependency at runtime.
        except ImportError as exc:
            _LOGGER.info("LiteLLM not installed; skipping async close: %s", exc)
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
            _LOGGER.warning("LiteLLM not installed: %s", exc)
            raise ModelNotAvailableError("litellm is not installed") from exc

        _register_safe_get_event_loop_atexit()
        messages: list[dict[str, str]] = [{"role": "system", "content": self._base_prompt}]
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        schema_str: str = ""
        if self._response_format:
            schema_str = f"<schema>{json.dumps(self._response_format.model_json_schema(), indent=2)}</schema>"
        messages.append(
            {
                "role": "user",
                "content": f"{schema_str}\n<vibe>{vibe}</vibe>\n<output>",
            }
        )

        request = _LiteLLMRequest(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format=self._response_format,
            api_key=self._api_key or None,
        ).model_dump(exclude_none=True)

        try:
            # LiteLLM response typing is not exported; keep runtime checks below.
            response: Any = await acompletion( 
                **request,
                reasoning_effort=self._reasoning_effort
            )
        except Exception as exc:  # pragma: no cover - provider errors
            _LOGGER.warning("LiteLLM request failed: %s", exc, exc_info=True)
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
                    _LOGGER.warning(
                        "LiteLLM returned invalid JSON after extraction.", exc_info=True
                    )
            snippet = _content_snippet(content) or "<empty>"
            _LOGGER.warning("LiteLLM returned invalid JSON: %s", snippet)
            raise ConfigGenerateError(f"LiteLLM returned non-JSON content: {snippet}") from exc
