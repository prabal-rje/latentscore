from __future__ import annotations

import asyncio
import json
import logging
import warnings
from collections.abc import Mapping
from threading import Event, Lock, Thread
from typing import Any, Callable, Coroutine, Literal

from pydantic import BaseModel, ValidationError

from ..config import MusicConfig, MusicConfigPromptPayload
from ..errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError
from ..models import EXTERNAL_PREFIX, build_litellm_prompt

_DEFAULT_TEMPERATURE = 0.0
_atexit_guard_registered = False
_safe_get_event_loop_installed = False
_LOGGER = logging.getLogger("latentscore.providers.litellm")
_RESERVED_LITELLM_KWARGS = frozenset(
    {"model", "messages", "response_format", "api_key", "reasoning_effort"}
)
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]
_litellm_loop: asyncio.AbstractEventLoop | None = None
_litellm_thread: Thread | None = None
_litellm_ready = Event()
_litellm_lock = Lock()
_litellm_shutdown_registered = False
_litellm_logging_configured = False

try:
    from json_repair import repair_json  # type: ignore[import]  # Optional dependency.
except ImportError:  # pragma: no cover - optional dependency
    repair_json: Callable[[str], str] | None = None


def _safe_async_cleanup(
    cleanup_coro: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _LOGGER.debug("LiteLLM cleanup running outside an event loop.")
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
            _LOGGER.debug("LiteLLM cleanup creating a new event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        if loop.is_closed():
            _LOGGER.debug("LiteLLM cleanup replacing a closed event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    asyncio.get_event_loop = safe_get_event_loop  # type: ignore[assignment]

    try:
        from litellm.llms.custom_httpx.async_client_cleanup import (
            close_litellm_async_clients,
        )
    except Exception as exc:
        _LOGGER.debug("LiteLLM cleanup import unavailable: %s", exc)
        return
    _safe_async_cleanup(close_litellm_async_clients)


def _register_safe_get_event_loop_atexit() -> None:
    global _atexit_guard_registered
    if _atexit_guard_registered:
        return
    _atexit_guard_registered = True
    import atexit

    atexit.register(_install_safe_get_event_loop)


def _shutdown_litellm_loop() -> None:
    loop = _litellm_loop
    thread = _litellm_thread
    if loop is None or thread is None:
        return
    if loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception as exc:
        _LOGGER.info("LiteLLM loop stop failed: %s", exc, exc_info=True)
    thread.join(timeout=1.0)


def _register_litellm_loop_atexit() -> None:
    global _litellm_shutdown_registered
    if _litellm_shutdown_registered:
        return
    _litellm_shutdown_registered = True
    import atexit

    atexit.register(_shutdown_litellm_loop)


def _configure_litellm_logging(litellm_module: Any) -> None:
    global _litellm_logging_configured
    if _litellm_logging_configured:
        return
    _litellm_logging_configured = True
    try:
        litellm_module.turn_off_message_logging = True
        litellm_module.disable_streaming_logging = True
        litellm_module.logging = False
    except Exception as exc:
        _LOGGER.info("LiteLLM logging config failed: %s", exc, exc_info=True)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def _ensure_litellm_loop() -> asyncio.AbstractEventLoop:
    global _litellm_loop, _litellm_thread
    with _litellm_lock:
        if _litellm_loop is not None and not _litellm_loop.is_closed():
            return _litellm_loop
        _litellm_ready.clear()

        def _runner() -> None:
            global _litellm_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _litellm_loop = loop
            _litellm_ready.set()
            loop.run_forever()
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception as exc:
                _LOGGER.info("LiteLLM loop shutdown failed: %s", exc, exc_info=True)
            loop.close()

        _litellm_thread = Thread(
            target=_runner,
            name="latentscore-litellm-loop",
            daemon=True,
        )
        _litellm_thread.start()
    _litellm_ready.wait()
    if _litellm_loop is None:
        raise RuntimeError("LiteLLM loop failed to start")
    _register_litellm_loop_atexit()
    return _litellm_loop


async def _run_on_litellm_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = _ensure_litellm_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        return await coro
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


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
    response_format: type[BaseModel] | None = None
    api_key: str | None = None


class LiteLLMAdapter:
    """LiteLLM model wrapper implementing the async model protocol."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        litellm_kwargs: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = MusicConfigPromptPayload,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        if model.startswith(EXTERNAL_PREFIX):
            model = model.removeprefix(EXTERNAL_PREFIX)
        self._model = model
        self._api_key = api_key
        self._litellm_kwargs = dict(litellm_kwargs or {})
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._base_prompt = build_litellm_prompt()
        self._response_format: type[BaseModel] | None = response_format
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort
        # self._schema_str = ""
        # if self._response_format:
        #     self._schema_str = f"<schema>{json.dumps(self._response_format.model_json_schema(), separators=(',', ':'))}</schema>"
        if self._api_key is None:
            _LOGGER.debug("No API key provided; letting LiteLLM read from env vars.")
        invalid_keys = _RESERVED_LITELLM_KWARGS.intersection(self._litellm_kwargs)
        if invalid_keys:
            keys = ", ".join(sorted(invalid_keys))
            raise ConfigGenerateError(f"litellm_kwargs cannot override: {keys}")

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

        async def _close() -> None:
            _register_safe_get_event_loop_atexit()
            close_fn: Any = getattr(litellm, "aclose", None)
            if close_fn is None:
                close_fn = getattr(litellm, "close_litellm_async_clients", None)
            if close_fn is None:
                return
            await close_fn()

        try:
            await _run_on_litellm_loop(_close())
        except Exception as exc:
            _LOGGER.warning("LiteLLM close failed: %s", exc, exc_info=True)

    async def generate(self, vibe: str) -> MusicConfig:
        try:
            import litellm  # type: ignore[import]
            from litellm import acompletion  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("LiteLLM not installed: %s", exc)
            raise ModelNotAvailableError("litellm is not installed") from exc

        _register_safe_get_event_loop_atexit()
        _configure_litellm_logging(litellm)
        messages: list[dict[str, str]] = [{"role": "system", "content": self._base_prompt}]
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        messages.append(
            {
                "role": "user",
                # "content": f"{self._schema_str}\n<vibe>{vibe}</vibe>\n<output>",
                "content": f"<vibe>{vibe}</vibe>\n<output>",
            }
        )

        print(json.dumps(messages, indent=2))

        request = _LiteLLMRequest(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format=self._response_format,
            api_key=self._api_key or None,
        ).model_dump(exclude_none=True)
        request.update(self._litellm_kwargs)

        try:
            response: Any = await _run_on_litellm_loop(
                acompletion(**request, reasoning_effort=self._reasoning_effort)
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
            print(self._parse_config_payload(content))
            return self._parse_config_payload(content)
        except ValidationError as exc:
            extracted = _extract_json_payload(content)
            if extracted:
                try:
                    return self._parse_config_payload(extracted)
                except ValidationError:
                    _LOGGER.warning(
                        "LiteLLM returned invalid JSON after extraction.", exc_info=True
                    )
            if repair_json is not None:
                repaired = repair_json(content)
                try:
                    return self._parse_config_payload(repaired)
                except ValidationError:
                    _LOGGER.warning("LiteLLM returned invalid JSON after repair.", exc_info=True)
            snippet = _content_snippet(content) or "<empty>"
            _LOGGER.warning("LiteLLM returned invalid JSON: %s", snippet)
            raise ConfigGenerateError(f"LiteLLM returned non-JSON content: {snippet}") from exc

    def _parse_config_payload(self, payload: str) -> MusicConfig:
        if self._response_format is MusicConfigPromptPayload:
            try:
                wrapper = MusicConfigPromptPayload.model_validate_json(payload)
            except ValidationError:
                return MusicConfig.model_validate_json(payload)
            return MusicConfig.model_validate(wrapper.config.model_dump())
        return MusicConfig.model_validate_json(payload)
