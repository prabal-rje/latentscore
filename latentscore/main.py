# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread, Timer
from typing import Any, Callable, Coroutine, Literal, TypeGuard, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .audio import SAMPLE_RATE, AudioNumbers, FloatArray, ensure_audio_contract, write_wav
from .config import (
    ConfigInput,
    MusicConfig,
    MusicConfigUpdate,
    UpdateInput,
    _MusicConfigInternal,
    _MusicConfigUpdateInternal,
    coerce_internal_config,
    coerce_internal_update,
    is_empty_update,
    merge_internal_config,
)
from .errors import InvalidConfigError
from .models import (
    EXTERNAL_PREFIX,
    ExternalModelSpec,
    ModelForGeneratingMusicConfig,
    ModelSpec,
    resolve_model,
)
from .spinner import Spinner, render_error
from .synth import MusicConfig as SynthConfig
from .synth import assemble, interpolate_configs

StreamContent = str | ConfigInput | UpdateInput
PreviewPolicy = Literal["none", "embedding"]
FallbackPolicy = Literal["none", "keep_last", "embedding"]
FallbackInput = FallbackPolicy | ConfigInput | UpdateInput
ModelLoadRole = Literal["primary", "fallback"]


class StreamEvent(BaseModel):
    kind: Literal[
        "model_load_start",
        "model_load_end",
        "stream_start",
        "item_resolve_start",
        "item_resolve_success",
        "item_resolve_error",
        "item_preview_start",
        "first_config_ready",
        "first_audio_chunk",
        "stream_end",
        "fallback_used",
        "error",
    ]
    model: object | None = None
    model_role: ModelLoadRole | None = None
    index: int | None = None
    item: Streamable | None = None
    error: Exception | None = None
    fallback: FallbackInput | None = None
    preview_policy: PreviewPolicy | None = None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: object | None) -> object | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, ExternalModelSpec):
            return value
        if hasattr(value, "generate"):
            return value
        raise InvalidConfigError("model must be a ModelSpec-compatible value")


class StreamHooks(BaseModel):
    on_event: Callable[[StreamEvent], None] | None = None
    on_model_load_start: Callable[[ModelSpec, ModelLoadRole], None] | None = None
    on_model_load_end: Callable[[ModelSpec, ModelLoadRole], None] | None = None
    on_stream_start: Callable[[], None] | None = None
    on_item_resolve_start: Callable[[int, Streamable], None] | None = None
    on_item_resolve_success: Callable[[int, Streamable], None] | None = None
    on_item_resolve_error: (
        Callable[[int, Streamable, Exception, FallbackInput | None], None] | None
    ) = None
    on_item_preview_start: Callable[[int, Streamable, PreviewPolicy], None] | None = None
    on_first_config_ready: Callable[[int, Streamable], None] | None = None
    on_first_audio_chunk: Callable[[], None] | None = None
    on_stream_end: Callable[[], None] | None = None
    on_error: Callable[[Exception], None] | None = None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class RenderHooks(BaseModel):
    on_start: Callable[[], None] | None = None
    on_model_start: Callable[[ModelSpec], None] | None = None
    on_model_end: Callable[[ModelSpec], None] | None = None
    on_synth_start: Callable[[], None] | None = None
    on_synth_end: Callable[[], None] | None = None
    on_end: Callable[[], None] | None = None
    on_error: Callable[[Exception], None] | None = None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class FirstAudioSpinner:
    """Spinner helper for first-audio latency with optional preview messaging."""

    def __init__(
        self,
        *,
        delay: float = 0.35,
        enabled: bool | None = None,
    ) -> None:
        self._delay = delay
        self._spinner = Spinner("Generating first audio", enabled=enabled)
        self._timer: Timer | None = None
        self._started = False
        self._stopped = False

    def hooks(self) -> StreamHooks:
        return StreamHooks(
            on_model_load_start=self._on_model_load_start,
            on_model_load_end=self._on_model_load_end,
            on_stream_start=self._on_stream_start,
            on_item_preview_start=self._on_preview_start,
            on_first_audio_chunk=self._on_first_audio_chunk,
            on_stream_end=self._on_stream_end,
        )

    def _on_model_load_start(self, model: ModelSpec, role: ModelLoadRole) -> None:
        _ = model
        if role != "primary":
            return
        if self._stopped:
            return
        self._spinner.update("Loading model")
        self._start_immediately()

    def _on_model_load_end(self, model: ModelSpec, role: ModelLoadRole) -> None:
        _ = model
        if role != "primary":
            return
        if self._stopped:
            return
        self._spinner.update("Generating first audio")

    def _on_stream_start(self) -> None:
        if self._stopped or self._started or self._timer is not None:
            return
        self._timer = Timer(self._delay, self._start)
        self._timer.daemon = True
        self._timer.start()

    def _on_preview_start(
        self,
        index: int,
        item: Streamable,
        preview_policy: PreviewPolicy,
    ) -> None:
        _ = index
        _ = item
        if preview_policy == "embedding":
            self._spinner.update(
                "Generating first audio (speculative preview running; set preview_policy='none' to disable)"
            )

    def _on_first_audio_chunk(self) -> None:
        self._stop()

    def _on_stream_end(self) -> None:
        self._stop()

    def _start(self) -> None:
        if self._stopped or self._started:
            return
        self._started = True
        self._spinner.start()

    def _start_immediately(self) -> None:
        if self._stopped or self._started:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._start()

    def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._spinner.stop()


_MIN_CHUNK_SECONDS = 1e-6
_LOGGER = logging.getLogger("latentscore.main")
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class Streamable(BaseModel):
    """Structured streaming input with timing controls."""

    content: StreamContent
    duration: float = Field(default=60.0, gt=0.0)
    transition_duration: float = Field(default=1.0, ge=0.0)
    fallback: FallbackInput | None = None
    preview_policy: PreviewPolicy | None = None

    model_config = ConfigDict(extra="forbid")


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _LOGGER.info("No running event loop; running coroutine directly.")
        return asyncio.run(coro)

    return _EXECUTOR.submit(lambda: asyncio.run(coro)).result()


async def _maybe_close_model(model: ModelForGeneratingMusicConfig | None) -> None:
    if model is None:
        return
    aclose = getattr(model, "aclose", None)
    if callable(aclose):
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
            return
        except Exception as exc:
            _LOGGER.warning("Model async close failed: %s", exc, exc_info=True)
            return
    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            _LOGGER.warning("Model close failed: %s", exc, exc_info=True)


def _should_auto_close_model(model: ModelSpec) -> bool:
    if isinstance(model, ExternalModelSpec):
        return True
    if isinstance(model, str):
        return model.startswith(EXTERNAL_PREFIX)
    return False


def _log_exception(context: str, exc: Exception, *, show_console: bool = True) -> None:
    debug = bool(os.environ.get("LATENTSCORE_DEBUG"))
    _LOGGER.warning("%s failed: %s", context, exc, exc_info=debug)
    if show_console:
        render_error(context, exc)


def _is_model_spec(value: object) -> TypeGuard[ModelSpec]:
    if isinstance(value, str):
        return True
    if isinstance(value, ExternalModelSpec):
        return True
    return hasattr(value, "generate")


def _emit_event(hooks: StreamHooks | None, event: StreamEvent) -> None:
    if hooks is None:
        return
    try:
        if hooks.on_event is not None:
            hooks.on_event(event)
        match event.kind:
            case "model_load_start":
                if (
                    hooks.on_model_load_start is not None
                    and event.model is not None
                    and event.model_role is not None
                ):
                    if _is_model_spec(event.model):
                        hooks.on_model_load_start(event.model, event.model_role)
            case "model_load_end":
                if (
                    hooks.on_model_load_end is not None
                    and event.model is not None
                    and event.model_role is not None
                ):
                    if _is_model_spec(event.model):
                        hooks.on_model_load_end(event.model, event.model_role)
            case "stream_start":
                if hooks.on_stream_start is not None:
                    hooks.on_stream_start()
            case "item_resolve_start":
                if (
                    hooks.on_item_resolve_start is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_item_resolve_start(event.index, event.item)
            case "item_resolve_success":
                if (
                    hooks.on_item_resolve_success is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_item_resolve_success(event.index, event.item)
            case "item_resolve_error":
                if (
                    hooks.on_item_resolve_error is not None
                    and event.index is not None
                    and event.item is not None
                    and event.error is not None
                ):
                    hooks.on_item_resolve_error(
                        event.index,
                        event.item,
                        event.error,
                        event.fallback,
                    )
            case "item_preview_start":
                if (
                    hooks.on_item_preview_start is not None
                    and event.index is not None
                    and event.item is not None
                    and event.preview_policy is not None
                ):
                    hooks.on_item_preview_start(
                        event.index,
                        event.item,
                        event.preview_policy,
                    )
            case "first_config_ready":
                if (
                    hooks.on_first_config_ready is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_first_config_ready(event.index, event.item)
            case "first_audio_chunk":
                if hooks.on_first_audio_chunk is not None:
                    hooks.on_first_audio_chunk()
            case "stream_end":
                if hooks.on_stream_end is not None:
                    hooks.on_stream_end()
            case "error":
                if hooks.on_error is not None and event.error is not None:
                    hooks.on_error(event.error)
            case "fallback_used":
                pass
            case _:
                pass
    except Exception as exc:
        debug = bool(os.environ.get("LATENTSCORE_DEBUG"))
        _LOGGER.warning("Stream hook failed: %s", exc, exc_info=debug)


def _emit_render(
    hooks: RenderHooks | None,
    *,
    kind: Literal[
        "start",
        "model_start",
        "model_end",
        "synth_start",
        "synth_end",
        "end",
        "error",
    ],
    model: ModelSpec | None = None,
    error: Exception | None = None,
) -> None:
    if hooks is None:
        return
    try:
        match kind:
            case "start":
                if hooks.on_start is not None:
                    hooks.on_start()
            case "model_start":
                if hooks.on_model_start is not None and model is not None:
                    hooks.on_model_start(model)
            case "model_end":
                if hooks.on_model_end is not None and model is not None:
                    hooks.on_model_end(model)
            case "synth_start":
                if hooks.on_synth_start is not None:
                    hooks.on_synth_start()
            case "synth_end":
                if hooks.on_synth_end is not None:
                    hooks.on_synth_end()
            case "end":
                if hooks.on_end is not None:
                    hooks.on_end()
            case "error":
                if hooks.on_error is not None and error is not None:
                    hooks.on_error(error)
            case _:
                raise InvalidConfigError(f"Unknown render hook kind: {kind}")
    except Exception as exc:
        debug = bool(os.environ.get("LATENTSCORE_DEBUG"))
        _LOGGER.warning("Render hook failed: %s", exc, exc_info=debug)


class _PrefetchedItem(BaseModel):
    index: int
    item: Streamable
    task: asyncio.Task[_MusicConfigInternal] | None

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class _ModelLoadState(BaseModel):
    model: Any
    role: ModelLoadRole
    hooks: StreamHooks | None
    loaded: bool = False

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: object) -> object:
        if not hasattr(value, "generate"):
            raise InvalidConfigError("model must implement generate()")
        return value

    async def ensure_loaded(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        if self.hooks is None:
            return
        _emit_event(
            self.hooks,
            StreamEvent(
                kind="model_load_start",
                model=self.model,
                model_role=self.role,
            ),
        )
        try:
            warmup = getattr(self.model, "warmup", None)
            if callable(warmup):
                await asyncio.to_thread(warmup)
        finally:
            _emit_event(
                self.hooks,
                StreamEvent(
                    kind="model_load_end",
                    model=self.model,
                    model_role=self.role,
                ),
            )


def _to_synth_config(config: _MusicConfigInternal) -> SynthConfig:
    return SynthConfig(
        tempo=config.tempo,
        root=config.root,
        mode=config.mode,
        brightness=config.brightness,
        space=config.space,
        density=config.density,
        bass=config.bass,
        pad=config.pad,
        melody=config.melody,
        rhythm=config.rhythm,
        texture=config.texture,
        accent=config.accent,
        motion=config.motion,
        attack=config.attack,
        stereo=config.stereo,
        depth=config.depth,
        echo=config.echo,
        human=config.human,
        grain=config.grain,
    )


def _from_synth_config(config: SynthConfig) -> _MusicConfigInternal:
    return _MusicConfigInternal(**config.model_dump())


def _apply_update(
    base: _MusicConfigInternal,
    update: UpdateInput | None,
) -> _MusicConfigInternal:
    match update:
        case None:
            return base
        case _:
            internal_update = coerce_internal_update(update)
            return merge_internal_config(base, internal_update)


def _chunk_count(duration: float, chunk_seconds: float) -> int:
    return max(1, int(round(duration / max(chunk_seconds, _MIN_CHUNK_SECONDS))))


def _transition_steps(transition_duration: float, chunk_seconds: float) -> int:
    match transition_duration:
        case (float() | int()) as duration if duration <= 0:
            return 0
        case (float() | int()) as duration:
            return max(1, int(round(duration / max(chunk_seconds, _MIN_CHUNK_SECONDS))))
        case _:
            raise InvalidConfigError("transition_duration must be a number")


def _iter_transition_configs(
    current: _MusicConfigInternal | None,
    target: _MusicConfigInternal,
    steps: int,
) -> Iterator[_MusicConfigInternal]:
    if current is None or steps <= 1:
        yield target
        return
    start = _to_synth_config(current)
    end = _to_synth_config(target)
    for step in range(1, steps + 1):
        t = step / steps
        interpolated = interpolate_configs(start, end, t)
        yield _from_synth_config(interpolated)


def _wrap_streamables(
    items: Iterable[StreamContent],
    *,
    duration: float,
    transition_duration: float,
) -> Iterator[Streamable]:
    for item in items:
        yield Streamable(
            content=item,
            duration=duration,
            transition_duration=transition_duration,
        )


def render(
    vibe: str,
    *,
    duration: float = 8.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    hooks: RenderHooks | None = None,
) -> FloatArray:
    _emit_render(hooks, kind="start")
    try:
        if config is None:
            resolved = resolve_model(model)
            should_close = _should_auto_close_model(model)
            _emit_render(hooks, kind="model_start", model=model)

            async def _generate_with_cleanup() -> MusicConfig:
                try:
                    return await resolved.generate(vibe)
                finally:
                    if should_close:
                        await _maybe_close_model(resolved)

            try:
                base = _run_async(_generate_with_cleanup()).to_internal()
            finally:
                _emit_render(hooks, kind="model_end", model=model)
        else:
            base = coerce_internal_config(config)
        target = _apply_update(base, update)

        _emit_render(hooks, kind="synth_start")
        audio = assemble(_to_synth_config(target), duration)
        normalized = ensure_audio_contract(audio, sample_rate=SAMPLE_RATE)
        _emit_render(hooks, kind="synth_end")
        _emit_render(hooks, kind="end")
        return normalized
    except Exception as exc:
        _emit_render(hooks, kind="error", error=exc)
        show_console = hooks is None or hooks.on_error is None
        _log_exception("render", exc, show_console=show_console)
        raise


def _bridge_async_stream(
    async_iter: AsyncIterable[FloatArray],
    *,
    queue_maxsize: int = 0,
) -> Iterator[FloatArray]:
    sentinel = object()
    queue: Queue[object] = Queue(maxsize=queue_maxsize) if queue_maxsize > 0 else Queue()

    def _runner() -> None:
        async def _run() -> None:
            try:
                async for chunk in async_iter:
                    queue.put(chunk)
            except BaseException as exc:
                queue.put(exc)
            finally:
                queue.put(sentinel)

        asyncio.run(_run())

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        if not isinstance(item, np.ndarray):
            raise InvalidConfigError("Async stream emitted non-audio chunk")
        yield item
    thread.join(timeout=0)


def stream(
    items: Iterable[Streamable],
    *,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
    queue_maxsize: int = 0,
) -> Iterable[FloatArray]:
    try:
        items_obj: object = items
        match items_obj:
            case Streamable():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream expects an iterable of Streamable items")
            case str():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream expects an iterable of Streamable items")
            case _:
                pass

        if prefetch_depth < 0:
            raise InvalidConfigError("prefetch_depth must be >= 0")

        async_iter = astream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
        for chunk in _bridge_async_stream(async_iter, queue_maxsize=queue_maxsize):
            yield chunk
    except Exception as exc:
        show_console = hooks is None or hooks.on_error is None
        _log_exception("stream", exc, show_console=show_console)
        raise


def stream_texts(
    prompts: Iterable[str],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        match prompts:
            case str():
                raise InvalidConfigError("stream_texts expects an iterable of strings")
            case _:
                pass

        items = _wrap_streamables(
            prompts,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        show_console = hooks is None or hooks.on_error is None
        _log_exception("stream_texts", exc, show_console=show_console)
        raise


def stream_configs(
    configs: Iterable[ConfigInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        configs_obj: object = configs
        match configs_obj:
            case MusicConfig() | _MusicConfigInternal():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream_configs expects an iterable of config objects")
            case _:
                pass

        items = _wrap_streamables(
            configs,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        show_console = hooks is None or hooks.on_error is None
        _log_exception("stream_configs", exc, show_console=show_console)
        raise


def stream_updates(
    updates: Iterable[UpdateInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        updates_obj: object = updates
        match updates_obj:
            case MusicConfigUpdate() | _MusicConfigUpdateInternal():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream_updates expects an iterable of update objects")
            case _:
                pass

        items = _wrap_streamables(
            updates,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        show_console = hooks is None or hooks.on_error is None
        _log_exception("stream_updates", exc, show_console=show_console)
        raise


def save_wav(path: str, audio_or_chunks: AudioNumbers | Iterable[AudioNumbers]) -> str:
    try:
        written = write_wav(path, audio_or_chunks, sample_rate=SAMPLE_RATE)
        return str(written)
    except Exception as exc:
        _log_exception("save_wav", exc, show_console=True)
        raise


def _coerce_async_items(
    items: Iterable[Streamable] | AsyncIterable[Streamable],
) -> AsyncIterable[Streamable]:
    items_obj: object = items
    match items_obj:
        case Streamable():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("astream expects an iterable of Streamable items")
        case str():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("astream expects an iterable of Streamable items")
        case AsyncIterable() as async_items:
            return async_items
        case Iterable() as sync_items:
            pass
        case _:
            raise InvalidConfigError("astream expects an iterable of Streamable items")

    async def _iter() -> AsyncIterator[Streamable]:
        for item in sync_items:
            yield item

    return _iter()


async def _generate_internal(
    vibe: str,
    model: ModelForGeneratingMusicConfig,
) -> _MusicConfigInternal:
    return (await model.generate(vibe)).to_internal()


async def _resolve_preview(
    item: Streamable,
    *,
    preview_policy: PreviewPolicy,
    fallback_model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
    fallback_load: _ModelLoadState | None = None,
) -> _MusicConfigInternal | None:
    if preview_policy != "embedding" or not isinstance(item.content, str):
        return None
    if fallback_load is not None:
        await fallback_load.ensure_loaded()
    preview = await fallback_model.generate(item.content)
    return _apply_update(preview.to_internal(), update)


async def _resolve_fallback(
    item: Streamable,
    *,
    current: _MusicConfigInternal | None,
    fallback: FallbackInput | None,
    fallback_model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
    fallback_load: _ModelLoadState | None = None,
) -> _MusicConfigInternal | None:
    if fallback is None or fallback == "keep_last":
        base = current or _MusicConfigInternal()
        return _apply_update(base, update)
    if fallback == "none":
        return None
    if fallback == "embedding":
        if not isinstance(item.content, str):
            base = current or _MusicConfigInternal()
            return _apply_update(base, update)
        if fallback_load is not None:
            await fallback_load.ensure_loaded()
        target = await fallback_model.generate(item.content)
        return _apply_update(target.to_internal(), update)
    if fallback_load is not None and isinstance(fallback, str):
        await fallback_load.ensure_loaded()
    return await _resolve_target_async(
        fallback,
        current=current,
        model=fallback_model,
        update=update,
    )


async def _render_chunk(
    config_item: _MusicConfigInternal,
    *,
    chunk_seconds: float,
) -> FloatArray:
    return await asyncio.to_thread(
        lambda: ensure_audio_contract(
            assemble(_to_synth_config(config_item), chunk_seconds),
            sample_rate=SAMPLE_RATE,
            check_peak=False,
        )
    )


async def _prefetch_item(
    item: Streamable,
    *,
    model: ModelForGeneratingMusicConfig,
    hooks: StreamHooks | None,
    index: int,
    load_state: _ModelLoadState | None = None,
) -> _PrefetchedItem:
    task: asyncio.Task[_MusicConfigInternal] | None = None
    if isinstance(item.content, str):
        if load_state is not None:
            await load_state.ensure_loaded()
        _emit_event(hooks, StreamEvent(kind="item_resolve_start", index=index, item=item))
        task = asyncio.create_task(_generate_internal(item.content, model))
    return _PrefetchedItem(index=index, item=item, task=task)


async def _resolve_target_async(
    item: StreamContent,
    *,
    current: _MusicConfigInternal | None,
    model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
) -> _MusicConfigInternal:
    match item:
        case _MusicConfigInternal():
            target = item
        case MusicConfig():
            target = item.to_internal()
        case _MusicConfigUpdateInternal():
            base = current or _MusicConfigInternal()
            target = (
                base
                if current is None and is_empty_update(item)
                else merge_internal_config(base, item)
            )
        case MusicConfigUpdate():
            base = current or _MusicConfigInternal()
            internal_update = item.to_internal()
            target = (
                base
                if current is None and is_empty_update(item)
                else merge_internal_config(base, internal_update)
            )
        case str():
            target = (await model.generate(item)).to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported input type: {type(item).__name__}")

    return _apply_update(target, update)


async def astream(
    items: Iterable[Streamable] | AsyncIterable[Streamable],
    *,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> AsyncIterable[FloatArray]:
    resolved: ModelForGeneratingMusicConfig | None = None
    fallback_resolved: ModelForGeneratingMusicConfig | None = None
    should_close = False
    fallback_should_close = False
    try:
        match chunk_seconds:
            case (float() | int()) as value if value <= 0:
                raise InvalidConfigError("chunk_seconds must be greater than 0")
            case float() | int():
                pass
            case _:
                raise InvalidConfigError("chunk_seconds must be a number")

        if prefetch_depth < 0:
            raise InvalidConfigError("prefetch_depth must be >= 0")

        current = coerce_internal_config(config) if config is not None else None
        resolved = resolve_model(model)
        fallback_resolved = resolve_model(fallback_model)
        should_close = _should_auto_close_model(model)
        fallback_should_close = _should_auto_close_model(fallback_model)
        primary_load = (
            _ModelLoadState(model=resolved, role="primary", hooks=hooks)
            if hooks is not None
            else None
        )
        fallback_load = (
            _ModelLoadState(model=fallback_resolved, role="fallback", hooks=hooks)
            if hooks is not None
            else None
        )

        _emit_event(hooks, StreamEvent(kind="stream_start"))

        async_items = _coerce_async_items(items)
        iterator = async_items.__aiter__()
        pending: deque[_PrefetchedItem] = deque()
        next_index = 0
        prefetch_target = max(1, prefetch_depth + 1)
        first_config_ready = False
        first_audio_chunk = False

        async def _next_item() -> Streamable | None:
            try:
                return await iterator.__anext__()
            except StopAsyncIteration:
                return None

        async def _fill_pending() -> None:
            nonlocal next_index
            while len(pending) < prefetch_target:
                item = await _next_item()
                if item is None:
                    break
                pending.append(
                    await _prefetch_item(
                        item,
                        model=resolved,
                        hooks=hooks,
                        index=next_index,
                        load_state=primary_load,
                    )
                )
                next_index += 1

        await _fill_pending()
        while pending:
            prefetched = pending.popleft()
            await _fill_pending()

            item = prefetched.item
            index = prefetched.index
            item_preview = item.preview_policy or preview_policy
            item_fallback = item.fallback if item.fallback is not None else fallback
            pending_task = prefetched.task

            pending_real_task: asyncio.Task[_MusicConfigInternal] | None = None
            target: _MusicConfigInternal | None = None

            if pending_task is None:
                target = await _resolve_target_async(
                    item.content,
                    current=current,
                    model=resolved,
                    update=update,
                )
                _emit_event(hooks, StreamEvent(kind="item_resolve_success", index=index, item=item))
            else:
                if pending_task.done():
                    try:
                        target = _apply_update(pending_task.result(), update)
                        _emit_event(
                            hooks,
                            StreamEvent(kind="item_resolve_success", index=index, item=item),
                        )
                    except Exception as exc:
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_error",
                                index=index,
                                item=item,
                                error=exc,
                                fallback=item_fallback,
                            ),
                        )
                        target = await _resolve_fallback(
                            item,
                            current=current,
                            fallback=item_fallback,
                            fallback_model=fallback_resolved,
                            update=update,
                            fallback_load=fallback_load,
                        )
                        if target is None:
                            raise
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="fallback_used",
                                index=index,
                                item=item,
                                fallback=item_fallback,
                            ),
                        )
                else:
                    preview = await _resolve_preview(
                        item,
                        preview_policy=item_preview,
                        fallback_model=fallback_resolved,
                        update=update,
                        fallback_load=fallback_load,
                    )
                    if preview is not None:
                        target = preview
                        pending_real_task = pending_task
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_preview_start",
                                index=index,
                                item=item,
                                preview_policy=item_preview,
                            ),
                        )
                    else:
                        try:
                            target = _apply_update(await pending_task, update)
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="item_resolve_success",
                                    index=index,
                                    item=item,
                                ),
                            )
                        except Exception as exc:
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="item_resolve_error",
                                    index=index,
                                    item=item,
                                    error=exc,
                                    fallback=item_fallback,
                                ),
                            )
                            target = await _resolve_fallback(
                                item,
                                current=current,
                                fallback=item_fallback,
                                fallback_model=fallback_resolved,
                                update=update,
                                fallback_load=fallback_load,
                            )
                            if target is None:
                                raise
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="fallback_used",
                                    index=index,
                                    item=item,
                                    fallback=item_fallback,
                                ),
                            )

            if not first_config_ready:
                _emit_event(hooks, StreamEvent(kind="first_config_ready", index=index, item=item))
                first_config_ready = True

            total_chunks = _chunk_count(item.duration, chunk_seconds)
            transition_chunks = (
                0
                if current is None
                else min(
                    total_chunks,
                    _transition_steps(item.transition_duration, chunk_seconds),
                )
            )

            remaining = total_chunks
            if transition_chunks > 0:
                for config_item in _iter_transition_configs(current, target, transition_chunks):
                    current = config_item
                    if not first_audio_chunk:
                        _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                        first_audio_chunk = True
                    yield await _render_chunk(config_item, chunk_seconds=chunk_seconds)
                remaining -= transition_chunks
            else:
                current = target

            match current:
                case _MusicConfigInternal() as current_config:
                    pass
                case None:
                    raise InvalidConfigError("Stream exhausted without a current config")
                case _:
                    raise InvalidConfigError("Stream exhausted with an invalid config")

            while remaining > 0:
                if pending_real_task is not None and pending_real_task.done():
                    try:
                        real_target = _apply_update(pending_real_task.result(), update)
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_success",
                                index=index,
                                item=item,
                            ),
                        )
                    except Exception as exc:
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_error",
                                index=index,
                                item=item,
                                error=exc,
                                fallback=item_fallback,
                            ),
                        )
                        real_target = await _resolve_fallback(
                            item,
                            current=current,
                            fallback=item_fallback,
                            fallback_model=fallback_resolved,
                            update=update,
                            fallback_load=fallback_load,
                        )
                        if real_target is None:
                            raise
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="fallback_used",
                                index=index,
                                item=item,
                                fallback=item_fallback,
                            ),
                        )

                    pending_real_task = None
                    transition = min(
                        remaining,
                        _transition_steps(item.transition_duration, chunk_seconds),
                    )
                    if transition > 0:
                        for config_item in _iter_transition_configs(
                            current_config,
                            real_target,
                            transition,
                        ):
                            current_config = config_item
                            if not first_audio_chunk:
                                _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                                first_audio_chunk = True
                            yield await _render_chunk(config_item, chunk_seconds=chunk_seconds)
                        remaining -= transition
                        current = current_config
                        continue
                    current_config = real_target
                    current = real_target

                if not first_audio_chunk:
                    _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                    first_audio_chunk = True
                yield await _render_chunk(current_config, chunk_seconds=chunk_seconds)
                remaining -= 1

            if pending_real_task is not None and not pending_real_task.done():
                pending_real_task.cancel()
                pending_real_task = None

        _emit_event(hooks, StreamEvent(kind="stream_end"))
    except Exception as exc:
        _emit_event(hooks, StreamEvent(kind="error", error=exc))
        show_console = hooks is None or hooks.on_error is None
        _log_exception("astream", exc, show_console=show_console)
        raise
    finally:
        if should_close:
            await _maybe_close_model(resolved)
        if (
            fallback_should_close
            and fallback_resolved is not None
            and fallback_resolved is not resolved
        ):
            await _maybe_close_model(fallback_resolved)
