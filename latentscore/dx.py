from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Sequence
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from .audio import SAMPLE_RATE, FloatArray, ensure_audio_contract, write_wav
from .config import (
    ConfigInput,
    MusicConfig,
    MusicConfigUpdate,
    SynthConfig,
    UpdateInput,
    coerce_internal_config,
    coerce_internal_update,
    merge_internal_config,
)
from .errors import InvalidConfigError, ModelNotAvailableError, PlaybackError
from .indicators import RichIndicator
from .main import FallbackInput, RenderHooks, Streamable, StreamContent, StreamHooks
from .main import (
    astream as astream_raw,
)
from .main import (
    render as render_raw,
)
from .main import (
    stream as stream_raw,
)
from .models import EXTERNAL_PREFIX, ModelSpec
from .spinner import render_error
from .synth import assemble

TrackContent = str | MusicConfig | MusicConfigUpdate

_LOGGER = logging.getLogger("latentscore.dx")


class Audio(BaseModel):
    samples: FloatArray
    sample_rate: int = SAMPLE_RATE

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def _normalize(self) -> "Audio":
        normalized = ensure_audio_contract(self.samples, sample_rate=self.sample_rate)
        object.__setattr__(self, "samples", normalized)
        return self

    def to_numpy(self) -> FloatArray:
        return self.samples

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
        return np.asarray(self.samples, dtype=dtype)

    def save(self, path: str | Path) -> Path:
        return write_wav(path, self.samples, sample_rate=self.sample_rate)

    def play(self) -> None:
        from .playback import play_audio

        try:
            play_audio(self.samples, sample_rate=self.sample_rate)
        except PlaybackError as exc:
            render_error("playback", exc)
            raise


class AudioStream:
    def __init__(
        self,
        sync_factory: Callable[[], Iterable[FloatArray]],
        async_factory: Callable[[], AsyncIterable[FloatArray]],
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._sync_factory = sync_factory
        self._async_factory = async_factory
        self.sample_rate = sample_rate

    def __iter__(self) -> Iterator[FloatArray]:
        return iter(self._sync_factory())

    def __aiter__(self) -> AsyncIterator[FloatArray]:
        return self._async_factory().__aiter__()

    def collect(self) -> Audio:
        chunks = [
            ensure_audio_contract(chunk, sample_rate=self.sample_rate)
            for chunk in self._sync_factory()
        ]
        if not chunks:
            return Audio(samples=np.array([], dtype=np.float32), sample_rate=self.sample_rate)
        return Audio(samples=np.concatenate(chunks), sample_rate=self.sample_rate)

    def save(self, path: str | Path) -> Path:
        return write_wav(path, self._sync_factory(), sample_rate=self.sample_rate)

    def play(self) -> None:
        from .playback import play_stream

        try:
            play_stream(self._sync_factory(), sample_rate=self.sample_rate)
        except PlaybackError:
            _LOGGER.info(
                "Streaming playback unavailable; falling back to buffered play.", exc_info=True
            )
            self.collect().play()


class Track(BaseModel):
    content: TrackContent
    duration: float = 60.0
    transition: float = 5.0
    name: str | None = None

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_streamable(self) -> Streamable:
        return Streamable(
            content=self.content,
            duration=self.duration,
            transition_duration=self.transition,
        )


StreamItems = Track | Streamable | TrackContent
StreamItemsInput = StreamItems | Sequence[StreamItems]


class Playlist(BaseModel):
    tracks: tuple[Track, ...]

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def stream(
        self,
        *,
        chunk_seconds: float = 1.0,
        pattern_seconds: float | None = None,
        transition_seconds: float = 5.0,
        model: ModelSpec = "fast",
        config: ConfigInput | None = None,
        update: UpdateInput | None = None,
        fallback: FallbackInput | None = None,
        fallback_model: ModelSpec = "fast",
        hooks: StreamHooks | None = None,
        queue_maxsize: int = 0,
    ) -> AudioStream:
        items = [track.to_streamable() for track in self.tracks]
        return _stream_from_items(
            items,
            chunk_seconds=chunk_seconds,
            pattern_seconds=pattern_seconds,
            transition_seconds=transition_seconds,
            model=model,
            config=config,
            update=update,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
            queue_maxsize=queue_maxsize,
        )

    def render(self, **kwargs: Any) -> Audio:
        return self.stream(**kwargs).collect()

    def play(self, **kwargs: Any) -> None:
        self.stream(**kwargs).play()


LiveSource = (
    Streamable
    | TrackContent
    | Track
    | Iterable[Streamable | TrackContent | Track]
    | AsyncIterable[Streamable | TrackContent | Track]
)

_NormalizedSource = Iterable[Streamable | StreamContent] | AsyncIterable[Streamable | StreamContent]


class LiveStream:
    """Live streaming wrapper around stream_raw/astream_raw.

    This is a single-use stream: once consumed, it cannot be replayed.
    """

    def __init__(
        self,
        source: LiveSource,
        *,
        chunk_seconds: float = 1.0,
        pattern_seconds: float | None = None,
        transition_seconds: float = 1.0,
        model: ModelSpec = "fast",
        config: ConfigInput | None = None,
        update: UpdateInput | None = None,
        fallback: FallbackInput | None = None,
        fallback_model: ModelSpec = "fast",
        hooks: StreamHooks | None = None,
        queue_maxsize: int = 0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._source: _NormalizedSource = self._normalize_source(source)
        self._chunk_seconds = chunk_seconds
        self._pattern_seconds = pattern_seconds
        self._transition_seconds = transition_seconds
        self._model: ModelSpec = _coerce_model(model)
        self._fallback_model: ModelSpec = _coerce_model(fallback_model)
        self._config = config
        self._update = update
        self._fallback: FallbackInput | None = fallback
        self._queue_maxsize = queue_maxsize
        self.sample_rate = sample_rate
        self._started = False
        self._sync_stop: Event | None = None
        self._sync_thread: Thread | None = None
        self._indicator = None
        if hooks is None:
            self._indicator = RichIndicator()
            self._hooks = self._indicator.stream_hooks()
        else:
            self._hooks = hooks

    def __iter__(self) -> Iterator[FloatArray]:
        return iter(self.chunks())

    def __aiter__(self) -> AsyncIterator[FloatArray]:
        return self.achunks()

    def chunks(self, seconds: float | None = None) -> Iterable[FloatArray]:
        self._claim()
        max_chunks = _seconds_to_chunks(seconds, self._chunk_seconds)
        queue: Queue[object] = Queue(maxsize=self._queue_maxsize)
        sentinel = object()
        stop_event = Event()
        self._sync_stop = stop_event

        def _runner() -> None:
            async def _run() -> None:
                stream = astream_raw(
                    self._source,
                    chunk_seconds=self._chunk_seconds,
                    pattern_seconds=self._pattern_seconds,
                    transition_seconds=self._transition_seconds,
                    model=self._model,
                    config=self._config,
                    update=self._update,
                    fallback=self._fallback,
                    fallback_model=self._fallback_model,
                    hooks=self._hooks,
                )
                async_iter = stream.__aiter__()
                count = 0
                try:
                    async for chunk in async_iter:
                        if stop_event.is_set():
                            break
                        if max_chunks is not None and count >= max_chunks:
                            break
                        count += 1
                        while True:
                            try:
                                queue.put(chunk, timeout=0.1)
                                break
                            except Full:
                                if stop_event.is_set():
                                    break
                                continue
                except BaseException as exc:
                    queue.put(exc)
                finally:
                    aclose = getattr(async_iter, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:
                            pass
                    queue.put(sentinel)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run())
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        thread = Thread(target=_runner, daemon=True)
        self._sync_thread = thread
        thread.start()

        def _iter() -> Iterator[FloatArray]:
            try:
                while True:
                    item = queue.get()
                    if item is sentinel:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    assert isinstance(item, np.ndarray)
                    yield item
            finally:
                stop_event.set()
                thread.join(timeout=2.0)
                self._sync_stop = None
                self._sync_thread = None

        return _iter()

    async def achunks(self, seconds: float | None = None) -> AsyncIterator[FloatArray]:
        self._claim()
        max_chunks = _seconds_to_chunks(seconds, self._chunk_seconds)
        stream = astream_raw(
            self._source,
            chunk_seconds=self._chunk_seconds,
            pattern_seconds=self._pattern_seconds,
            transition_seconds=self._transition_seconds,
            model=self._model,
            config=self._config,
            update=self._update,
            fallback=self._fallback,
            fallback_model=self._fallback_model,
            hooks=self._hooks,
        )
        async_iter = stream.__aiter__()
        count = 0
        try:
            async for chunk in async_iter:
                yield chunk
                count += 1
                if max_chunks is not None and count >= max_chunks:
                    break
        finally:
            aclose = getattr(async_iter, "aclose", None)
            if aclose is not None:
                await aclose()

    def collect(self, seconds: float | None = None) -> Audio:
        chunks = [
            ensure_audio_contract(chunk, sample_rate=self.sample_rate)
            for chunk in self.chunks(seconds)
        ]
        if not chunks:
            return Audio(samples=np.array([], dtype=np.float32), sample_rate=self.sample_rate)
        return Audio(samples=np.concatenate(chunks), sample_rate=self.sample_rate)

    def save(self, path: str | Path, seconds: float | None = None) -> Path:
        return write_wav(path, self.chunks(seconds), sample_rate=self.sample_rate)

    def play(self, seconds: float | None = None) -> None:
        from .playback import play_stream

        try:
            play_stream(self.chunks(seconds), sample_rate=self.sample_rate)
        except PlaybackError:
            _LOGGER.info(
                "Streaming playback unavailable; falling back to buffered play.",
                exc_info=True,
            )
            self.collect(seconds).play()

    async def aplay(self, seconds: float | None = None) -> None:
        await asyncio.to_thread(self.play, seconds)

    def close(self) -> None:
        if self._sync_stop is not None:
            self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)
            self._sync_thread = None
            self._sync_stop = None

    def _claim(self) -> None:
        if self._started:
            raise InvalidConfigError("LiveStream can only be consumed once")
        self._started = True

    def _normalize_source(self, source: LiveSource) -> _NormalizedSource:
        if isinstance(source, Track):
            return [source.to_streamable()]
        if isinstance(source, (str, MusicConfig, MusicConfigUpdate, Streamable)):
            return [source]
        if isinstance(source, AsyncIterable):

            async def _aiter() -> AsyncIterator[Streamable | TrackContent]:
                async for item in source:
                    if isinstance(item, Track):
                        yield item.to_streamable()
                    else:
                        yield item

            return _aiter()

        def _iter() -> Iterator[Streamable | TrackContent]:
            for item in source:
                if isinstance(item, Track):
                    yield item.to_streamable()
                else:
                    yield item

        return _iter()


def render(
    vibe_or_config: TrackContent,
    *,
    duration: float = 8.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    sample_rate: int = SAMPLE_RATE,
    hooks: RenderHooks | None = None,
) -> Audio:
    if sample_rate != SAMPLE_RATE:
        raise InvalidConfigError(f"sample_rate must be {SAMPLE_RATE}")

    resolved_hooks = hooks
    if hooks is None:
        resolved_hooks = RichIndicator().render_hooks()

    if isinstance(vibe_or_config, str):
        resolved_model = _coerce_model(model)
        audio = render_raw(
            vibe_or_config,
            duration=duration,
            model=resolved_model,
            config=config,
            update=update,
            hooks=resolved_hooks,
        )
        return Audio(samples=audio, sample_rate=sample_rate)

    base = coerce_internal_config(config) if config is not None else MusicConfig().to_internal()
    match vibe_or_config:
        case MusicConfig():
            target = vibe_or_config.to_internal()
        case MusicConfigUpdate():
            internal_update = coerce_internal_update(vibe_or_config)
            target = merge_internal_config(base, internal_update)
        case _:
            raise InvalidConfigError(
                f"Unsupported render input type: {type(vibe_or_config).__name__}"
            )

    if update is not None:
        target = merge_internal_config(target, coerce_internal_update(update))

    _emit_render_hooks(resolved_hooks, kind="start")
    try:
        synth_config = SynthConfig.model_validate(target.model_dump())
        _emit_render_hooks(resolved_hooks, kind="synth_start")
        audio = assemble(synth_config, duration=duration)
        normalized = ensure_audio_contract(audio, sample_rate=sample_rate)
        _emit_render_hooks(resolved_hooks, kind="synth_end")
        _emit_render_hooks(resolved_hooks, kind="end")
        return Audio(samples=normalized, sample_rate=sample_rate)
    except Exception as exc:
        _emit_render_hooks(resolved_hooks, kind="error", error=exc)
        debug = bool(os.environ.get("LATENTSCORE_DEBUG"))
        _LOGGER.warning("render failed: %s", exc, exc_info=debug)
        if hooks is not None and hooks.on_error is None:
            render_error("render", exc)
        raise


async def arender(
    vibe_or_config: TrackContent,
    *,
    duration: float = 8.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    sample_rate: int = SAMPLE_RATE,
    hooks: RenderHooks | None = None,
) -> Audio:
    return await asyncio.to_thread(
        render,
        vibe_or_config,
        duration=duration,
        model=model,
        config=config,
        update=update,
        sample_rate=sample_rate,
        hooks=hooks,
    )


def stream(
    *items: StreamItemsInput,
    duration: float | None = None,
    transition: float = 5.0,
    chunk_seconds: float = 1.0,
    pattern_seconds: float | None = None,
    transition_seconds: float | None = None,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
    queue_maxsize: int = 0,
) -> AudioStream:
    items_list = _normalize_items(items)
    stream_items = _coerce_stream_items(items_list, duration=duration, transition=transition)
    resolved_transition = transition if transition_seconds is None else transition_seconds
    return _stream_from_items(
        stream_items,
        chunk_seconds=chunk_seconds,
        pattern_seconds=pattern_seconds,
        transition_seconds=resolved_transition,
        model=model,
        config=config,
        update=update,
        fallback=fallback,
        fallback_model=fallback_model,
        hooks=hooks,
        queue_maxsize=queue_maxsize,
    )


def _normalize_items(items: tuple[StreamItemsInput, ...]) -> list[StreamItems]:
    if (
        len(items) == 1
        and isinstance(items[0], Sequence)
        and not isinstance(
            items[0],
            (str, bytes),
        )
    ):
        sequence = items[0]
        items_list = list(sequence)
    else:
        items_list: list[StreamItems] = []
        for item in items:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                raise InvalidConfigError("stream expects items or a single sequence")
            items_list.append(item)

    if not items_list:
        raise InvalidConfigError("stream expects at least one item")

    return items_list


def _coerce_stream_items(
    items: Sequence[StreamItems],
    *,
    duration: float | None,
    transition: float,
) -> list[Streamable | TrackContent]:
    if duration is None:
        stream_items: list[Streamable | TrackContent] = []
        for item in items:
            if isinstance(item, Track):
                stream_items.append(item.to_streamable())
            else:
                stream_items.append(item)
        return stream_items

    count = len(items)
    per_track = duration / count
    stream_items = []
    for item in items:
        if isinstance(item, Streamable):
            stream_items.append(item)
        elif isinstance(item, Track):
            stream_items.append(item.to_streamable())
        else:
            stream_items.append(
                Track(content=item, duration=per_track, transition=transition).to_streamable()
            )
    return stream_items


def _emit_render_hooks(
    hooks: RenderHooks | None,
    *,
    kind: Literal["start", "synth_start", "synth_end", "end", "error"],
    error: Exception | None = None,
) -> None:
    if hooks is None:
        return
    try:
        match kind:
            case "start":
                if hooks.on_start is not None:
                    hooks.on_start()
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


def _stream_from_items(
    items: Iterable[Streamable | TrackContent],
    *,
    chunk_seconds: float,
    pattern_seconds: float | None,
    transition_seconds: float,
    model: ModelSpec,
    config: ConfigInput | None,
    update: UpdateInput | None,
    fallback: FallbackInput | None,
    fallback_model: ModelSpec,
    hooks: StreamHooks | None,
    queue_maxsize: int,
) -> AudioStream:
    resolved_model = _coerce_model(model)
    resolved_fallback = _coerce_model(fallback_model)
    resolved_hooks = hooks
    indicator = None
    if hooks is None:
        indicator = RichIndicator()
        resolved_hooks = indicator.stream_hooks()

    def sync_factory() -> Iterable[FloatArray]:
        _ = indicator
        return stream_raw(
            items,
            chunk_seconds=chunk_seconds,
            pattern_seconds=pattern_seconds,
            transition_seconds=transition_seconds,
            model=resolved_model,
            config=config,
            update=update,
            fallback=fallback,
            fallback_model=resolved_fallback,
            hooks=resolved_hooks,
            queue_maxsize=queue_maxsize,
        )

    def async_factory() -> AsyncIterable[FloatArray]:
        _ = indicator
        return astream_raw(
            items,
            chunk_seconds=chunk_seconds,
            pattern_seconds=pattern_seconds,
            transition_seconds=transition_seconds,
            model=resolved_model,
            config=config,
            update=update,
            fallback=fallback,
            fallback_model=resolved_fallback,
            hooks=resolved_hooks,
        )

    return AudioStream(sync_factory, async_factory, sample_rate=SAMPLE_RATE)


def live(
    source: LiveSource,
    *,
    chunk_seconds: float = 1.0,
    pattern_seconds: float | None = None,
    transition_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
    queue_maxsize: int = 0,
    sample_rate: int = SAMPLE_RATE,
) -> LiveStream:
    return LiveStream(
        source,
        chunk_seconds=chunk_seconds,
        pattern_seconds=pattern_seconds,
        transition_seconds=transition_seconds,
        model=model,
        config=config,
        update=update,
        fallback=fallback,
        fallback_model=fallback_model,
        hooks=hooks,
        queue_maxsize=queue_maxsize,
        sample_rate=sample_rate,
    )


def _seconds_to_chunks(seconds: float | None, chunk_seconds: float) -> int | None:
    if seconds is None:
        return None
    if seconds <= 0:
        return 0
    return int(math.ceil(seconds / max(chunk_seconds, 1e-6)))


def _coerce_model(model: ModelSpec) -> ModelSpec:
    if isinstance(model, str) and model.startswith(EXTERNAL_PREFIX):
        name = model[len(EXTERNAL_PREFIX) :].strip()
        if not name:
            raise InvalidConfigError("external model string must be 'external:<model-name>'")
        try:
            from .providers.litellm import LiteLLMAdapter
        except ImportError as exc:
            _LOGGER.warning("LiteLLM not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError(
                "LiteLLM is not installed; install latentscore[litellm]"
            ) from exc
        return LiteLLMAdapter(model=name)

    return model
