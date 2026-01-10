from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from .audio import SAMPLE_RATE, FloatArray, ensure_audio_contract, write_wav
from .config import (
    ConfigInput,
    MusicConfig,
    MusicConfigUpdate,
    UpdateInput,
    coerce_internal_config,
    coerce_internal_update,
    merge_internal_config,
)
from .errors import InvalidConfigError, ModelNotAvailableError, PlaybackError
from .indicators import RichIndicator
from .main import (
    FallbackInput,
    PreviewPolicy,
    RenderHooks,
    Streamable,
    StreamHooks,
)
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
from .config import SynthConfig
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


StreamItems = Track | TrackContent
StreamItemsInput = StreamItems | list[StreamItems] | tuple[StreamItems, ...]


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
        model: ModelSpec = "fast",
        config: ConfigInput | None = None,
        update: UpdateInput | None = None,
        prefetch_depth: int = 1,
        preview_policy: PreviewPolicy = "none",
        fallback: FallbackInput | None = None,
        fallback_model: ModelSpec = "fast",
        hooks: StreamHooks | None = None,
        queue_maxsize: int = 0,
    ) -> AudioStream:
        return _stream_from_tracks(
            list(self.tracks),
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
            queue_maxsize=queue_maxsize,
        )

    def render(self, **kwargs: Any) -> Audio:
        return self.stream(**kwargs).collect()

    def play(self, **kwargs: Any) -> None:
        self.stream(**kwargs).play()


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


def stream(
    *items: StreamItemsInput,
    duration: float = 60.0,
    transition: float = 5.0,
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
) -> AudioStream:
    items_list = _normalize_items(items)
    tracks = _coerce_tracks(items_list, duration=duration, transition=transition)
    return _stream_from_tracks(
        tracks,
        chunk_seconds=chunk_seconds,
        model=model,
        config=config,
        update=update,
        prefetch_depth=prefetch_depth,
        preview_policy=preview_policy,
        fallback=fallback,
        fallback_model=fallback_model,
        hooks=hooks,
        queue_maxsize=queue_maxsize,
    )


def _normalize_items(items: tuple[StreamItemsInput, ...]) -> list[StreamItems]:
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        sequence = items[0]
        items_list = list(sequence)
    else:
        items_list: list[StreamItems] = []
        for item in items:
            if isinstance(item, (list, tuple)):
                raise InvalidConfigError("stream expects items or a single list/tuple")
            items_list.append(item)

    if not items_list:
        raise InvalidConfigError("stream expects at least one item")

    return items_list


def _coerce_tracks(
    items: Sequence[StreamItems],
    *,
    duration: float,
    transition: float,
) -> list[Track]:
    count = len(items)
    per_track = duration / count
    tracks: list[Track] = []
    for item in items:
        if isinstance(item, Track):
            tracks.append(item)
        else:
            tracks.append(Track(content=item, duration=per_track, transition=transition))
    return tracks


def _streamables_for_tracks(tracks: Sequence[Track]) -> Iterable[Streamable]:
    for track in tracks:
        yield track.to_streamable()


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


def _stream_from_tracks(
    tracks: list[Track],
    *,
    chunk_seconds: float,
    model: ModelSpec,
    config: ConfigInput | None,
    update: UpdateInput | None,
    prefetch_depth: int,
    preview_policy: PreviewPolicy,
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
            _streamables_for_tracks(tracks),
            chunk_seconds=chunk_seconds,
            model=resolved_model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=resolved_fallback,
            hooks=resolved_hooks,
            queue_maxsize=queue_maxsize,
        )

    def async_factory() -> AsyncIterable[FloatArray]:
        _ = indicator
        return astream_raw(
            _streamables_for_tracks(tracks),
            chunk_seconds=chunk_seconds,
            model=resolved_model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=resolved_fallback,
            hooks=resolved_hooks,
        )

    return AudioStream(sync_factory, async_factory, sample_rate=SAMPLE_RATE)


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
