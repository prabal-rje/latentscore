# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Coroutine, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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
from .models import ModelForGeneratingMusicConfig, ModelSpec, resolve_model
from .synth import MusicConfig as SynthConfig
from .synth import assemble, interpolate_configs

StreamContent = str | ConfigInput | UpdateInput

_MIN_CHUNK_SECONDS = 1e-6


class Streamable(BaseModel):
    """Structured streaming input with timing controls."""

    content: StreamContent
    duration: float = Field(default=60.0, gt=0.0)
    transition_duration: float = Field(default=1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


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
    return _MusicConfigInternal(**asdict(config))


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


def _resolve_target(
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
            target = _run_async(model.generate(item)).to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported input type: {type(item).__name__}")

    return _apply_update(target, update)


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
) -> FloatArray:
    resolved = resolve_model(model)
    match config:
        case None:
            base = _run_async(resolved.generate(vibe)).to_internal()
        case _:
            base = coerce_internal_config(config)
    target = _apply_update(base, update)

    audio = assemble(_to_synth_config(target), duration)
    return ensure_audio_contract(audio, sample_rate=SAMPLE_RATE)


def stream(
    items: Iterable[Streamable],
    *,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
) -> Iterable[FloatArray]:
    items_obj: object = items
    match items_obj:
        case Streamable():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("stream expects an iterable of Streamable items")
        case str():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("stream expects an iterable of Streamable items")
        case _:
            pass

    match chunk_seconds:
        case (float() | int()) as value if value <= 0:
            raise InvalidConfigError("chunk_seconds must be greater than 0")
        case float() | int():
            pass
        case _:
            raise InvalidConfigError("chunk_seconds must be a number")

    current = coerce_internal_config(config) if config is not None else None
    resolved = resolve_model(model)

    def _chunk_for(config_item: _MusicConfigInternal) -> FloatArray:
        chunk = assemble(_to_synth_config(config_item), chunk_seconds)
        return ensure_audio_contract(chunk, sample_rate=SAMPLE_RATE)

    for item in items:
        target = _resolve_target(
            item.content,
            current=current,
            model=resolved,
            update=update,
        )
        total_chunks = _chunk_count(item.duration, chunk_seconds)
        transition_chunks = (
            0
            if current is None
            else min(total_chunks, _transition_steps(item.transition_duration, chunk_seconds))
        )

        if transition_chunks > 0:
            for config_item in _iter_transition_configs(current, target, transition_chunks):
                current = config_item
                yield _chunk_for(config_item)
        else:
            current = target

        match current:
            case _MusicConfigInternal() as current_config:
                pass
            case None:
                raise InvalidConfigError("Stream exhausted without a current config")
            case _:
                raise InvalidConfigError("Stream exhausted with an invalid config")

        for _ in range(total_chunks - transition_chunks):
            yield _chunk_for(current_config)


def stream_texts(
    prompts: Iterable[str],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
) -> Iterable[FloatArray]:
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
    )


def stream_configs(
    configs: Iterable[ConfigInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
) -> Iterable[FloatArray]:
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
    )


def stream_updates(
    updates: Iterable[UpdateInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
) -> Iterable[FloatArray]:
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
    )


def save_wav(path: str, audio_or_chunks: AudioNumbers | Iterable[AudioNumbers]) -> str:
    written = write_wav(path, audio_or_chunks, sample_rate=SAMPLE_RATE)
    return str(written)


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
) -> AsyncIterable[FloatArray]:
    match chunk_seconds:
        case (float() | int()) as value if value <= 0:
            raise InvalidConfigError("chunk_seconds must be greater than 0")
        case float() | int():
            pass
        case _:
            raise InvalidConfigError("chunk_seconds must be a number")

    current = coerce_internal_config(config) if config is not None else None
    resolved = resolve_model(model)

    async def _render_chunk(config_item: _MusicConfigInternal) -> FloatArray:
        return await asyncio.to_thread(
            lambda: ensure_audio_contract(
                assemble(_to_synth_config(config_item), chunk_seconds),
                sample_rate=SAMPLE_RATE,
            )
        )

    async for item in _coerce_async_items(items):
        target = await _resolve_target_async(
            item.content,
            current=current,
            model=resolved,
            update=update,
        )
        total_chunks = _chunk_count(item.duration, chunk_seconds)
        transition_chunks = (
            0
            if current is None
            else min(total_chunks, _transition_steps(item.transition_duration, chunk_seconds))
        )

        if transition_chunks > 0:
            for config_item in _iter_transition_configs(current, target, transition_chunks):
                current = config_item
                yield await _render_chunk(config_item)
        else:
            current = target

        match current:
            case _MusicConfigInternal() as current_config:
                pass
            case None:
                raise InvalidConfigError("Stream exhausted without a current config")
            case _:
                raise InvalidConfigError("Stream exhausted with an invalid config")

        for _ in range(total_chunks - transition_chunks):
            yield await _render_chunk(current_config)
