"""
Simplified async streaming API for latentscore.

Usage:
    async def my_composition():
        yield "warm ambient sunrise"
        await asyncio.sleep(30)
        yield MusicConfigUpdate(brightness=Step(+1))
        await asyncio.sleep(30)
        yield MusicConfig(tempo="fast")

    async for chunk in aplay(my_composition()):
        play_audio(chunk)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Union

import numpy as np

from latentscore.config import MusicConfig, MusicConfigUpdate, Step, SynthConfig
from latentscore.synth import FloatArray, SAMPLE_RATE, assemble, interpolate_configs

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

# Note: TypeAlias causes pyright issues with pattern matching, so we use direct annotation
Instruction = str | MusicConfig | MusicConfigUpdate
AudioChunk = FloatArray  # NDArray[np.float64] from synth

# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigResolved:
    """Emitted when an instruction becomes a concrete config."""

    config: MusicConfig
    source: Instruction


@dataclass(frozen=True, slots=True)
class TransitionStart:
    """Emitted when beginning a crossfade between configs."""

    from_config: MusicConfig
    to_config: MusicConfig


@dataclass(frozen=True, slots=True)
class ChunkReady:
    """Emitted when an audio chunk is generated."""

    t_offset: float
    duration: float


Event = Union[ConfigResolved, TransitionStart, ChunkReady]
EventCallback = Callable[[Event], None]


@dataclass(frozen=True, slots=True)
class Hooks:
    """Optional callbacks for stream events."""

    on_event: EventCallback | None = None


# -----------------------------------------------------------------------------
# Streaming Cache
# -----------------------------------------------------------------------------


def _seconds_per_beat(config: SynthConfig) -> float:
    tempo = float(np.clip(config.tempo, 0.0, 1.0))
    bpm = 55.0 + 110.0 * tempo
    return 60.0 / bpm


def _default_pattern_seconds(config: SynthConfig, chunk_seconds: float) -> float:
    beats_per_bar = 4
    seconds_per_bar = beats_per_bar * _seconds_per_beat(config)
    phrase_bars = max(2, int(config.phrase_len_bars))
    min_seconds = max(8.0, chunk_seconds * 4.0)
    bars = max(phrase_bars, int(np.ceil(min_seconds / seconds_per_bar)))
    bars = max(1, min(64, bars))
    return bars * seconds_per_bar


@dataclass(slots=True)
class _ChunkCache:
    buffer: FloatArray
    chunk_samples: int
    cursor: int = 0

    def next_chunk(self) -> FloatArray:
        if self.chunk_samples <= 0:
            return np.zeros(0, dtype=self.buffer.dtype)
        if self.buffer.size == 0:
            return np.zeros(self.chunk_samples, dtype=np.float64)

        if self.chunk_samples >= self.buffer.size:
            repeats = int(np.ceil(self.chunk_samples / self.buffer.size))
            tiled = np.tile(self.buffer, repeats)
            chunk = tiled[: self.chunk_samples].copy()
            self.cursor = self.chunk_samples % self.buffer.size
            return chunk

        end = self.cursor + self.chunk_samples
        if end <= self.buffer.size:
            chunk = self.buffer[self.cursor : end].copy()
            self.cursor = 0 if end == self.buffer.size else end
            return chunk

        part_a = self.buffer[self.cursor :]
        part_b = self.buffer[: end - self.buffer.size]
        chunk = np.concatenate((part_a, part_b)).copy()
        self.cursor = end - self.buffer.size
        return chunk


def _build_chunk_cache(
    config: MusicConfig,
    *,
    chunk_seconds: float,
    pattern_seconds: float | None,
    t_offset: float,
) -> _ChunkCache:
    synth = config.to_internal()
    pattern = pattern_seconds or _default_pattern_seconds(synth, chunk_seconds)
    pattern = max(pattern, chunk_seconds)
    buffer = assemble(synth, pattern, normalize=False, t_offset=t_offset)
    chunk_samples = max(1, int(round(chunk_seconds * SAMPLE_RATE)))
    return _ChunkCache(buffer=buffer, chunk_samples=chunk_samples)


# -----------------------------------------------------------------------------
# LLM Integration
# -----------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a music configuration generator for ambient/generative synthesis.
Given a vibe description, output a JSON object with music parameters.
Use sensible defaults for any fields not clearly implied by the vibe."""


async def _vibe_to_config(vibe: str, model: str) -> MusicConfig:
    """Convert vibe text to MusicConfig via LLM."""
    from litellm import acompletion  # type: ignore[import-untyped]

    schema = MusicConfig.model_json_schema()
    system = f"{_SYSTEM_PROMPT}\n\nOutput JSON matching:\n{json.dumps(schema, indent=2)}"

    resp = await acompletion(  # type: ignore[no-untyped-call]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": vibe},
        ],
        response_format={"type": "json_object"},
    )
    content: str = resp.choices[0].message.content  # type: ignore[union-attr]
    return MusicConfig.model_validate_json(content)


# -----------------------------------------------------------------------------
# Core Streaming API
# -----------------------------------------------------------------------------


async def aplay(
    source: AsyncIterator[Instruction],
    *,
    model: str = "gpt-4o-mini",
    chunk_seconds: float = 2.0,
    transition_seconds: float = 4.0,  # Should be >= 2*chunk_seconds for smooth fade
    pattern_seconds: float | None = None,
    hooks: Hooks | None = None,
) -> AsyncIterator[AudioChunk]:
    """
    Stream audio chunks from an async instruction source.

    Args:
        source: Async generator yielding str, MusicConfig, or MusicConfigUpdate.
                First instruction MUST be str or MusicConfig (not an update).
        model: LiteLLM model name for vibe-to-config conversion.
        chunk_seconds: Duration of each audio chunk.
        transition_seconds: Crossfade duration when config changes.
        pattern_seconds: Optional steady-state pattern cache length (seconds).
        hooks: Optional event callbacks.

    Yields:
        Audio chunks as float32 numpy arrays.

    The current config keeps playing while waiting for new instructions.
    Transitions only begin after the new config is fully resolved.
    """

    def _noop(_: Event) -> None:
        pass

    emit = hooks.on_event if hooks and hooks.on_event else _noop

    async def resolve(instr: Instruction, curr: MusicConfig | None) -> MusicConfig:
        if isinstance(instr, str):
            return await _vibe_to_config(instr, model)
        if isinstance(instr, MusicConfigUpdate):
            if curr is None:
                raise ValueError("Cannot apply MusicConfigUpdate without base config")
            return instr.apply_to(curr)
        # Must be MusicConfig at this point
        return instr

    # --- Bootstrap: first instruction ---
    try:
        first = await anext(source)
    except StopAsyncIteration:
        return  # Empty source, nothing to play

    if isinstance(first, MusicConfigUpdate):
        raise ValueError("First instruction cannot be MusicConfigUpdate")

    current = await resolve(first, None)
    emit(ConfigResolved(current, first))
    t_offset = 0.0
    chunk_cache = _build_chunk_cache(
        current,
        chunk_seconds=chunk_seconds,
        pattern_seconds=pattern_seconds,
        t_offset=t_offset,
    )

    # --- Background instruction handling ---
    pending: asyncio.Task[tuple[Instruction, MusicConfig] | None] | None = None
    exhausted = False

    async def fetch_and_resolve(curr: MusicConfig) -> tuple[Instruction, MusicConfig] | None:
        try:
            instr = await anext(source)
            return (instr, await resolve(instr, curr))
        except StopAsyncIteration:
            return None

    # --- Main loop ---
    while True:
        # Start background fetch if idle and source not exhausted
        if pending is None and not exhausted:
            pending = asyncio.create_task(fetch_and_resolve(current))

        # Check if new config is ready
        if pending is not None and pending.done():
            result = pending.result()
            pending = None

            if result is None:
                exhausted = True
            else:
                instr, target = result
                emit(ConfigResolved(target, instr))
                emit(TransitionStart(current, target))
                chunk_cache = None

                # Crossfade transition
                steps = max(1, int(transition_seconds / chunk_seconds))
                src_synth = current.to_internal()
                dst_synth = target.to_internal()

                for i in range(1, steps + 1):
                    interp = interpolate_configs(src_synth, dst_synth, i / steps)
                    chunk = assemble(interp, chunk_seconds, normalize=False, t_offset=t_offset)
                    emit(ChunkReady(t_offset, chunk_seconds))
                    yield chunk
                    t_offset += chunk_seconds

                current = target
                chunk_cache = _build_chunk_cache(
                    current,
                    chunk_seconds=chunk_seconds,
                    pattern_seconds=pattern_seconds,
                    t_offset=t_offset,
                )
                continue  # Restart loop without emitting steady-state chunk

        # Steady state: render current config
        if chunk_cache is None:
            chunk_cache = _build_chunk_cache(
                current,
                chunk_seconds=chunk_seconds,
                pattern_seconds=pattern_seconds,
                t_offset=t_offset,
            )
        chunk = chunk_cache.next_chunk()
        emit(ChunkReady(t_offset, chunk_seconds))
        yield chunk
        t_offset += chunk_seconds

        # Yield to event loop so background tasks can progress
        await asyncio.sleep(0)


# -----------------------------------------------------------------------------
# Convenience: Step re-export
# -----------------------------------------------------------------------------

__all__ = [
    "aplay",
    "Instruction",
    "AudioChunk",
    "Event",
    "ConfigResolved",
    "TransitionStart",
    "ChunkReady",
    "Hooks",
    "MusicConfig",
    "MusicConfigUpdate",
    "Step",
]

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from latentscore.audio import ensure_audio_contract
    from latentscore.synth import SAMPLE_RATE

    async def example_composition() -> AsyncIterator[Instruction]:
        """Fixed config to generate a consistent sound for testing."""
        yield MusicConfig(
            tempo="very_slow",
            root="d",
            mode="dorian",
            brightness="very_dark",
            density=4,
            bass="drone",
            pad="warm_slow",
        )

    import asyncio

    async def get_audio_chunks(chunk_seconds: float, total_seconds: float) -> list[np.ndarray]:
        """Generate audio chunks using a fixed composition for given chunk size and duration."""
        chunks = []
        seconds_collected = 0.0

        async for chunk in aplay(
            example_composition(),
            chunk_seconds=chunk_seconds,
        ):
            # ensure numpy float64 array
            normalized = ensure_audio_contract(chunk, sample_rate=SAMPLE_RATE)
            chunks.append(np.copy(normalized))
            seconds_collected += chunk_seconds
            if seconds_collected >= total_seconds:
                break
        return chunks

    def chunk_energies(chunks: list[np.ndarray]) -> list[float]:
        return [float(np.mean(np.square(chunk))) for chunk in chunks]

    def consecutive_correlation(chunks: list[np.ndarray]) -> list[float]:
        """Pearson correlation between consecutive chunks (flattened)."""
        corrs = []
        for i in range(len(chunks) - 1):
            a = chunks[i].flatten()
            b = chunks[i + 1].flatten()
            # If chunk is zero energy, correlation is nan: handle separately
            if np.all(a == 0) or np.all(b == 0):
                corrs.append(float("nan"))
                continue
            c = np.corrcoef(a, b)[0, 1]
            corrs.append(float(c))
        return corrs

    async def run_tests():
        total_duration = 10.0  # seconds
        for chunk_size in [1.0, 2.0]:
            print(f"\n== Chunk size: {chunk_size} s ==")
            chunks = await get_audio_chunks(chunk_size, total_duration)
            energies = chunk_energies(chunks)
            corrs = consecutive_correlation(chunks)
            print(f" Energies for {len(energies)} chunks: {[round(e, 5) for e in energies]}")
            if corrs:
                print(
                    f" Correlations between consecutive chunks: {[round(c, 5) if not np.isnan(c) else 'nan' for c in corrs]}"
                )
            else:
                print(" Not enough chunks for correlation")

    if True:
        asyncio.run(run_tests())
