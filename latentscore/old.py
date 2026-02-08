"""Legacy API smoke tests for streaming/render behavior.

Toggle the flags below to enable/disable specific checks.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import math
import os

import numpy as np

import latentscore as ls
from latentscore import MusicConfig, MusicConfigUpdate, Streamable
from latentscore.audio import ensure_audio_contract
from latentscore.synth import assemble
from latentscore.config import Step


# -----------------------------------------------------------------------------
# Test switches (set True/False)
# -----------------------------------------------------------------------------

RUN_ENERGY_MATCH_TEST = True  # Compare render vs stream RMS
RUN_SYNC_STREAM_TEST = True  # Sync stream yields expected chunk sizes
RUN_ASYNC_STREAM_TEST = True  # Async stream yields expected chunk sizes
RUN_STEP_UPDATE_TEST = True  # Step updates resolve without errors
RUN_GENERATOR_STREAM_TEST = True  # stream_raw accepts generator input
RUN_TRANSITION_TEST = True  # Transition path yields expected chunks
RUN_INSTRUCTION_CONTINUATION_TEST = True  # Instruction keeps streaming after generator ends
PLAY_AUDIO = False  # Play audio during each test
AUDIBLE_SECONDS = 8.0  # Duration per test when PLAY_AUDIO is enabled
PATTERN_SECONDS = 4.0 if PLAY_AUDIO else 1.0
USE_STUB_SYNTH = os.environ.get("LATENTSCORE_STUB_SYNTH", "").lower() in ("1", "true")


class DummyModel:
    async def generate(self, vibe: str) -> MusicConfig:
        _ = vibe
        return MusicConfig()


if USE_STUB_SYNTH:
    import latentscore.main as main

    def _stub_assemble(*_args, **_kwargs):
        return np.zeros(int(ls.SAMPLE_RATE), dtype=np.float32)

    main.assemble = _stub_assemble


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.square(audio)))


def _test_duration(short_seconds: float) -> float:
    return AUDIBLE_SECONDS if PLAY_AUDIO else short_seconds


def _audible_chunk_count(default_count: int, chunk_seconds: float = 1.0) -> int:
    if not PLAY_AUDIO:
        return default_count
    return max(default_count, int(math.ceil(AUDIBLE_SECONDS / chunk_seconds)))


def _should_play() -> bool:
    return PLAY_AUDIO and not USE_STUB_SYNTH


def _play_audio(samples: np.ndarray, *, label: str) -> None:
    if not _should_play():
        return
    from latentscore.playback import play_audio

    print(f"[audio] {label}")
    play_audio(samples, sample_rate=ls.SAMPLE_RATE)


def _play_stream(chunks: list[np.ndarray], *, label: str) -> None:
    if not _should_play():
        return
    from latentscore.playback import play_stream

    print(f"[audio] {label}")
    play_stream(chunks, sample_rate=ls.SAMPLE_RATE)


def _collect_stream(items: list[Streamable]) -> list[np.ndarray]:
    return list(ls.stream_raw(items, model=DummyModel(), pattern_seconds=PATTERN_SECONDS))


async def _collect_astream(items: list[Streamable]) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    async for chunk in ls.astream_raw(items, model=DummyModel(), pattern_seconds=PATTERN_SECONDS):
        chunks.append(chunk)
    return chunks


async def _collect_astream_n(items, count: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    async with aclosing(
        ls.astream_raw(items, model=DummyModel(), pattern_seconds=PATTERN_SECONDS)
    ) as stream:
        async for chunk in stream:
            chunks.append(chunk)
            if len(chunks) >= count:
                break
    return chunks


def test_energy_match() -> None:
    if USE_STUB_SYNTH:
        print("[skip] energy_match (stub synth enabled)")
        return
    duration = _test_duration(1.0)
    config = MusicConfig(tempo="slow", mode="minor", brightness="dark")
    render_audio = ensure_audio_contract(
        assemble(config.to_internal(), duration, normalize=False),
        sample_rate=ls.SAMPLE_RATE,
        check_peak=False,
    )
    items = [Streamable(content=config, duration=duration, transition_duration=0.0)]
    chunks = _collect_stream(items)
    stream_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    render_rms = _rms(render_audio)
    stream_rms = _rms(stream_audio)
    ratio = stream_rms / render_rms if render_rms > 0 else 0.0
    print(
        f"[energy_match] render_rms={render_rms:.6f} stream_rms={stream_rms:.6f} ratio={ratio:.3f}"
    )
    _play_audio(render_audio, label="energy_match render")
    _play_audio(stream_audio, label="energy_match stream")
    if render_rms > 0 and not (0.3 <= ratio <= 3.0):
        raise AssertionError("RMS ratio out of expected range")


def test_sync_stream() -> None:
    duration = _test_duration(2.0)
    items = [Streamable(content=MusicConfig(), duration=duration, transition_duration=0.0)]
    chunks = _collect_stream(items)
    if not chunks:
        raise AssertionError("No chunks returned from stream_raw")
    expected = int(round(ls.SAMPLE_RATE * 1.0))
    for chunk in chunks:
        if abs(len(chunk) - expected) > 1:
            raise AssertionError("Chunk length does not match sample rate")
    _play_stream(chunks, label="sync_stream")


def test_async_stream() -> None:
    duration = _test_duration(2.0)
    items = [Streamable(content=MusicConfig(), duration=duration, transition_duration=0.0)]
    chunks = asyncio.run(_collect_astream(items))
    if not chunks:
        raise AssertionError("No chunks returned from astream_raw")
    expected = int(round(ls.SAMPLE_RATE * 1.0))
    for chunk in chunks:
        if abs(len(chunk) - expected) > 1:
            raise AssertionError("Async chunk length does not match sample rate")
    _play_stream(chunks, label="async_stream")


def test_step_update() -> None:
    duration = _test_duration(1.0)
    items = [
        Streamable(
            content=MusicConfig(brightness="medium"),
            duration=duration,
            transition_duration=0.0,
        ),
        Streamable(
            content=MusicConfigUpdate(brightness=Step(+1)),
            duration=duration,
            transition_duration=0.0,
        ),
    ]
    chunks = _collect_stream(items)
    if not chunks:
        raise AssertionError("No chunks returned for Step update test")
    _play_stream(chunks, label="step_update")


def test_generator_stream() -> None:
    duration = _test_duration(1.0)

    def _gen():
        yield Streamable(
            content=MusicConfig(tempo="slow"), duration=duration, transition_duration=0.0
        )
        yield Streamable(
            content=MusicConfig(tempo="fast"), duration=duration, transition_duration=0.0
        )

    chunks = list(ls.stream_raw(_gen(), model=DummyModel(), pattern_seconds=PATTERN_SECONDS))
    if len(chunks) < 2:
        raise AssertionError("Generator stream returned too few chunks")
    _play_stream(chunks, label="generator_stream")


def test_instruction_continues_after_generator() -> None:
    def _gen():
        yield "warm sunrise"

    count = _audible_chunk_count(3)
    chunks = asyncio.run(_collect_astream_n(_gen(), count))
    if len(chunks) < count:
        raise AssertionError("Instruction stream ended early")
    _play_stream(chunks, label="instruction_continuation")


def test_transition() -> None:
    duration = _test_duration(2.0)
    chunk_seconds = 1.0
    items = [
        Streamable(
            content=MusicConfig(tempo="slow"),
            duration=duration,
            transition_duration=1.0,
        ),
        Streamable(
            content=MusicConfig(tempo="fast"),
            duration=duration,
            transition_duration=1.0,
        ),
    ]
    chunks = _collect_stream(items)
    expected = 2 * int(round(duration / chunk_seconds))
    if len(chunks) != expected:
        raise AssertionError(f"Expected {expected} chunks, got {len(chunks)}")
    _play_stream(chunks, label="transition")


def _run_test(name: str, enabled: bool, fn) -> bool:
    if not enabled:
        print(f"[skip] {name}")
        return True
    try:
        fn()
        print(f"[ok] {name}")
        return True
    except Exception as exc:
        print(f"[fail] {name}: {exc}")
        return False


def main() -> None:
    checks = [
        ("energy_match", RUN_ENERGY_MATCH_TEST, test_energy_match),
        ("sync_stream", RUN_SYNC_STREAM_TEST, test_sync_stream),
        ("async_stream", RUN_ASYNC_STREAM_TEST, test_async_stream),
        ("step_update", RUN_STEP_UPDATE_TEST, test_step_update),
        ("generator_stream", RUN_GENERATOR_STREAM_TEST, test_generator_stream),
        ("transition", RUN_TRANSITION_TEST, test_transition),
        (
            "instruction_continuation",
            RUN_INSTRUCTION_CONTINUATION_TEST,
            test_instruction_continues_after_generator,
        ),
    ]
    failures = 0
    for name, enabled, fn in checks:
        if not _run_test(name, enabled, fn):
            failures += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
