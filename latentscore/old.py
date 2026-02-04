"""Legacy API smoke tests for streaming/render behavior.

Toggle the flags below to enable/disable specific checks.
"""

from __future__ import annotations

import asyncio

import numpy as np

import latentscore as ls
from latentscore import MusicConfig, MusicConfigUpdate, Streamable
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


class DummyModel:
    async def generate(self, vibe: str) -> MusicConfig:
        _ = vibe
        return MusicConfig()


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.square(audio)))


def _collect_stream(items: list[Streamable]) -> list[np.ndarray]:
    return list(ls.stream_raw(items, model=DummyModel()))


async def _collect_astream(items: list[Streamable]) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    async for chunk in ls.astream_raw(items, model=DummyModel()):
        chunks.append(chunk)
    return chunks


def test_energy_match() -> None:
    duration = 8.0
    config = MusicConfig(tempo="slow", mode="minor", brightness="dark")
    render_audio = ls.render_raw("smoke", duration=duration, config=config)
    items = [Streamable(content=config, duration=duration, transition_duration=0.0)]
    chunks = _collect_stream(items)
    stream_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    render_rms = _rms(render_audio)
    stream_rms = _rms(stream_audio)
    ratio = stream_rms / render_rms if render_rms > 0 else 0.0
    print(
        f"[energy_match] render_rms={render_rms:.6f} stream_rms={stream_rms:.6f} ratio={ratio:.3f}"
    )
    if render_rms > 0 and not (0.5 <= ratio <= 2.0):
        raise AssertionError("RMS ratio out of expected range")


def test_sync_stream() -> None:
    duration = 2.0
    items = [Streamable(content=MusicConfig(), duration=duration, transition_duration=0.0)]
    chunks = _collect_stream(items)
    if not chunks:
        raise AssertionError("No chunks returned from stream_raw")
    expected = int(round(ls.SAMPLE_RATE * 1.0))
    for chunk in chunks:
        if abs(len(chunk) - expected) > 1:
            raise AssertionError("Chunk length does not match sample rate")


def test_async_stream() -> None:
    duration = 2.0
    items = [Streamable(content=MusicConfig(), duration=duration, transition_duration=0.0)]
    chunks = asyncio.run(_collect_astream(items))
    if not chunks:
        raise AssertionError("No chunks returned from astream_raw")
    expected = int(round(ls.SAMPLE_RATE * 1.0))
    for chunk in chunks:
        if abs(len(chunk) - expected) > 1:
            raise AssertionError("Async chunk length does not match sample rate")


def test_step_update() -> None:
    items = [
        Streamable(content=MusicConfig(brightness="medium"), duration=1.0, transition_duration=0.0),
        Streamable(
            content=MusicConfigUpdate(brightness=Step(+1)),
            duration=1.0,
            transition_duration=0.0,
        ),
    ]
    chunks = _collect_stream(items)
    if not chunks:
        raise AssertionError("No chunks returned for Step update test")


def test_generator_stream() -> None:
    def _gen():
        yield Streamable(content=MusicConfig(tempo="slow"), duration=1.0, transition_duration=0.0)
        yield Streamable(content=MusicConfig(tempo="fast"), duration=1.0, transition_duration=0.0)

    chunks = list(ls.stream_raw(_gen(), model=DummyModel()))
    if len(chunks) < 2:
        raise AssertionError("Generator stream returned too few chunks")


def test_transition() -> None:
    items = [
        Streamable(content=MusicConfig(tempo="slow"), duration=2.0, transition_duration=1.0),
        Streamable(content=MusicConfig(tempo="fast"), duration=2.0, transition_duration=1.0),
    ]
    chunks = _collect_stream(items)
    expected = 4
    if len(chunks) != expected:
        raise AssertionError(f"Expected {expected} chunks, got {len(chunks)}")


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
    ]
    failures = 0
    for name, enabled, fn in checks:
        if not _run_test(name, enabled, fn):
            failures += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
