"""Simpler end-user DX examples.

Focus: render, stream, astream, arender, and generator patterns.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Iterable

import latentscore as ls
import latentscore.dx as dx
from latentscore import ExternalModelSpec, ModelSpec, MusicConfig, MusicConfigUpdate, Streamable
from latentscore.config import Step
from latentscore.errors import ModelNotAvailableError, PlaybackError


# -----------------------------------------------------------------------------
# Toggles
# -----------------------------------------------------------------------------

RUN_RENDER = True
RUN_ARENDER = True
RUN_STREAM = True
RUN_ASTREAM = True
RUN_INSTRUCTION_GENERATOR = True
RUN_STREAMABLE_GENERATOR = True

# Models to try (enable as needed)
RUN_FAST = True
RUN_EXPRESSIVE = False
RUN_GPT52 = False
RUN_SONNET = False

# -----------------------------------------------------------------------------
# Playback + timing
# -----------------------------------------------------------------------------

PLAY_AUDIO = True
RENDER_SECONDS = 6.0 if PLAY_AUDIO else 2.0
STREAM_SECONDS = 8.0 if PLAY_AUDIO else 2.0
CHUNK_SECONDS = 2.0
TRANSITION_SECONDS = 2.0
INSTRUCTION_HOLD_SECONDS = 4.0 if PLAY_AUDIO else 1.0
INSTRUCTION_TAIL_SECONDS = 4.0 if PLAY_AUDIO else 1.0
STREAMABLE_DURATION_SECONDS = 4.0 if PLAY_AUDIO else 1.0


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

GPT52_MODEL = ExternalModelSpec(model="openai/gpt-5.2")
SONNET_MODEL = ExternalModelSpec(model="anthropic/claude-sonnet-4-5-20250929")
EXPRESSIVE_MODEL: ModelSpec = "expressive"


def _guard_env(label: str, required: Iterable[str]) -> bool:
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"[skip] {label}: missing {', '.join(missing)}")
        return False
    return True


def _play_chunks(chunks, *, label: str) -> None:
    if not PLAY_AUDIO:
        return
    from latentscore.playback import play_stream

    print(f"[audio] {label}")
    play_stream(chunks, sample_rate=ls.SAMPLE_RATE)


def _render_demo(model: ModelSpec) -> None:
    audio = dx.render("warm sunrise ambient", duration=RENDER_SECONDS, model=model)
    print("[render] ok")
    if PLAY_AUDIO:
        audio.play()


def _stream_demo(model: ModelSpec) -> None:
    stream = dx.stream(
        "warm sunrise ambient",
        "late night neon",
        duration=STREAM_SECONDS,
        transition=TRANSITION_SECONDS,
        model=model,
    )
    if PLAY_AUDIO:
        stream.play()
        return
    _ = stream.collect()
    print("[stream] ok")


async def _astream_demo(model: ModelSpec) -> None:
    total_seconds = STREAM_SECONDS
    chunks = []
    async for chunk in ls.astream_raw(
        ["warm sunrise ambient"],
        model=model,
        chunk_seconds=CHUNK_SECONDS,
    ):
        chunks.append(chunk)
        if len(chunks) * CHUNK_SECONDS >= total_seconds:
            break
    if PLAY_AUDIO:
        _play_chunks(chunks, label="astream")
    print("[astream] ok")


async def _instruction_generator():
    """Sleep controls how long each yield plays.

    The last yield keeps playing until you stop consuming chunks.
    """
    yield "warm sunrise ambient"
    await asyncio.sleep(INSTRUCTION_HOLD_SECONDS)
    yield MusicConfigUpdate(brightness=Step(+1))
    await asyncio.sleep(INSTRUCTION_HOLD_SECONDS)
    yield MusicConfig(tempo="fast", brightness="bright", density=6)


async def _instruction_generator_demo(model: ModelSpec) -> None:
    total_seconds = INSTRUCTION_HOLD_SECONDS * 2 + INSTRUCTION_TAIL_SECONDS
    start = time.monotonic()

    def _log_event(event) -> None:
        if event.kind == "item_resolve_success":
            elapsed = time.monotonic() - start
            print(f"[instruction] resolved item {event.index} at {elapsed:.1f}s")

    hooks = ls.StreamHooks(on_event=_log_event)
    live = dx.live(
        _instruction_generator(),
        model=model,
        chunk_seconds=CHUNK_SECONDS,
        transition_seconds=TRANSITION_SECONDS,
        hooks=hooks,
    )
    if PLAY_AUDIO:
        live.play(seconds=total_seconds)
    else:
        _ = list(live.chunks(seconds=total_seconds))
    print("[instruction_generator] ok")


def _streamable_generator_demo(model: ModelSpec) -> None:
    def _gen():
        yield Streamable(
            content="warm sunrise ambient",
            duration=STREAMABLE_DURATION_SECONDS,
            transition_duration=TRANSITION_SECONDS,
        )
        yield Streamable(
            content=MusicConfigUpdate(brightness=Step(+1)),
            duration=STREAMABLE_DURATION_SECONDS,
            transition_duration=TRANSITION_SECONDS,
        )
        yield Streamable(
            content="late night neon",
            duration=STREAMABLE_DURATION_SECONDS,
            transition_duration=TRANSITION_SECONDS,
        )

    live = dx.live(
        _gen(),
        model=model,
        chunk_seconds=CHUNK_SECONDS,
        transition_seconds=TRANSITION_SECONDS,
    )
    if PLAY_AUDIO:
        live.play()
    else:
        _ = list(live.chunks())
    print("[streamable_generator] ok")


def _run_model(label: str, model: ModelSpec) -> None:
    try:
        if RUN_RENDER:
            _render_demo(model)
        if RUN_ARENDER:
            _ = asyncio.run(
                dx.arender("warm sunrise ambient", duration=RENDER_SECONDS, model=model)
            )
            print("[arender] ok")
        if RUN_STREAM:
            _stream_demo(model)
        if RUN_ASTREAM:
            asyncio.run(_astream_demo(model))
        if RUN_INSTRUCTION_GENERATOR:
            asyncio.run(_instruction_generator_demo(model))
        if RUN_STREAMABLE_GENERATOR:
            _streamable_generator_demo(model)
    except PlaybackError as exc:
        print(f"[fail] {label}: playback unavailable ({exc})")
    except ModelNotAvailableError as exc:
        print(f"[fail] {label}: model unavailable ({exc})")


def main() -> None:
    if RUN_FAST:
        _run_model("fast", "fast")

    if RUN_EXPRESSIVE:
        _run_model("expressive", EXPRESSIVE_MODEL)

    if RUN_GPT52 and _guard_env("gpt-5.2", ["OPENAI_API_KEY"]):
        _run_model("gpt-5.2", GPT52_MODEL)

    if RUN_SONNET and _guard_env("claude-sonnet-4.5", ["ANTHROPIC_API_KEY"]):
        _run_model("claude-sonnet-4.5", SONNET_MODEL)


if __name__ == "__main__":
    main()
