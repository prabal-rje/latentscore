"""End-user DX smoke tests with real models.

Toggle flags at the top to enable/disable models and playback.
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
# Model toggles
# -----------------------------------------------------------------------------

RUN_FAST_MODEL = True
RUN_EXPRESSIVE_MODEL = False
RUN_GPT52_MODEL = False
RUN_SONNET_MODEL = False

RUN_INSTRUCTION_GENERATOR_EXAMPLE = True
RUN_STREAMABLE_GENERATOR_EXAMPLE = True

# -----------------------------------------------------------------------------
# Playback + durations
# -----------------------------------------------------------------------------

PLAY_AUDIO = True  # Play audio for each test
RENDER_SECONDS = 12.0 if PLAY_AUDIO else 2.0
STREAM_SECONDS = 12.0 if PLAY_AUDIO else 4.0
TRANSITION_SECONDS = 2.0
CHUNK_SECONDS = 2.0
PATTERN_SECONDS = 6.0 if PLAY_AUDIO else 2.0
INSTRUCTION_HOLD_SECONDS = 4.0 if PLAY_AUDIO else 1.0
INSTRUCTION_TAIL_SECONDS = 4.0 if PLAY_AUDIO else 1.0
STREAMABLE_DURATION_SECONDS = 4.0 if PLAY_AUDIO else 1.0

# -----------------------------------------------------------------------------
# Model names (LiteLLM routes)
# -----------------------------------------------------------------------------

# LiteLLM docs (Feb 2026): https://docs.litellm.ai/docs/providers/openai
GPT52_MODEL = "openai/gpt-5.2"

# LiteLLM docs (Feb 2026): https://docs.litellm.ai/docs/providers/anthropic
SONNET_MODEL = "anthropic/claude-sonnet-4-5-20250929"

# Expressive is the fine-tuned local model (download via `latentscore download expressive`).
EXPRESSIVE_MODEL: ModelSpec = "expressive"


def _external(model: str) -> ExternalModelSpec:
    if model.startswith("external:"):
        model = model.removeprefix("external:")
    return ExternalModelSpec(model=model)


def _has_env(var: str) -> bool:
    return bool(os.environ.get(var))


def _guard_model(name: str, required_env: Iterable[str]) -> bool:
    missing = [var for var in required_env if not _has_env(var)]
    if missing:
        print(f"[skip] {name}: missing {', '.join(missing)}")
        return False
    return True


def _play_if_enabled(chunks, *, label: str) -> None:
    if not PLAY_AUDIO:
        return
    from latentscore.playback import play_stream

    print(f"[audio] {label}")
    play_stream(chunks, sample_rate=ls.SAMPLE_RATE)


def _render_test(label: str, model: ModelSpec) -> None:
    audio = dx.render("warm sunrise ambient", duration=RENDER_SECONDS, model=model)
    print(f"[render] {label}")
    if PLAY_AUDIO:
        audio.play()


def _stream_test(label: str, model: ModelSpec) -> None:
    stream = dx.stream(
        "warm sunrise ambient",
        "late night neon",
        duration=STREAM_SECONDS,
        transition=TRANSITION_SECONDS,
        chunk_seconds=CHUNK_SECONDS,
        pattern_seconds=PATTERN_SECONDS,
        model=model,
        fallback="keep_last",
        fallback_model="fast",
    )
    if PLAY_AUDIO:
        stream.play()
        return
    _ = stream.collect()
    print(f"[stream] {label}")


def _update_test(label: str, model: ModelSpec) -> None:
    stream = dx.stream(
        "warm sunrise ambient",
        MusicConfigUpdate(brightness=Step(+1)),
        duration=STREAM_SECONDS,
        transition=TRANSITION_SECONDS,
        chunk_seconds=CHUNK_SECONDS,
        pattern_seconds=PATTERN_SECONDS,
        model=model,
        fallback="keep_last",
        fallback_model="fast",
    )
    if PLAY_AUDIO:
        stream.play()
        return
    _ = stream.collect()
    print(f"[update] {label}")


async def _instruction_generator():
    """Async generator of str/update/config; sleep controls how long each plays."""
    yield "warm sunrise ambient"
    await asyncio.sleep(INSTRUCTION_HOLD_SECONDS)
    yield MusicConfigUpdate(brightness=Step(+1))
    await asyncio.sleep(INSTRUCTION_HOLD_SECONDS)
    yield MusicConfig(tempo="fast", brightness="bright", density=6)


async def _collect_instruction_chunks(model: ModelSpec) -> list:
    chunks = []
    total_seconds = INSTRUCTION_HOLD_SECONDS * 2 + INSTRUCTION_TAIL_SECONDS
    start = time.monotonic()

    def _log_event(event) -> None:
        if event.kind == "item_resolve_success":
            elapsed = time.monotonic() - start
            print(f"[instruction] resolved item {event.index} at {elapsed:.1f}s")

    hooks = ls.StreamHooks(on_event=_log_event)
    stream = ls.astream_raw(
        _instruction_generator(),
        model=model,
        chunk_seconds=CHUNK_SECONDS,
        pattern_seconds=PATTERN_SECONDS,
        transition_seconds=TRANSITION_SECONDS,
        fallback="keep_last",
        fallback_model="fast",
        hooks=hooks,
    )
    async_iter = stream.__aiter__()
    try:
        async for chunk in async_iter:
            chunks.append(chunk)
            if len(chunks) * CHUNK_SECONDS >= total_seconds:
                break
    finally:
        aclose = getattr(async_iter, "aclose", None)
        if aclose is not None:
            await aclose()
    return chunks


def _instruction_generator_example(label: str, model: ModelSpec) -> None:
    total_seconds = INSTRUCTION_HOLD_SECONDS * 2 + INSTRUCTION_TAIL_SECONDS
    print(
        "[instruction] sleep controls duration; last item keeps playing until cut "
        f"(hold={INSTRUCTION_HOLD_SECONDS}s tail={INSTRUCTION_TAIL_SECONDS}s total={total_seconds}s)"
    )
    chunks = asyncio.run(_collect_instruction_chunks(model))
    _play_if_enabled(chunks, label=f"instruction:{label}")


def _streamable_generator_example(label: str, model: ModelSpec) -> None:
    print(f"[streamable] duration controls length (duration={STREAMABLE_DURATION_SECONDS}s)")

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

    chunks = list(
        ls.stream_raw(
            _gen(),
            model=model,
            chunk_seconds=CHUNK_SECONDS,
            pattern_seconds=PATTERN_SECONDS,
            transition_seconds=TRANSITION_SECONDS,
            fallback="keep_last",
            fallback_model="fast",
        )
    )
    _play_if_enabled(chunks, label=f"streamable:{label}")
    if not PLAY_AUDIO:
        print(f"[streamable] {label}: {len(chunks)} chunks")


def _run_suite(label: str, model: ModelSpec) -> None:
    try:
        _render_test(label, model)
        _stream_test(label, model)
        _update_test(label, model)
        if RUN_INSTRUCTION_GENERATOR_EXAMPLE:
            _instruction_generator_example(label, model)
        if RUN_STREAMABLE_GENERATOR_EXAMPLE:
            _streamable_generator_example(label, model)
    except PlaybackError as exc:
        print(f"[fail] {label}: playback unavailable ({exc})")
    except ModelNotAvailableError as exc:
        print(f"[fail] {label}: model unavailable ({exc})")


def main() -> None:
    if RUN_FAST_MODEL:
        _run_suite("fast", "fast")

    if RUN_EXPRESSIVE_MODEL:
        _run_suite("expressive", EXPRESSIVE_MODEL)

    if RUN_GPT52_MODEL and _guard_model("gpt-5.2", ["OPENAI_API_KEY"]):
        _run_suite("gpt-5.2", _external(GPT52_MODEL))

    if RUN_SONNET_MODEL and _guard_model("claude-sonnet-4.5", ["ANTHROPIC_API_KEY"]):
        _run_suite("claude-sonnet-4.5", _external(SONNET_MODEL))


if __name__ == "__main__":
    main()
