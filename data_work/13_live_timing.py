"""
Live streaming timing benchmark — event-logged example.

Measures real-time resolve latency for different model backends
(LLM vs embed lookup) through the live() streaming API.

Run:
    python -m data_work.13_live_timing              # play to speakers (Gemini)
    python -m data_work.13_live_timing --save out.wav  # save to file instead
    python -m data_work.13_live_timing --model fast    # use built-in embed lookup
"""

import argparse
import asyncio
import time
from collections.abc import AsyncIterator

import dotenv

import latentscore as ls
from latentscore.config import Step
from latentscore.models import MODEL_CHOICES

dotenv.load_dotenv()  # load .env (GEMINI_API_KEY, etc.)


# ── timing logger ────────────────────────────────────────────────────
_clock = {"t0": 0.0}


def _ts() -> str:
    """Wall-clock seconds since stream start."""
    return f"{time.monotonic() - _clock['t0']:6.2f}s"


def log_event(event: ls.StreamEvent) -> None:
    """Single hook that logs every event with a timestamp."""
    match event.kind:
        case "stream_start":
            print(f"[{_ts()}]  stream started")

        case "model_load_start":
            print(f"[{_ts()}]  loading model ({event.model_role}) ...")

        case "model_load_end":
            print(f"[{_ts()}]  model ready ({event.model_role})")

        case "item_resolve_start":
            label = _item_label(event.item)
            print(f"[{_ts()}]  resolve #{event.index} START  {label}")

        case "item_resolve_success":
            label = _item_label(event.item)
            print(f"[{_ts()}]  resolve #{event.index} DONE   {label}")

        case "item_resolve_error":
            print(f"[{_ts()}]  resolve #{event.index} ERROR  {event.error}")

        case "first_config_ready":
            print(f"[{_ts()}]  first config ready (audio can begin)")

        case "first_audio_chunk":
            print(f"[{_ts()}]  first audio chunk rendered")

        case "fallback_used":
            print(f"[{_ts()}]  fallback triggered for #{event.index}")

        case "stream_end":
            print(f"[{_ts()}]  stream ended")

        case "error":
            print(f"[{_ts()}]  ERROR: {event.error}")


def _item_label(item: object) -> str:
    """Short human-readable label for a stream item."""
    match item:
        case str():
            return f'"{item[:40]}..."' if len(item) > 40 else f'"{item}"'
        case ls.MusicConfigUpdate():
            fields = {
                k: v for k, v in item.model_dump(exclude_none=True).items()
            }
            return f"MCU({fields})"
        case _:
            return repr(item)[:60]


# ── generator (the "performance") ────────────────────────────────────

async def performance() -> AsyncIterator[str | ls.MusicConfigUpdate]:
    """
    Async generator that drives the live session.

    Yields vibes and config updates; sleeps control how long each
    segment plays before the next yield.
    """
    print(f"[{_ts()}]  yield: \"warm jazz cafe at midnight\"")
    yield "warm jazz cafe at midnight"

    await asyncio.sleep(8)

    update = ls.MusicConfigUpdate(brightness=Step(-2), echo="heavy")
    print(f"[{_ts()}]  yield: MCU(brightness=Step(-2), echo='heavy')")
    yield update

    await asyncio.sleep(8)

    print(f"[{_ts()}]  yield: \"neon rain on empty streets\"")
    yield "neon rain on empty streets"


# ── main ─────────────────────────────────────────────────────────────

def _parse_model(model_str: str) -> ls.ModelSpec:
    """Parse CLI model string into a ModelSpec."""
    if model_str in MODEL_CHOICES:
        return model_str  # type: ignore[return-value]  # built-in ModelChoice
    return ls.ExternalModelSpec(model=model_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save to WAV instead of playing to speakers",
    )
    parser.add_argument(
        "--model", type=str, default="gemini/gemini-3-flash-preview",
        help="LiteLLM model string or built-in name (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--seconds", type=float, default=60.0,
        help="Total playback duration (default: 60)",
    )
    args = parser.parse_args()

    _clock["t0"] = time.monotonic()

    model = _parse_model(args.model)
    hooks = ls.StreamHooks(on_event=log_event)

    print(f"[{_ts()}]  creating live session (model={args.model})")

    session = ls.live(
        performance(),
        model=model,
        transition_seconds=2.0,
        hooks=hooks,
        queue_maxsize=1,
    )

    if args.save:
        path = session.save(args.save, seconds=args.seconds)
        print(f"\n[{_ts()}]  saved to {path}")
    else:
        print(f"[{_ts()}]  playing for {args.seconds}s ...")
        session.play(seconds=args.seconds)
        print(f"\n[{_ts()}]  done")


if __name__ == "__main__":
    main()
