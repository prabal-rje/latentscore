from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import latentscore as ls
from latentscore.playback import play_stream

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".examples"
# EXTERNAL_MODEL = "external:gemini/gemini-3-flash-preview"
# EXTERNAL_API_ENV = "GEMINI_API_KEY"
EXTERNAL_MODEL = "external:anthropic/claude-opus-4-5-20251101"
EXTERNAL_API_ENV = "ANTHROPIC_API_KEY"
# EXTERNAL_MODEL = "external:openrouter/mistralai/voxtral-small-24b-2507"
# EXTERNAL_API_ENV = "OPENROUTER_API_KEY"


def _live_playlist_generator() -> Iterable[ls.Streamable]:
    DURATION, TRANSITION_DURATION = 6.0, 1.5
    yield ls.Streamable(
        content="misty harbor at dawn",
        duration=DURATION,
        transition_duration=TRANSITION_DURATION,
    )
    yield ls.Streamable(
        content="neon skyline rain",
        duration=DURATION,
        transition_duration=TRANSITION_DURATION,
    )
    yield ls.Streamable(
        content="quiet orchard dusk",
        duration=DURATION,
        transition_duration=TRANSITION_DURATION,
    )
    yield ls.Streamable(
        content="starlit highway",
        duration=DURATION,
        transition_duration=TRANSITION_DURATION,
    )


def _demo_live_from_generator() -> None:
    live_chunks = ls.stream_raw(
        _live_playlist_generator(),
        chunk_seconds=1.0,
        model="fast",
    )
    play_stream(live_chunks, sample_rate=ls.SAMPLE_RATE)


def _demo_external_with_key(model: str, api_key: str, *, save: bool) -> None:
    audio = ls.render(
        #"glass elevator through clouds",
        #"8-bit mario the video game theme song",
        "Cyberpunk rainy night in tokyo",
        duration=30.0,
        model=ls.ExternalModelSpec(model=model, api_key=api_key),
    )
    audio.play()
    if save:
        audio.save(OUTPUT_DIR / "demo_external_key.wav")


def _demo_external_with_env(model: str, api_env: str, *, save: bool) -> bool:
    api_key = os.environ.get(api_env, "").strip()
    if not api_key:
        return False
    os.environ[api_env] = api_key
    audio = ls.render(
        "vinyl crackle midnight jazz",
        duration=6.0,
        model=ls.ExternalModelSpec(model=model, api_key=None),
    )
    audio.play()
    if save:
        audio.save(OUTPUT_DIR / "demo_external_env.wav")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="latentscore.demo")
    parser.add_argument("--live", action="store_true", help="Run the live generator stream.")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run external LLM demos.",
    )
    parser.add_argument(
        "--external-model",
        type=str,
        default=EXTERNAL_MODEL,
        help=f"External model name (default: {EXTERNAL_MODEL}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the explicit external LLM demo.",
    )
    parser.add_argument(
        "--api-env",
        type=str,
        default=EXTERNAL_API_ENV,
        help=f"Environment variable name for API key (default: {EXTERNAL_API_ENV}).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Save demo audio into {OUTPUT_DIR}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # audio = ls.render("warm sunrise over water", duration=8.0)
    # render_path = OUTPUT_DIR / "demo_render.wav"
    # audio.save(render_path)

    # stream = ls.stream(
    #     "late night neon",
    #     "after hours",
    #     duration=12.0,
    #     transition=3.0,
    #     chunk_seconds=1.0,
    # )
    # stream_path = OUTPUT_DIR / "demo_stream.wav"
    # stream.play()
    # stream.save(stream_path)

    # playlist = ls.Playlist(
    #     tracks=(
    #         ls.Track(content="foggy dock", duration=6.0, transition=2.0),
    #         ls.Track(
    #             content=ls.MusicConfig(tempo="fast", brightness="bright"),
    #             duration=6.0,
    #             transition=2.0,
    #         ),
    #         ls.Track(
    #             content=ls.MusicConfigUpdate(tempo="slow", space="vast"),
    #             duration=6.0,
    #             transition=2.0,
    #         ),
    #     )
    # )
    # playlist_path = OUTPUT_DIR / "demo_playlist.wav"
    # playlist.render(chunk_seconds=1.0).save(playlist_path)

    # print("Saved demo outputs:")
    # print(f"- {render_path}")
    # print(f"- {stream_path}")
    # print(f"- {playlist_path}")

    if args.live:
        _demo_live_from_generator()

    if args.external:
        ran = False
        api_key = (
            args.api_key.strip() if args.api_key else os.environ.get(args.api_env, "").strip() or ""
        )
        if api_key:
            _demo_external_with_key(args.external_model, api_key, save=args.save)
            ran = True

        if not api_key:
            if _demo_external_with_env(args.external_model, args.api_env, save=args.save):
                ran = True

        if not ran:
            print(
                f"No external demos executed. Provide --api-key your_key_here or set {args.api_env} to run them."
            )


if __name__ == "__main__":
    main()
