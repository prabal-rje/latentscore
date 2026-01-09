from __future__ import annotations

import argparse
import functools
import json
import os
import re
from pathlib import Path
from typing import Iterable, NamedTuple

import latentscore as ls
from latentscore.config import MusicConfig
from latentscore.playback import play_stream
from latentscore.prompt_examples import FEW_SHOT_EXAMPLES as TRACK_EXAMPLES_TEXT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".examples"
# EXTERNAL_MODEL = "external:gemini/gemini-3-flash-preview"
# EXTERNAL_API_ENV = "GEMINI_API_KEY"
EXTERNAL_MODEL = "external:anthropic/claude-opus-4-5-20251101"
EXTERNAL_API_ENV = "ANTHROPIC_API_KEY"
# EXTERNAL_MODEL = "external:openrouter/mistralai/voxtral-small-24b-2507"
# EXTERNAL_API_ENV = "OPENROUTER_API_KEY"
TRACK_DURATION = 12.0

_TRACK_EXAMPLE_PATTERN = re.compile(
    r"\*\*Example\s+(?P<num>\d+)\*\*\s*Input:\s*\"(?P<input>.*?)\"\s*Output:\s*\n\n```json\n(?P<json>.*?)\n```",
    re.DOTALL,
)


class TrackExample(NamedTuple):
    index: int
    key: str
    label: str
    input_text: str
    config: MusicConfig


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return normalized.strip("-")


def _parse_track_config(payload: str) -> MusicConfig:
    data = json.loads(payload)
    match data:
        case {"config": config} if isinstance(config, dict):
            return MusicConfig.model_validate(config)
        case _:
            raise ValueError("Track example missing config payload.")


@functools.lru_cache(maxsize=1)
def _track_examples() -> tuple[TrackExample, ...]:
    examples: list[TrackExample] = []
    for match in _TRACK_EXAMPLE_PATTERN.finditer(TRACK_EXAMPLES_TEXT):
        index = int(match.group("num"))
        input_text = match.group("input").strip()
        label = input_text.split(" - ", 1)[0].strip()
        key = _slugify(label) or f"example-{index}"
        payload = match.group("json").strip()
        config = _parse_track_config(payload)
        examples.append(
            TrackExample(
                index=index,
                key=key,
                label=label,
                input_text=input_text,
                config=config,
            )
        )
    return tuple(examples)


def _track_index() -> dict[str, TrackExample]:
    index: dict[str, TrackExample] = {}
    for example in _track_examples():
        keys = {
            example.key,
            example.label.lower(),
            example.input_text.lower(),
            str(example.index),
            f"example-{example.index}",
        }
        for key in keys:
            if key and key not in index:
                index[key] = example
    return index


def _resolve_track_example(track: str) -> TrackExample | None:
    query = track.strip()
    if not query:
        return None
    index = _track_index()
    normalized = query.lower()
    for key in (normalized, _slugify(query)):
        if key in index:
            return index[key]
    return None


def _print_track_examples() -> None:
    examples = _track_examples()
    if not examples:
        print("No track examples available.")
        return
    print("Available track examples:")
    for example in examples:
        print(f"- example-{example.index}: {example.label} ({example.key})")


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
    print("HELLO CHAD THIS IS HERE!")
    audio = ls.render(
        # "glass elevator through clouds",
        # "8-bit mario the video game theme song",
        # "Highly Cyberpunk style music",
        "vinyl crackle midnight jazz",
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


def _demo_track_example(example: TrackExample, *, save: bool) -> None:
    audio = ls.render(
        example.input_text,
        duration=TRACK_DURATION,
        config=example.config,
    )
    audio.play()
    if save:
        audio.save(OUTPUT_DIR / f"demo_track_{example.key}.wav")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="latentscore.demo")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Run the live generator stream.")
    mode_group.add_argument(
        "--external",
        action="store_true",
        help="Run external LLM demos.",
    )
    mode_group.add_argument(
        "--track",
        type=str,
        default=None,
        help="Render a named few-shot track using LiteLLM (see --list-tracks).",
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
        "--list-tracks",
        action="store_true",
        help="List available track names for --track.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Save demo audio into {OUTPUT_DIR}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.list_tracks:
        _print_track_examples()
        return
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
        return

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
        return

    if args.track:
        example = _resolve_track_example(args.track)
        if example is None:
            print(f"Unknown track: {args.track}")
            _print_track_examples()
            return
        _demo_track_example(example, save=args.save)
        return

    print("No mode selected. Use --live, --external, or --track.")


if __name__ == "__main__":
    main()
