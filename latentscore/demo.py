from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import latentscore as ls
from latentscore.playback import play_stream

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".examples"
EXTERNAL_MODEL = "external:gemini/gemini-3-flash-preview"
EXTERNAL_API_ENV = "GEMINI_API_KEY"


def _live_playlist() -> Iterable[ls.Streamable]:
    vibes = [
        "misty harbor at dawn",
        "neon skyline rain",
        "quiet orchard dusk",
        "starlit highway",
    ]
    for vibe in vibes:
        yield ls.Streamable(
            content=vibe,
            duration=6.0,
            transition_duration=1.5,
        )


def _demo_live_generator() -> None:
    live_chunks = ls.stream_raw(
        _live_playlist(),
        chunk_seconds=1.0,
        model="fast",
    )
    play_stream(live_chunks, sample_rate=ls.SAMPLE_RATE)


def _demo_external_with_key() -> None:
    api_key = os.environ.get(EXTERNAL_API_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {EXTERNAL_API_ENV} to run external LLM demo.")
    audio = ls.render(
        "glass elevator through clouds",
        duration=6.0,
        model=ls.ExternalModelSpec(model=EXTERNAL_MODEL, api_key=api_key),
    )
    audio.save(OUTPUT_DIR / "demo_external_key.wav")


def _demo_external_with_env() -> None:
    api_key = os.environ.get(EXTERNAL_API_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {EXTERNAL_API_ENV} to run external LLM demo.")
    os.environ[EXTERNAL_API_ENV] = api_key
    audio = ls.render(
        "vinyl crackle midnight jazz",
        duration=6.0,
        model=ls.ExternalModelSpec(model=EXTERNAL_MODEL, api_key=None),
    )
    audio.save(OUTPUT_DIR / "demo_external_env.wav")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio = ls.render("warm sunrise over water", duration=8.0)
    render_path = OUTPUT_DIR / "demo_render.wav"
    audio.save(render_path)

    stream = ls.stream(
        "late night neon",
        "after hours",
        duration=12.0,
        transition=3.0,
        chunk_seconds=1.0,
    )
    stream_path = OUTPUT_DIR / "demo_stream.wav"
    stream.play()
    stream.save(stream_path)

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

    # Set LATENTSCORE_DEMO_LIVE=1 to run the live generator stream.
    if os.environ.get("LATENTSCORE_DEMO_LIVE"):
        _demo_live_generator()

    if os.environ.get("LATENTSCORE_DEMO_EXTERNAL"):
        _demo_external_with_key()
        _demo_external_with_env()

    # External LLM examples (require network + API key).
    # Set LATENTSCORE_DEMO_EXTERNAL=1 to run.


if __name__ == "__main__":
    main()
