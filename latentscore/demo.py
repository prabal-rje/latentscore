from __future__ import annotations

from pathlib import Path

import latentscore as ls

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".examples"


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
    stream.save(stream_path)

    playlist = ls.Playlist(
        tracks=(
            ls.Track(content="foggy dock", duration=6.0, transition=2.0),
            ls.Track(
                content=ls.MusicConfig(tempo="fast", brightness="bright"),
                duration=6.0,
                transition=2.0,
            ),
            ls.Track(
                content=ls.MusicConfigUpdate(tempo="slow", space="vast"),
                duration=6.0,
                transition=2.0,
            ),
        )
    )
    playlist_path = OUTPUT_DIR / "demo_playlist.wav"
    playlist.render(chunk_seconds=1.0).save(playlist_path)

    print("Saved demo outputs:")
    print(f"- {render_path}")
    print(f"- {stream_path}")
    print(f"- {playlist_path}")


if __name__ == "__main__":
    main()
