from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from latentscore import (
    FirstAudioSpinner,
    MusicConfig,
    MusicConfigUpdate,
    Playlist,
    Track,
    render,
    save_wav,
    stream,
)
from latentscore.providers.litellm import LiteLLMAdapter

ASTREAM_SECONDS = 120.0
ASTREAM_CHUNK_SECONDS = 2.0
ASTREAM_TRANSITION_SECONDS = 1.0
RENDER_SECONDS = 120.0
STREAM_SECONDS = 120.0
STREAM_CHUNK_SECONDS = 1.0
STREAM_TRANSITION_SECONDS = 1.0

GEMINI_MODEL = "external:gemini/gemini-3-flash-preview"
GEMINI_API_ENV = "GEMINI_API_KEY"
ANTHROPIC_MODEL = "external:claude-opus-4-5-20251101"  # claude-sonnet-4-5-20250929
ANTHROPIC_API_ENV = "ANTHROPIC_API_KEY"
DOTENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_api_key(api_env: str) -> str | None:
    load_dotenv(DOTENV_PATH)
    api_key = os.getenv(api_env, "").strip()
    return api_key or None


def _assert_audio(audio: np.ndarray) -> None:
    array = np.asarray(audio)
    assert array.dtype == np.float32
    assert array.ndim == 1
    assert float(np.max(np.abs(array))) <= 1.0


async def _write_astream(path: Path, model: LiteLLMAdapter) -> None:
    segment_seconds = ASTREAM_SECONDS / 3
    tracks = [
        Track(
            content=MusicConfig(tempo="slow", brightness="dark", root="d"),
            duration=segment_seconds,
            transition=ASTREAM_TRANSITION_SECONDS,
        ),
        Track(
            content=MusicConfig(tempo="medium", brightness="medium", root="f"),
            duration=segment_seconds,
            transition=ASTREAM_TRANSITION_SECONDS,
        ),
        Track(
            content=MusicConfig(tempo="fast", brightness="bright", root="c"),
            duration=segment_seconds,
            transition=ASTREAM_TRANSITION_SECONDS,
        ),
    ]
    playlist = Playlist(tracks=tracks)
    chunks = [
        chunk
        async for chunk in playlist.stream(
            chunk_seconds=ASTREAM_CHUNK_SECONDS,
            model=model,
        )
    ]
    save_wav(str(path), chunks)


def main() -> None:
    api_key = _load_api_key(GEMINI_API_ENV)
    adapter = LiteLLMAdapter(
        model=GEMINI_MODEL,
        api_key=api_key,
        litellm_kwargs={"timeout": 60},
    )
    output_dir = Path(__file__).resolve().parent / ".outputs"
    output_dir.mkdir(exist_ok=True)
    spinner = FirstAudioSpinner(delay=0.35)

    try:
        if True:
            audio = render("I AM INSANE", duration=RENDER_SECONDS, model=adapter)
            _assert_audio(np.asarray(audio))
            render_path = output_dir / "render.wav"
            audio.save(render_path)

        if False:
            gemini_key = _load_api_key(GEMINI_API_ENV)
            _ = LiteLLMAdapter(
                model=GEMINI_MODEL,
                api_key=gemini_key,
                litellm_kwargs={"timeout": 60},
            )

        stream_path = output_dir / "stream_transition.wav"
        stream(
            "warm sunrise",
            "mario the video game",
            MusicConfigUpdate(tempo="slow", brightness="dark"),
            duration=STREAM_SECONDS,
            transition=STREAM_TRANSITION_SECONDS,
            chunk_seconds=STREAM_CHUNK_SECONDS,
            model=adapter,
            prefetch_depth=1,
            preview_policy="embedding",
            hooks=spinner.hooks(),
        ).save(stream_path)
    finally:
        # Ensure LiteLLM async clients close before the loop is torn down.
        adapter.close()

    # asyncio.run(_write_astream(output_dir / "astream_tween.wav", adapter))

    # print("Latentscore API smoke OK")
    # print(f"Wrote: {render_path}")
    # print(f"Wrote: {stream_path}")
    # print("Wrote: astream_tween.wav")
    # print(f"Sample rate: {SAMPLE_RATE}")


if __name__ == "__main__":
    from latentscore.logging_utils import log_exception
    from latentscore.spinner import render_error

    try:
        main()
    except Exception as exc:
        log_exception("smoke example", exc)
        render_error("smoke example", exc)
        raise SystemExit(1) from exc
