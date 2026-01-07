from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from dotenv import load_dotenv

from latentscore import (
    SAMPLE_RATE,
    MusicConfig,
    Streamable,
    astream,
    render,
    save_wav,
    MusicConfigUpdate,
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

GEMINI_MODEL = "gemini/gemini-3-flash-preview" 
GEMINI_API_ENV = "GEMINI_API_KEY"
ANTHROPIC_MODEL = "claude-opus-4-5-20251101" #claude-sonnet-4-5-20250929
ANTHROPIC_API_ENV = "ANTHROPIC_API_KEY"
DOTENV_PATH = Path(__file__).resolve().parent / ".env"

def _load_api_key(api_env: str) -> str:
    load_dotenv(DOTENV_PATH)
    api_key = os.getenv(api_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {api_env} in {DOTENV_PATH}")
    return api_key

def _assert_audio(audio: np.ndarray) -> None:
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert float(np.max(np.abs(audio))) <= 1.0


async def _write_astream(path: Path, model: LiteLLMAdapter) -> None:
    configs = [
        MusicConfig(tempo="slow", brightness="dark", root="d"),
        MusicConfig(tempo="medium", brightness="medium", root="f"),
        MusicConfig(tempo="fast", brightness="bright", root="c"),
    ]
    segment_seconds = ASTREAM_SECONDS / len(configs)

    async def inputs() -> AsyncIterator[Streamable]:
        for config in configs:
            yield Streamable(
                content=config,
                duration=segment_seconds,
                transition_duration=ASTREAM_TRANSITION_SECONDS,
            )

    chunks = [
        chunk
        async for chunk in astream(
            inputs(),
            chunk_seconds=ASTREAM_CHUNK_SECONDS,
            model=model,
            config=configs[0],
        )
    ]
    save_wav(str(path), chunks)


def main() -> None:
    api_key = _load_api_key(ANTHROPIC_API_ENV)
    adapter = LiteLLMAdapter(model=ANTHROPIC_MODEL, api_key=api_key)
    output_dir = Path(__file__).resolve().parent / ".outputs"
    output_dir.mkdir(exist_ok=True)

    try:
        audio = render("I AM INSANE", duration=RENDER_SECONDS, model=adapter)
        _assert_audio(audio)
        render_path = output_dir / "render.wav"
        save_wav(str(render_path), audio)

        segment_seconds = STREAM_SECONDS / 3
        stream_items = [
            Streamable(
                content="mario the video game",
                duration=segment_seconds,
                transition_duration=STREAM_TRANSITION_SECONDS,
            ),
            # Streamable(
            #     content=MusicConfig(tempo="fast", brightness="bright", root="c"),
            #     duration=segment_seconds,
            #     transition_duration=STREAM_TRANSITION_SECONDS,
            # ),
            # Streamable(
            #     content=MusicConfigUpdate(tempo="slow", brightness="dark"),
            #     duration=segment_seconds,
            #     transition_duration=STREAM_TRANSITION_SECONDS,
            # ),
            # Streamable(
            #     content=MusicConfig(tempo="fast", brightness="bright", root="c"),
            #     duration=segment_seconds,
            #     transition_duration=STREAM_TRANSITION_SECONDS,
            # ),
        ]
        stream_path = output_dir / "stream_transition.wav"
        save_wav(
            str(stream_path),
            stream(
                stream_items,
                chunk_seconds=STREAM_CHUNK_SECONDS,
                model=adapter,
            ),
        )
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
    main()
