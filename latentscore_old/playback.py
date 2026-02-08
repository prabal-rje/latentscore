from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .audio import FloatArray, ensure_audio_contract
from .errors import PlaybackError
from .spinner import ProgressBar, Spinner

_LOGGER = logging.getLogger("latentscore.playback")
_MUSIC_FRAMES = "\u266a\u266b\u266c\u2669"


class PlaybackBackend(BaseModel):
    name: str
    play_audio: Callable[[FloatArray, int], None]
    play_stream: Callable[[Iterable[FloatArray], int], None]

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


def _load_backend() -> PlaybackBackend | None:
    return _load_sounddevice() or _load_simpleaudio() or _load_ipython()


def _resolve_backend() -> PlaybackBackend:
    backend = _load_backend()
    if backend is None:
        raise PlaybackError(
            "Playback requires sounddevice or simpleaudio. "
            "Install one of them (or use .save()); for notebook playback install ipython."
        )
    return backend


def play_audio(samples: FloatArray, *, sample_rate: int) -> None:
    backend = _resolve_backend()
    duration = len(samples) / sample_rate if sample_rate > 0 else 0.0

    def _run() -> None:
        backend.play_audio(samples, sample_rate)

    _play_with_progress(_run, duration=duration, message="♪ Playing audio ... ")


def play_stream(chunks: Iterable[FloatArray], *, sample_rate: int) -> None:
    backend = _resolve_backend()

    def _run() -> None:
        backend.play_stream(chunks, sample_rate)

    _play_with_spinner(_run, message="Playing stream", frames=_MUSIC_FRAMES)


def _play_with_progress(
    play_fn: Callable[[], None],
    *,
    duration: float,
    message: str,
) -> None:
    progress = ProgressBar(message, total=max(duration, 0.0))
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            play_fn()
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    start = time.monotonic()
    progress.start()
    try:
        while thread.is_alive():
            elapsed = time.monotonic() - start
            progress.update(elapsed)
            time.sleep(0.1)
    finally:
        thread.join(timeout=0.2)
        progress.update(duration)
        progress.stop()
    if error:
        raise error[0]


def _play_with_spinner(
    play_fn: Callable[[], None],
    *,
    message: str,
    frames: str,
) -> None:
    spinner = Spinner(
        f"{message} {_MUSIC_FRAMES[0]}",
        frames=frames,
        spinner="dots",
    )
    spinner.start()
    try:
        play_fn()
    finally:
        spinner.stop()


def _load_sounddevice() -> PlaybackBackend | None:
    try:
        import sounddevice as sd_module  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.info("sounddevice not available: %s", exc, exc_info=True)
        return None
    sd: Any = sd_module

    def _play_audio(samples: FloatArray, sample_rate: int) -> None:
        normalized = ensure_audio_contract(samples, sample_rate=sample_rate)
        sd.play(normalized, sample_rate)
        sd.wait()

    def _play_stream(chunks: Iterable[FloatArray], sample_rate: int) -> None:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        ) as stream:
            for chunk in chunks:
                normalized = ensure_audio_contract(chunk, sample_rate=sample_rate)
                stream.write(normalized.reshape(-1, 1))

    return PlaybackBackend(
        name="sounddevice",
        play_audio=_play_audio,
        play_stream=_play_stream,
    )


def _load_simpleaudio() -> PlaybackBackend | None:
    try:
        import simpleaudio as sa_module  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.info("simpleaudio not available: %s", exc, exc_info=True)
        return None
    sa: Any = sa_module

    def _to_int16(samples: FloatArray, sample_rate: int) -> NDArray[np.int16]:
        normalized = ensure_audio_contract(samples, sample_rate=sample_rate)
        clipped = np.clip(normalized, -1.0, 1.0)
        return (clipped * 32_767).astype(np.int16)

    def _play_audio(samples: FloatArray, sample_rate: int) -> None:
        audio = _to_int16(samples, sample_rate)
        play = sa.play_buffer(audio, 1, 2, sample_rate)
        play.wait_done()

    def _play_stream(chunks: Iterable[FloatArray], sample_rate: int) -> None:
        buffered = [ensure_audio_contract(chunk, sample_rate=sample_rate) for chunk in chunks]
        if not buffered:
            return
        _play_audio(np.concatenate(buffered), sample_rate)

    return PlaybackBackend(
        name="simpleaudio",
        play_audio=_play_audio,
        play_stream=_play_stream,
    )


def _load_ipython() -> PlaybackBackend | None:
    try:
        from IPython import get_ipython  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.info("IPython not available: %s", exc, exc_info=True)
        return None
    if get_ipython() is None:
        _LOGGER.info("IPython shell not active; skipping IPython playback backend.")
        return None
    try:
        import IPython.display as ipy_display  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.info("IPython not available: %s", exc, exc_info=True)
        return None
    ipy_display_any: Any = ipy_display
    ipy_audio = ipy_display_any.Audio
    display_fn = ipy_display_any.display

    def _play_audio(samples: FloatArray, sample_rate: int) -> None:
        normalized = ensure_audio_contract(samples, sample_rate=sample_rate)
        display_fn(ipy_audio(normalized, rate=sample_rate))

    def _play_stream(chunks: Iterable[FloatArray], sample_rate: int) -> None:
        buffered = [ensure_audio_contract(chunk, sample_rate=sample_rate) for chunk in chunks]
        if not buffered:
            return
        _play_audio(np.concatenate(buffered), sample_rate)

    return PlaybackBackend(
        name="ipython",
        play_audio=_play_audio,
        play_stream=_play_stream,
    )
