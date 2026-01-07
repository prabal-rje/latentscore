from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray

from .errors import InvalidConfigError

FloatArray = NDArray[np.float32]
AudioNumbers = NDArray[np.floating[Any]] | Sequence[float] | FloatArray

SAMPLE_RATE = 44_100


def ensure_audio_contract(
    audio: AudioNumbers,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> FloatArray:
    """Normalize dtype/range/shape to the audio contract."""

    _ = sample_rate
    mono: FloatArray = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        mono = mono / peak
    return mono


def iter_chunks(chunks: Iterable[AudioNumbers]) -> Iterator[FloatArray]:
    """Yield chunks that already respect the audio contract."""

    for chunk in chunks:
        yield ensure_audio_contract(chunk)


def _looks_like_samples(sequence: Sequence[object]) -> bool:
    match sequence:
        case []:
            return True
        case [int() | float() | np.floating(), *_]:
            return True
        case _:
            return False


def write_wav(
    path: str | Path,
    audio_or_chunks: AudioNumbers | Iterable[AudioNumbers],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Write a full array or chunk iterator to a wav file."""

    target = Path(path)
    audio_obj: object = audio_or_chunks
    match audio_obj:
        case np.ndarray():
            write_fn = getattr(sf, "write", None)
            assert callable(write_fn)
            write_audio = cast(Callable[[Path | str, AudioNumbers, int], None], write_fn)
            array_float: NDArray[np.float32] = np.asarray(audio_obj, dtype=np.float32)
            normalized = ensure_audio_contract(array_float, sample_rate=sample_rate)
            # soundfile stubs are incomplete; cast is intentional for type safety.
            write_audio(target, normalized, sample_rate)  # type: ignore[reportUnknownMemberType]
            return target
        case Sequence() as sequence if _looks_like_samples(sequence):
            write_fn = getattr(sf, "write", None)
            assert callable(write_fn)
            write_audio = cast(Callable[[Path | str, AudioNumbers, int], None], write_fn)
            assert _looks_like_samples(sequence)
            sequence_array: NDArray[np.float32] = np.asarray(sequence, dtype=np.float32)
            normalized = ensure_audio_contract(sequence_array, sample_rate=sample_rate)
            write_audio(target, normalized, sample_rate)  # type: ignore[reportUnknownMemberType]
            return target
        case str() | bytes():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("audio_or_chunks must be audio samples or chunk iterables")
        case Iterable() as chunks:
            match chunks:
                case str() | bytes():  # type: ignore[reportUnnecessaryComparison]
                    raise InvalidConfigError(
                        "audio_or_chunks must be audio samples or chunk iterables"
                    )
                case _:
                    pass
        case _:
            raise InvalidConfigError("audio_or_chunks must be audio samples or chunk iterables")

    with sf.SoundFile(
        target,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="FLOAT",
    ) as handle:
        write_fn = getattr(handle, "write", None)
        assert callable(write_fn)
        write_chunk = cast(Callable[[FloatArray], None], write_fn)  # type: ignore[reportUnknownMemberType]
        assert isinstance(chunks, Iterable)
        chunks_iter = cast(Iterable[AudioNumbers], chunks)
        for chunk in iter_chunks(chunks_iter):
            write_chunk(chunk)  # type: ignore[reportUnknownMemberType]

    return target
