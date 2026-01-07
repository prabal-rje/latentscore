This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py, **/*.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
providers/
  __init__.py
  litellm.py
__init__.py
audio.py
cli.py
config.py
errors.py
main.py
models.py
synth.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="providers/__init__.py">
from __future__ import annotations

from .litellm import LiteLLMAdapter

__all__ = ["LiteLLMAdapter"]
</file>

<file path="providers/litellm.py">
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Literal

from pydantic import BaseModel, ValidationError

from ..config import MusicConfig
from ..errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError
from ..models import build_expressive_prompt

_DEFAULT_TEMPERATURE = 0.0
_atexit_guard_registered = False
_safe_get_event_loop_installed = False
_LOGGER = logging.getLogger("latentscore.providers.litellm")
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]


def _safe_async_cleanup(
    cleanup_coro: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _LOGGER.info("LiteLLM cleanup running outside an event loop.")
        loop = None

    if loop is not None:
        try:
            loop.create_task(cleanup_coro())
        except Exception as exc:
            _LOGGER.warning("LiteLLM async cleanup task failed: %s", exc, exc_info=True)
        return

    try:
        asyncio.run(cleanup_coro())
    except Exception as exc:
        _LOGGER.warning("LiteLLM async cleanup failed: %s", exc, exc_info=True)


def _install_safe_get_event_loop() -> None:
    global _safe_get_event_loop_installed
    if _safe_get_event_loop_installed:
        return
    _safe_get_event_loop_installed = True

    original_get_event_loop = asyncio.get_event_loop

    def safe_get_event_loop() -> asyncio.AbstractEventLoop:
        try:
            loop = original_get_event_loop()
        except RuntimeError:
            _LOGGER.info("LiteLLM cleanup creating a new event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        if loop.is_closed():
            _LOGGER.info("LiteLLM cleanup replacing a closed event loop.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    asyncio.get_event_loop = safe_get_event_loop  # type: ignore[assignment]

    try:
        from litellm.llms.custom_httpx.async_client_cleanup import (
            close_litellm_async_clients,
        )
    except Exception as exc:
        _LOGGER.info("LiteLLM cleanup import unavailable: %s", exc)
        return
    _safe_async_cleanup(close_litellm_async_clients)


def _register_safe_get_event_loop_atexit() -> None:
    global _atexit_guard_registered
    if _atexit_guard_registered:
        return
    _atexit_guard_registered = True
    import atexit

    atexit.register(_install_safe_get_event_loop)


def _extract_json_payload(content: str) -> str | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1]


def _content_snippet(content: str, limit: int = 200) -> str:
    cleaned = content.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


class _LiteLLMRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    response_format: type[BaseModel] | None = None
    api_key: str | None = None


class LiteLLMAdapter:
    """LiteLLM model wrapper implementing the async model protocol."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = MusicConfig,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._base_prompt = build_expressive_prompt()
        self._response_format: type[BaseModel] | None = response_format
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOGGER.info("LiteLLM close running outside event loop.")
            asyncio.run(self.aclose())
            return
        loop.create_task(self.aclose())

    async def aclose(self) -> None:
        try:
            import litellm  # type: ignore[import]  # Optional dependency at runtime.
        except ImportError as exc:
            _LOGGER.info("LiteLLM not installed; skipping async close: %s", exc)
            return

        _register_safe_get_event_loop_atexit()
        close_fn: Any = getattr(litellm, "aclose", None)
        if close_fn is None:
            close_fn = getattr(litellm, "close_litellm_async_clients", None)
        if close_fn is None:
            return
        await close_fn()

    async def generate(self, vibe: str) -> MusicConfig:
        try:
            from litellm import acompletion  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("LiteLLM not installed: %s", exc)
            raise ModelNotAvailableError("litellm is not installed") from exc

        _register_safe_get_event_loop_atexit()
        messages: list[dict[str, str]] = [{"role": "system", "content": self._base_prompt}]
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        schema_str: str = ""
        if self._response_format:
            schema_str = f"<schema>{json.dumps(self._response_format.model_json_schema(), indent=2)}</schema>"
        messages.append(
            {
                "role": "user",
                "content": f"{schema_str}\n<vibe>{vibe}</vibe>\n<output>",
            }
        )

        request = _LiteLLMRequest(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format=self._response_format,
            api_key=self._api_key or None,
        ).model_dump(exclude_none=True)

        try:
            # LiteLLM response typing is not exported; keep runtime checks below.
            response: Any = await acompletion(**request, reasoning_effort=self._reasoning_effort)
        except Exception as exc:  # pragma: no cover - provider errors
            _LOGGER.warning("LiteLLM request failed: %s", exc, exc_info=True)
            raise LLMInferenceError(str(exc)) from exc

        content = ""
        try:
            assert response is not None
            if not hasattr(response, "choices"):
                raise ConfigGenerateError("LiteLLM response missing choices")
            raw_content = response.choices[0].message.content
            if not isinstance(raw_content, str) or not raw_content.strip():
                raise ConfigGenerateError("LiteLLM returned empty content")
            content = raw_content.strip()
            return MusicConfig.model_validate_json(content)
        except ValidationError as exc:
            extracted = _extract_json_payload(content)
            if extracted:
                try:
                    return MusicConfig.model_validate_json(extracted)
                except ValidationError:
                    _LOGGER.warning(
                        "LiteLLM returned invalid JSON after extraction.", exc_info=True
                    )
            snippet = _content_snippet(content) or "<empty>"
            _LOGGER.warning("LiteLLM returned invalid JSON: %s", snippet)
            raise ConfigGenerateError(f"LiteLLM returned non-JSON content: {snippet}") from exc
</file>

<file path="__init__.py">
from __future__ import annotations

from .audio import SAMPLE_RATE
from .config import (
    AccentStyle,
    AttackStyle,
    BassStyle,
    BrightnessLabel,
    DensityLevel,
    EchoLabel,
    GrainStyle,
    HumanFeelLabel,
    MelodyStyle,
    ModeName,
    MotionLabel,
    MusicConfig,
    MusicConfigUpdate,
    PadStyle,
    RhythmStyle,
    RootNote,
    SpaceLabel,
    StereoLabel,
    TempoLabel,
    TextureStyle,
)
from .main import (
    FallbackInput,
    PreviewPolicy,
    Streamable,
    StreamEvent,
    StreamHooks,
    astream,
    render,
    save_wav,
    stream,
    stream_configs,
    stream_texts,
    stream_updates,
)
from .models import ModelForGeneratingMusicConfig, ModelSpec

__all__ = [
    "SAMPLE_RATE",
    "AccentStyle",
    "AttackStyle",
    "BassStyle",
    "BrightnessLabel",
    "DensityLevel",
    "EchoLabel",
    "GrainStyle",
    "HumanFeelLabel",
    "MelodyStyle",
    "ModeName",
    "MotionLabel",
    "MusicConfig",
    "MusicConfigUpdate",
    "ModelSpec",
    "ModelForGeneratingMusicConfig",
    "PadStyle",
    "RhythmStyle",
    "RootNote",
    "SpaceLabel",
    "StereoLabel",
    "StreamEvent",
    "StreamHooks",
    "Streamable",
    "TextureStyle",
    "TempoLabel",
    "FallbackInput",
    "PreviewPolicy",
    "astream",
    "render",
    "save_wav",
    "stream",
    "stream_configs",
    "stream_texts",
    "stream_updates",
]

__version__ = "0.1.0"
</file>

<file path="audio.py">
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

SAMPLE_RATE = 44100


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
</file>

<file path="cli.py">
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

from .audio import SAMPLE_RATE
from .errors import ModelNotAvailableError
from .main import render, save_wav

_EXPRESSIVE_REPO = "mlx-community/gemma-3-1b-it-qat-8bit"
_EXPRESSIVE_DIR = "gemma-3-1b-it-qat-8bit"
_LOGGER = logging.getLogger("latentscore.cli")


def _default_model_base() -> Path:
    configured = os.environ.get("LATENTSCORE_MODEL_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "latentscore" / "models"


def _download_expressive(model_base: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
        raise ModelNotAvailableError("huggingface_hub is not installed") from exc

    target = model_base / _EXPRESSIVE_DIR
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(_EXPRESSIVE_REPO, local_dir=str(target))
    return target


def _doctor_report(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latentscore")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Render a short demo clip.")
    demo.add_argument("--duration", type=float, default=2.5)
    demo.add_argument("--output", type=str, default="demo.wav")

    download = sub.add_parser("download", help="Download model assets.")
    download.add_argument("model", choices=["expressive"], type=str)

    sub.add_parser("doctor", help="Check model availability and cache paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "demo":
            audio = render("warm sunrise", duration=args.duration)
            path = save_wav(args.output, audio)
            print(f"Wrote demo to {path} (sr={SAMPLE_RATE})")
            return 0

        if args.command == "download":
            model_base = _default_model_base()
            model_base.mkdir(parents=True, exist_ok=True)
            if args.model == "expressive":
                target = _download_expressive(model_base)
                print(f"Downloaded expressive model to {target}")
                return 0

        if args.command == "doctor":
            model_base = _default_model_base()
            expressive_dir = model_base / _EXPRESSIVE_DIR
            embeddings_dir = Path("models") / "all-MiniLM-L6-v2"
            report = [
                f"Model cache base: {model_base}",
                f"Expressive model present: {expressive_dir.exists()}",
                f"Embeddings model present: {embeddings_dir.exists()}",
                "Hints:",
                "- Run `latentscore download expressive` to prefetch the LLM weights.",
                "- Set LATENTSCORE_MODEL_DIR to point at a preseeded models directory.",
            ]
            _doctor_report(report)
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        _LOGGER.warning("latentscore CLI failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
</file>

<file path="config.py">
from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

from .errors import InvalidConfigError

_LOGGER = logging.getLogger("latentscore.config")

TempoLabel = Literal["very_slow", "slow", "medium", "fast", "very_fast"]
BrightnessLabel = Literal["very_dark", "dark", "medium", "bright", "very_bright"]
SpaceLabel = Literal["dry", "small", "medium", "large", "vast"]
MotionLabel = Literal["static", "slow", "medium", "fast", "chaotic"]
StereoLabel = Literal["mono", "narrow", "medium", "wide", "ultra_wide"]
EchoLabel = Literal["none", "subtle", "medium", "heavy", "infinite"]
HumanFeelLabel = Literal["robotic", "tight", "natural", "loose", "drunk"]
RootNote = Literal["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
ModeName = Literal["major", "minor", "dorian", "mixolydian"]
DensityLevel = Literal[2, 3, 4, 5, 6]
BassStyle = Literal[
    "drone",
    "sustained",
    "pulsing",
    "walking",
    "fifth_drone",
    "sub_pulse",
    "octave",
    "arp_bass",
]
PadStyle = Literal[
    "warm_slow",
    "dark_sustained",
    "cinematic",
    "thin_high",
    "ambient_drift",
    "stacked_fifths",
    "bright_open",
]
MelodyStyle = Literal[
    "contemplative",
    "rising",
    "falling",
    "minimal",
    "ornamental",
    "arp_melody",
    "contemplative_minor",
    "call_response",
    "heroic",
]
RhythmStyle = Literal[
    "none",
    "minimal",
    "heartbeat",
    "soft_four",
    "hats_only",
    "electronic",
    "kit_light",
    "kit_medium",
    "military",
    "tabla_essence",
    "brush",
]
TextureStyle = Literal[
    "none",
    "shimmer",
    "shimmer_slow",
    "vinyl_crackle",
    "breath",
    "stars",
    "glitch",
    "noise_wash",
    "crystal",
    "pad_whisper",
]
AccentStyle = Literal[
    "none",
    "bells",
    "pluck",
    "chime",
    "bells_dense",
    "blip",
    "blip_random",
    "brass_hit",
    "wind",
    "arp_accent",
    "piano_note",
]
AttackStyle = Literal["soft", "medium", "sharp"]
GrainStyle = Literal["clean", "warm", "gritty"]


T = TypeVar("T")

_TEMPO_VERY_SLOW = 0.15
_TEMPO_SLOW = 0.3
_TEMPO_MEDIUM = 0.5
_TEMPO_FAST = 0.7
_TEMPO_VERY_FAST = 0.9

_BRIGHTNESS_VERY_DARK = 0.1
_BRIGHTNESS_DARK = 0.3
_BRIGHTNESS_MEDIUM = 0.5
_BRIGHTNESS_BRIGHT = 0.7
_BRIGHTNESS_VERY_BRIGHT = 0.9

_SPACE_DRY = 0.1
_SPACE_SMALL = 0.3
_SPACE_MEDIUM = 0.5
_SPACE_LARGE = 0.7
_SPACE_VAST = 0.95

_MOTION_STATIC = 0.1
_MOTION_SLOW = 0.3
_MOTION_MEDIUM = 0.5
_MOTION_FAST = 0.7
_MOTION_CHAOTIC = 0.9

_STEREO_MONO = 0.0
_STEREO_NARROW = 0.25
_STEREO_MEDIUM = 0.5
_STEREO_WIDE = 0.75
_STEREO_ULTRA_WIDE = 1.0

_ECHO_NONE = 0.0
_ECHO_SUBTLE = 0.25
_ECHO_MEDIUM = 0.5
_ECHO_HEAVY = 0.75
_ECHO_INFINITE = 0.95

_HUMAN_ROBOTIC = 0.0
_HUMAN_TIGHT = 0.15
_HUMAN_NATURAL = 0.3
_HUMAN_LOOSE = 0.5
_HUMAN_DRUNK = 0.8


def _optional_map(value: T | None, mapper: Callable[[T], float]) -> float | None:
    match value:
        case None:
            return None
        case _:
            return mapper(value)


def tempo_to_float(value: TempoLabel) -> float:
    match value:
        case "very_slow":
            return _TEMPO_VERY_SLOW
        case "slow":
            return _TEMPO_SLOW
        case "medium":
            return _TEMPO_MEDIUM
        case "fast":
            return _TEMPO_FAST
        case "very_fast":
            return _TEMPO_VERY_FAST
        case _:
            raise InvalidConfigError(f"Unknown tempo label: {value!r}")


def brightness_to_float(value: BrightnessLabel) -> float:
    match value:
        case "very_dark":
            return _BRIGHTNESS_VERY_DARK
        case "dark":
            return _BRIGHTNESS_DARK
        case "medium":
            return _BRIGHTNESS_MEDIUM
        case "bright":
            return _BRIGHTNESS_BRIGHT
        case "very_bright":
            return _BRIGHTNESS_VERY_BRIGHT
        case _:
            raise InvalidConfigError(f"Unknown brightness label: {value!r}")


def space_to_float(value: SpaceLabel) -> float:
    match value:
        case "dry":
            return _SPACE_DRY
        case "small":
            return _SPACE_SMALL
        case "medium":
            return _SPACE_MEDIUM
        case "large":
            return _SPACE_LARGE
        case "vast":
            return _SPACE_VAST
        case _:
            raise InvalidConfigError(f"Unknown space label: {value!r}")


def motion_to_float(value: MotionLabel) -> float:
    match value:
        case "static":
            return _MOTION_STATIC
        case "slow":
            return _MOTION_SLOW
        case "medium":
            return _MOTION_MEDIUM
        case "fast":
            return _MOTION_FAST
        case "chaotic":
            return _MOTION_CHAOTIC
        case _:
            raise InvalidConfigError(f"Unknown motion label: {value!r}")


def stereo_to_float(value: StereoLabel) -> float:
    match value:
        case "mono":
            return _STEREO_MONO
        case "narrow":
            return _STEREO_NARROW
        case "medium":
            return _STEREO_MEDIUM
        case "wide":
            return _STEREO_WIDE
        case "ultra_wide":
            return _STEREO_ULTRA_WIDE
        case _:
            raise InvalidConfigError(f"Unknown stereo label: {value!r}")


def echo_to_float(value: EchoLabel) -> float:
    match value:
        case "none":
            return _ECHO_NONE
        case "subtle":
            return _ECHO_SUBTLE
        case "medium":
            return _ECHO_MEDIUM
        case "heavy":
            return _ECHO_HEAVY
        case "infinite":
            return _ECHO_INFINITE
        case _:
            raise InvalidConfigError(f"Unknown echo label: {value!r}")


def human_to_float(value: HumanFeelLabel) -> float:
    match value:
        case "robotic":
            return _HUMAN_ROBOTIC
        case "tight":
            return _HUMAN_TIGHT
        case "natural":
            return _HUMAN_NATURAL
        case "loose":
            return _HUMAN_LOOSE
        case "drunk":
            return _HUMAN_DRUNK
        case _:
            raise InvalidConfigError(f"Unknown human feel label: {value!r}")


class _MusicConfigInternal(BaseModel):
    """Internal config with numeric values used by the synth engine."""

    schema_version: Literal[1] = 1
    tempo: float = 0.5
    brightness: float = 0.5
    root: RootNote = "c"
    mode: ModeName = "minor"
    space: float = 0.5
    density: DensityLevel = 5

    bass: BassStyle = "drone"
    pad: PadStyle = "warm_slow"
    melody: MelodyStyle = "contemplative"
    rhythm: RhythmStyle = "minimal"
    texture: TextureStyle = "shimmer"
    accent: AccentStyle = "bells"

    motion: float = 0.5
    attack: AttackStyle = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: GrainStyle = "clean"

    model_config = ConfigDict(extra="forbid")


class MusicConfig(BaseModel):
    """Strictly-typed public config with tolerant extra capture."""

    schema_version: Literal[1] = 1
    tempo: TempoLabel = "medium"
    brightness: BrightnessLabel = "medium"
    root: RootNote = "c"
    mode: ModeName = "minor"
    space: SpaceLabel = "medium"
    density: DensityLevel = 5

    bass: BassStyle = "drone"
    pad: PadStyle = "warm_slow"
    melody: MelodyStyle = "contemplative"
    rhythm: RhythmStyle = "minimal"
    texture: TextureStyle = "shimmer"
    accent: AccentStyle = "bells"

    motion: MotionLabel = "medium"
    attack: AttackStyle = "medium"
    stereo: StereoLabel = "medium"
    depth: bool = False
    echo: EchoLabel = "medium"
    human: HumanFeelLabel = "robotic"
    grain: GrainStyle = "clean"

    model_config = ConfigDict(extra="allow")

    @property
    def extras(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def to_internal(self) -> _MusicConfigInternal:
        return _MusicConfigInternal(
            tempo=tempo_to_float(self.tempo),
            brightness=brightness_to_float(self.brightness),
            root=self.root,
            mode=self.mode,
            space=space_to_float(self.space),
            density=self.density,
            bass=self.bass,
            pad=self.pad,
            melody=self.melody,
            rhythm=self.rhythm,
            texture=self.texture,
            accent=self.accent,
            motion=motion_to_float(self.motion),
            attack=self.attack,
            stereo=stereo_to_float(self.stereo),
            depth=self.depth,
            echo=echo_to_float(self.echo),
            human=human_to_float(self.human),
            grain=self.grain,
        )


class _MusicConfigUpdateInternal(BaseModel):
    """Partial update for internal numeric configs."""

    tempo: Optional[float] = None
    brightness: Optional[float] = None
    root: Optional[RootNote] = None
    mode: Optional[ModeName] = None
    space: Optional[float] = None
    density: Optional[DensityLevel] = None

    bass: Optional[BassStyle] = None
    pad: Optional[PadStyle] = None
    melody: Optional[MelodyStyle] = None
    rhythm: Optional[RhythmStyle] = None
    texture: Optional[TextureStyle] = None
    accent: Optional[AccentStyle] = None

    motion: Optional[float] = None
    attack: Optional[AttackStyle] = None
    stereo: Optional[float] = None
    depth: Optional[bool] = None
    echo: Optional[float] = None
    human: Optional[float] = None
    grain: Optional[GrainStyle] = None

    model_config = ConfigDict(extra="forbid")


class MusicConfigUpdate(BaseModel):
    """Partial update for human inputs; unknown keys are rejected."""

    tempo: Optional[TempoLabel] = None
    brightness: Optional[BrightnessLabel] = None
    root: Optional[RootNote] = None
    mode: Optional[ModeName] = None
    space: Optional[SpaceLabel] = None
    density: Optional[DensityLevel] = None

    bass: Optional[BassStyle] = None
    pad: Optional[PadStyle] = None
    melody: Optional[MelodyStyle] = None
    rhythm: Optional[RhythmStyle] = None
    texture: Optional[TextureStyle] = None
    accent: Optional[AccentStyle] = None

    motion: Optional[MotionLabel] = None
    attack: Optional[AttackStyle] = None
    stereo: Optional[StereoLabel] = None
    depth: Optional[bool] = None
    echo: Optional[EchoLabel] = None
    human: Optional[HumanFeelLabel] = None
    grain: Optional[GrainStyle] = None

    model_config = ConfigDict(extra="forbid")

    def to_internal(self) -> _MusicConfigUpdateInternal:
        return _MusicConfigUpdateInternal(
            tempo=_optional_map(self.tempo, tempo_to_float),
            brightness=_optional_map(self.brightness, brightness_to_float),
            root=self.root,
            mode=self.mode,
            space=_optional_map(self.space, space_to_float),
            density=self.density,
            bass=self.bass,
            pad=self.pad,
            melody=self.melody,
            rhythm=self.rhythm,
            texture=self.texture,
            accent=self.accent,
            motion=_optional_map(self.motion, motion_to_float),
            attack=self.attack,
            stereo=_optional_map(self.stereo, stereo_to_float),
            depth=self.depth,
            echo=_optional_map(self.echo, echo_to_float),
            human=_optional_map(self.human, human_to_float),
            grain=self.grain,
        )


ConfigInput = MusicConfig | _MusicConfigInternal
UpdateInput = MusicConfigUpdate | _MusicConfigUpdateInternal


def coerce_internal_config(config: ConfigInput) -> _MusicConfigInternal:
    match config:
        case _MusicConfigInternal():
            return config
        case MusicConfig():
            return config.to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported config type: {type(config).__name__}")


def coerce_internal_update(update: UpdateInput) -> _MusicConfigUpdateInternal:
    match update:
        case _MusicConfigUpdateInternal():
            return update
        case MusicConfigUpdate():
            return update.to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported update type: {type(update).__name__}")


def merge_config(base: MusicConfig, update: MusicConfigUpdate) -> MusicConfig:
    """Apply a partial update to a base config."""

    changes = update.model_dump(exclude_none=True)
    return base.model_copy(update=changes)


def merge_internal_config(
    base: _MusicConfigInternal,
    update: _MusicConfigUpdateInternal,
) -> _MusicConfigInternal:
    """Apply a partial update to a base internal config."""

    changes = update.model_dump(exclude_none=True)
    return base.model_copy(update=changes)


def parse_update(payload: Mapping[str, Any]) -> MusicConfigUpdate:
    """Parse a strict update payload, raising InvalidConfigError on failure."""

    try:
        return MusicConfigUpdate.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Failed to parse config update: %s", exc, exc_info=True)
        raise InvalidConfigError(str(exc)) from exc


def parse_config(payload: Mapping[str, Any]) -> MusicConfig:
    """Parse a tolerant config payload, raising InvalidConfigError on failure."""

    try:
        return MusicConfig.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Failed to parse config payload: %s", exc, exc_info=True)
        raise InvalidConfigError(str(exc)) from exc


def is_empty_update(update: UpdateInput) -> bool:
    """Return True when the update does not specify any fields."""

    return not update.model_dump(exclude_none=True)
</file>

<file path="errors.py">
from __future__ import annotations


class LatentScoreError(Exception):
    """Base error for the LatentScore library."""


class InvalidConfigError(LatentScoreError):
    """Raised when a config cannot be parsed or validated."""


class LLMInferenceError(LatentScoreError):
    """Raised when a model provider fails to produce a response."""


class ConfigGenerateError(LatentScoreError):
    """Raised when a model fails while generating a config."""


class ModelNotAvailableError(LatentScoreError):
    """Raised when required model weights or dependencies are missing."""
</file>

<file path="main.py">
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Callable, Coroutine, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from .audio import SAMPLE_RATE, AudioNumbers, FloatArray, ensure_audio_contract, write_wav
from .config import (
    ConfigInput,
    MusicConfig,
    MusicConfigUpdate,
    UpdateInput,
    _MusicConfigInternal,
    _MusicConfigUpdateInternal,
    coerce_internal_config,
    coerce_internal_update,
    is_empty_update,
    merge_internal_config,
)
from .errors import InvalidConfigError
from .models import ModelForGeneratingMusicConfig, ModelSpec, resolve_model
from .synth import MusicConfig as SynthConfig
from .synth import assemble, interpolate_configs

StreamContent = str | ConfigInput | UpdateInput
PreviewPolicy = Literal["none", "embedding"]
FallbackPolicy = Literal["none", "keep_last", "embedding"]
FallbackInput = FallbackPolicy | ConfigInput | UpdateInput


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal[
        "stream_start",
        "item_resolve_start",
        "item_resolve_success",
        "item_resolve_error",
        "item_preview_start",
        "first_config_ready",
        "first_audio_chunk",
        "stream_end",
        "fallback_used",
    ]
    index: int | None = None
    item: Streamable | None = None
    error: Exception | None = None
    fallback: FallbackInput | None = None
    preview_policy: PreviewPolicy | None = None


@dataclass(frozen=True)
class StreamHooks:
    on_event: Callable[[StreamEvent], None] | None = None
    on_stream_start: Callable[[], None] | None = None
    on_item_resolve_start: Callable[[int, Streamable], None] | None = None
    on_item_resolve_success: Callable[[int, Streamable], None] | None = None
    on_item_resolve_error: (
        Callable[[int, Streamable, Exception, FallbackInput | None], None] | None
    ) = None
    on_item_preview_start: Callable[[int, Streamable, PreviewPolicy], None] | None = None
    on_first_config_ready: Callable[[int, Streamable], None] | None = None
    on_first_audio_chunk: Callable[[], None] | None = None
    on_stream_end: Callable[[], None] | None = None


_MIN_CHUNK_SECONDS = 1e-6
_LOGGER = logging.getLogger("latentscore.main")


class Streamable(BaseModel):
    """Structured streaming input with timing controls."""

    content: StreamContent
    duration: float = Field(default=60.0, gt=0.0)
    transition_duration: float = Field(default=1.0, ge=0.0)
    fallback: FallbackInput | None = None
    preview_policy: PreviewPolicy | None = None

    model_config = ConfigDict(extra="forbid")


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _LOGGER.info("No running event loop; running coroutine directly.")
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def _log_exception(context: str, exc: Exception) -> None:
    _LOGGER.warning("%s failed: %s", context, exc, exc_info=True)


def _emit_event(hooks: StreamHooks | None, event: StreamEvent) -> None:
    if hooks is None:
        return
    try:
        if hooks.on_event is not None:
            hooks.on_event(event)
        match event.kind:
            case "stream_start":
                if hooks.on_stream_start is not None:
                    hooks.on_stream_start()
            case "item_resolve_start":
                if (
                    hooks.on_item_resolve_start is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_item_resolve_start(event.index, event.item)
            case "item_resolve_success":
                if (
                    hooks.on_item_resolve_success is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_item_resolve_success(event.index, event.item)
            case "item_resolve_error":
                if (
                    hooks.on_item_resolve_error is not None
                    and event.index is not None
                    and event.item is not None
                    and event.error is not None
                ):
                    hooks.on_item_resolve_error(
                        event.index,
                        event.item,
                        event.error,
                        event.fallback,
                    )
            case "item_preview_start":
                if (
                    hooks.on_item_preview_start is not None
                    and event.index is not None
                    and event.item is not None
                    and event.preview_policy is not None
                ):
                    hooks.on_item_preview_start(
                        event.index,
                        event.item,
                        event.preview_policy,
                    )
            case "first_config_ready":
                if (
                    hooks.on_first_config_ready is not None
                    and event.index is not None
                    and event.item is not None
                ):
                    hooks.on_first_config_ready(event.index, event.item)
            case "first_audio_chunk":
                if hooks.on_first_audio_chunk is not None:
                    hooks.on_first_audio_chunk()
            case "stream_end":
                if hooks.on_stream_end is not None:
                    hooks.on_stream_end()
            case "fallback_used":
                pass
            case _:
                pass
    except Exception as exc:
        _LOGGER.warning("Stream hook failed: %s", exc, exc_info=True)


@dataclass(slots=True)
class _PrefetchedItem:
    index: int
    item: Streamable
    task: asyncio.Task[_MusicConfigInternal] | None


def _to_synth_config(config: _MusicConfigInternal) -> SynthConfig:
    return SynthConfig(
        tempo=config.tempo,
        root=config.root,
        mode=config.mode,
        brightness=config.brightness,
        space=config.space,
        density=config.density,
        bass=config.bass,
        pad=config.pad,
        melody=config.melody,
        rhythm=config.rhythm,
        texture=config.texture,
        accent=config.accent,
        motion=config.motion,
        attack=config.attack,
        stereo=config.stereo,
        depth=config.depth,
        echo=config.echo,
        human=config.human,
        grain=config.grain,
    )


def _from_synth_config(config: SynthConfig) -> _MusicConfigInternal:
    return _MusicConfigInternal(**config.model_dump())


def _apply_update(
    base: _MusicConfigInternal,
    update: UpdateInput | None,
) -> _MusicConfigInternal:
    match update:
        case None:
            return base
        case _:
            internal_update = coerce_internal_update(update)
            return merge_internal_config(base, internal_update)


def _chunk_count(duration: float, chunk_seconds: float) -> int:
    return max(1, int(round(duration / max(chunk_seconds, _MIN_CHUNK_SECONDS))))


def _transition_steps(transition_duration: float, chunk_seconds: float) -> int:
    match transition_duration:
        case (float() | int()) as duration if duration <= 0:
            return 0
        case (float() | int()) as duration:
            return max(1, int(round(duration / max(chunk_seconds, _MIN_CHUNK_SECONDS))))
        case _:
            raise InvalidConfigError("transition_duration must be a number")


def _iter_transition_configs(
    current: _MusicConfigInternal | None,
    target: _MusicConfigInternal,
    steps: int,
) -> Iterator[_MusicConfigInternal]:
    if current is None or steps <= 1:
        yield target
        return
    start = _to_synth_config(current)
    end = _to_synth_config(target)
    for step in range(1, steps + 1):
        t = step / steps
        interpolated = interpolate_configs(start, end, t)
        yield _from_synth_config(interpolated)


def _wrap_streamables(
    items: Iterable[StreamContent],
    *,
    duration: float,
    transition_duration: float,
) -> Iterator[Streamable]:
    for item in items:
        yield Streamable(
            content=item,
            duration=duration,
            transition_duration=transition_duration,
        )


def render(
    vibe: str,
    *,
    duration: float = 8.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
) -> FloatArray:
    try:
        resolved = resolve_model(model)
        match config:
            case None:
                base = _run_async(resolved.generate(vibe)).to_internal()
            case _:
                base = coerce_internal_config(config)
        target = _apply_update(base, update)

        audio = assemble(_to_synth_config(target), duration)
        return ensure_audio_contract(audio, sample_rate=SAMPLE_RATE)
    except Exception as exc:
        _log_exception("render", exc)
        raise


def _bridge_async_stream(
    async_iter: AsyncIterable[FloatArray],
    *,
    queue_maxsize: int = 0,
) -> Iterator[FloatArray]:
    sentinel = object()
    queue: Queue[object] = Queue(maxsize=queue_maxsize) if queue_maxsize > 0 else Queue()

    def _runner() -> None:
        async def _run() -> None:
            try:
                async for chunk in async_iter:
                    queue.put(chunk)
            except BaseException as exc:
                queue.put(exc)
            finally:
                queue.put(sentinel)

        asyncio.run(_run())

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield cast(FloatArray, item)
    thread.join(timeout=0)


def stream(
    items: Iterable[Streamable],
    *,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
    queue_maxsize: int = 0,
) -> Iterable[FloatArray]:
    try:
        items_obj: object = items
        match items_obj:
            case Streamable():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream expects an iterable of Streamable items")
            case str():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream expects an iterable of Streamable items")
            case _:
                pass

        if prefetch_depth < 0:
            raise InvalidConfigError("prefetch_depth must be >= 0")

        async_iter = astream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
        for chunk in _bridge_async_stream(async_iter, queue_maxsize=queue_maxsize):
            yield chunk
    except Exception as exc:
        _log_exception("stream", exc)
        raise


def stream_texts(
    prompts: Iterable[str],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        match prompts:
            case str():
                raise InvalidConfigError("stream_texts expects an iterable of strings")
            case _:
                pass

        items = _wrap_streamables(
            prompts,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        _log_exception("stream_texts", exc)
        raise


def stream_configs(
    configs: Iterable[ConfigInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        configs_obj: object = configs
        match configs_obj:
            case MusicConfig() | _MusicConfigInternal():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream_configs expects an iterable of config objects")
            case _:
                pass

        items = _wrap_streamables(
            configs,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        _log_exception("stream_configs", exc)
        raise


def stream_updates(
    updates: Iterable[UpdateInput],
    *,
    duration: float = 60.0,
    transition_duration: float = 1.0,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> Iterable[FloatArray]:
    try:
        updates_obj: object = updates
        match updates_obj:
            case MusicConfigUpdate() | _MusicConfigUpdateInternal():  # type: ignore[reportUnnecessaryComparison]
                raise InvalidConfigError("stream_updates expects an iterable of update objects")
            case _:
                pass

        items = _wrap_streamables(
            updates,
            duration=duration,
            transition_duration=transition_duration,
        )
        return stream(
            items,
            chunk_seconds=chunk_seconds,
            model=model,
            config=config,
            update=update,
            prefetch_depth=prefetch_depth,
            preview_policy=preview_policy,
            fallback=fallback,
            fallback_model=fallback_model,
            hooks=hooks,
        )
    except Exception as exc:
        _log_exception("stream_updates", exc)
        raise


def save_wav(path: str, audio_or_chunks: AudioNumbers | Iterable[AudioNumbers]) -> str:
    try:
        written = write_wav(path, audio_or_chunks, sample_rate=SAMPLE_RATE)
        return str(written)
    except Exception as exc:
        _log_exception("save_wav", exc)
        raise


def _coerce_async_items(
    items: Iterable[Streamable] | AsyncIterable[Streamable],
) -> AsyncIterable[Streamable]:
    items_obj: object = items
    match items_obj:
        case Streamable():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("astream expects an iterable of Streamable items")
        case str():  # type: ignore[reportUnnecessaryComparison]
            raise InvalidConfigError("astream expects an iterable of Streamable items")
        case AsyncIterable() as async_items:
            return async_items
        case Iterable() as sync_items:
            pass
        case _:
            raise InvalidConfigError("astream expects an iterable of Streamable items")

    async def _iter() -> AsyncIterator[Streamable]:
        for item in sync_items:
            yield item

    return _iter()


async def _generate_internal(
    vibe: str,
    model: ModelForGeneratingMusicConfig,
) -> _MusicConfigInternal:
    return (await model.generate(vibe)).to_internal()


async def _resolve_preview(
    item: Streamable,
    *,
    preview_policy: PreviewPolicy,
    fallback_model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
) -> _MusicConfigInternal | None:
    if preview_policy != "embedding" or not isinstance(item.content, str):
        return None
    preview = await fallback_model.generate(item.content)
    return _apply_update(preview.to_internal(), update)


async def _resolve_fallback(
    item: Streamable,
    *,
    current: _MusicConfigInternal | None,
    fallback: FallbackInput | None,
    fallback_model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
) -> _MusicConfigInternal | None:
    if fallback is None or fallback == "keep_last":
        base = current or _MusicConfigInternal()
        return _apply_update(base, update)
    if fallback == "none":
        return None
    if fallback == "embedding":
        if not isinstance(item.content, str):
            base = current or _MusicConfigInternal()
            return _apply_update(base, update)
        target = await fallback_model.generate(item.content)
        return _apply_update(target.to_internal(), update)
    return await _resolve_target_async(
        fallback,
        current=current,
        model=fallback_model,
        update=update,
    )


async def _render_chunk(
    config_item: _MusicConfigInternal,
    *,
    chunk_seconds: float,
) -> FloatArray:
    return await asyncio.to_thread(
        lambda: ensure_audio_contract(
            assemble(_to_synth_config(config_item), chunk_seconds),
            sample_rate=SAMPLE_RATE,
        )
    )


async def _prefetch_item(
    item: Streamable,
    *,
    model: ModelForGeneratingMusicConfig,
    hooks: StreamHooks | None,
    index: int,
) -> _PrefetchedItem:
    task: asyncio.Task[_MusicConfigInternal] | None = None
    if isinstance(item.content, str):
        _emit_event(hooks, StreamEvent(kind="item_resolve_start", index=index, item=item))
        task = asyncio.create_task(_generate_internal(item.content, model))
    return _PrefetchedItem(index=index, item=item, task=task)


async def _resolve_target_async(
    item: StreamContent,
    *,
    current: _MusicConfigInternal | None,
    model: ModelForGeneratingMusicConfig,
    update: UpdateInput | None,
) -> _MusicConfigInternal:
    match item:
        case _MusicConfigInternal():
            target = item
        case MusicConfig():
            target = item.to_internal()
        case _MusicConfigUpdateInternal():
            base = current or _MusicConfigInternal()
            target = (
                base
                if current is None and is_empty_update(item)
                else merge_internal_config(base, item)
            )
        case MusicConfigUpdate():
            base = current or _MusicConfigInternal()
            internal_update = item.to_internal()
            target = (
                base
                if current is None and is_empty_update(item)
                else merge_internal_config(base, internal_update)
            )
        case str():
            target = (await model.generate(item)).to_internal()
        case _:
            raise InvalidConfigError(f"Unsupported input type: {type(item).__name__}")

    return _apply_update(target, update)


async def astream(
    items: Iterable[Streamable] | AsyncIterable[Streamable],
    *,
    chunk_seconds: float = 1.0,
    model: ModelSpec = "fast",
    config: ConfigInput | None = None,
    update: UpdateInput | None = None,
    prefetch_depth: int = 1,
    preview_policy: PreviewPolicy = "none",
    fallback: FallbackInput | None = None,
    fallback_model: ModelSpec = "fast",
    hooks: StreamHooks | None = None,
) -> AsyncIterable[FloatArray]:
    try:
        match chunk_seconds:
            case (float() | int()) as value if value <= 0:
                raise InvalidConfigError("chunk_seconds must be greater than 0")
            case float() | int():
                pass
            case _:
                raise InvalidConfigError("chunk_seconds must be a number")

        if prefetch_depth < 0:
            raise InvalidConfigError("prefetch_depth must be >= 0")

        current = coerce_internal_config(config) if config is not None else None
        resolved = resolve_model(model)
        fallback_resolved = resolve_model(fallback_model)

        _emit_event(hooks, StreamEvent(kind="stream_start"))

        async_items = _coerce_async_items(items)
        iterator = async_items.__aiter__()
        pending: deque[_PrefetchedItem] = deque()
        next_index = 0
        prefetch_target = max(1, prefetch_depth + 1)
        first_config_ready = False
        first_audio_chunk = False

        async def _next_item() -> Streamable | None:
            try:
                return await iterator.__anext__()
            except StopAsyncIteration:
                return None

        async def _fill_pending() -> None:
            nonlocal next_index
            while len(pending) < prefetch_target:
                item = await _next_item()
                if item is None:
                    break
                pending.append(
                    await _prefetch_item(
                        item,
                        model=resolved,
                        hooks=hooks,
                        index=next_index,
                    )
                )
                next_index += 1

        await _fill_pending()
        while pending:
            prefetched = pending.popleft()
            await _fill_pending()

            item = prefetched.item
            index = prefetched.index
            item_preview = item.preview_policy or preview_policy
            item_fallback = item.fallback if item.fallback is not None else fallback
            pending_task = prefetched.task

            pending_real_task: asyncio.Task[_MusicConfigInternal] | None = None
            target: _MusicConfigInternal | None = None

            if pending_task is None:
                target = await _resolve_target_async(
                    item.content,
                    current=current,
                    model=resolved,
                    update=update,
                )
                _emit_event(hooks, StreamEvent(kind="item_resolve_success", index=index, item=item))
            else:
                if pending_task.done():
                    try:
                        target = _apply_update(pending_task.result(), update)
                        _emit_event(
                            hooks,
                            StreamEvent(kind="item_resolve_success", index=index, item=item),
                        )
                    except Exception as exc:
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_error",
                                index=index,
                                item=item,
                                error=exc,
                                fallback=item_fallback,
                            ),
                        )
                        target = await _resolve_fallback(
                            item,
                            current=current,
                            fallback=item_fallback,
                            fallback_model=fallback_resolved,
                            update=update,
                        )
                        if target is None:
                            raise
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="fallback_used",
                                index=index,
                                item=item,
                                fallback=item_fallback,
                            ),
                        )
                else:
                    preview = await _resolve_preview(
                        item,
                        preview_policy=item_preview,
                        fallback_model=fallback_resolved,
                        update=update,
                    )
                    if preview is not None:
                        target = preview
                        pending_real_task = pending_task
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_preview_start",
                                index=index,
                                item=item,
                                preview_policy=item_preview,
                            ),
                        )
                    else:
                        try:
                            target = _apply_update(await pending_task, update)
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="item_resolve_success",
                                    index=index,
                                    item=item,
                                ),
                            )
                        except Exception as exc:
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="item_resolve_error",
                                    index=index,
                                    item=item,
                                    error=exc,
                                    fallback=item_fallback,
                                ),
                            )
                            target = await _resolve_fallback(
                                item,
                                current=current,
                                fallback=item_fallback,
                                fallback_model=fallback_resolved,
                                update=update,
                            )
                            if target is None:
                                raise
                            _emit_event(
                                hooks,
                                StreamEvent(
                                    kind="fallback_used",
                                    index=index,
                                    item=item,
                                    fallback=item_fallback,
                                ),
                            )

            if not first_config_ready:
                _emit_event(hooks, StreamEvent(kind="first_config_ready", index=index, item=item))
                first_config_ready = True

            total_chunks = _chunk_count(item.duration, chunk_seconds)
            transition_chunks = (
                0
                if current is None
                else min(
                    total_chunks,
                    _transition_steps(item.transition_duration, chunk_seconds),
                )
            )

            remaining = total_chunks
            if transition_chunks > 0:
                for config_item in _iter_transition_configs(current, target, transition_chunks):
                    current = config_item
                    if not first_audio_chunk:
                        _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                        first_audio_chunk = True
                    yield await _render_chunk(config_item, chunk_seconds=chunk_seconds)
                remaining -= transition_chunks
            else:
                current = target

            match current:
                case _MusicConfigInternal() as current_config:
                    pass
                case None:
                    raise InvalidConfigError("Stream exhausted without a current config")
                case _:
                    raise InvalidConfigError("Stream exhausted with an invalid config")

            while remaining > 0:
                if pending_real_task is not None and pending_real_task.done():
                    try:
                        real_target = _apply_update(pending_real_task.result(), update)
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_success",
                                index=index,
                                item=item,
                            ),
                        )
                    except Exception as exc:
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="item_resolve_error",
                                index=index,
                                item=item,
                                error=exc,
                                fallback=item_fallback,
                            ),
                        )
                        real_target = await _resolve_fallback(
                            item,
                            current=current,
                            fallback=item_fallback,
                            fallback_model=fallback_resolved,
                            update=update,
                        )
                        if real_target is None:
                            raise
                        _emit_event(
                            hooks,
                            StreamEvent(
                                kind="fallback_used",
                                index=index,
                                item=item,
                                fallback=item_fallback,
                            ),
                        )

                    pending_real_task = None
                    transition = min(
                        remaining,
                        _transition_steps(item.transition_duration, chunk_seconds),
                    )
                    if transition > 0:
                        for config_item in _iter_transition_configs(
                            current_config,
                            real_target,
                            transition,
                        ):
                            current_config = config_item
                            if not first_audio_chunk:
                                _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                                first_audio_chunk = True
                            yield await _render_chunk(config_item, chunk_seconds=chunk_seconds)
                        remaining -= transition
                        current = current_config
                        continue
                    current_config = real_target
                    current = real_target

                if not first_audio_chunk:
                    _emit_event(hooks, StreamEvent(kind="first_audio_chunk"))
                    first_audio_chunk = True
                yield await _render_chunk(current_config, chunk_seconds=chunk_seconds)
                remaining -= 1

            if pending_real_task is not None and not pending_real_task.done():
                pending_real_task.cancel()
                pending_real_task = None

        _emit_event(hooks, StreamEvent(kind="stream_end"))
    except Exception as exc:
        _log_exception("astream", exc)
        raise
</file>

<file path="models.py">
from __future__ import annotations

import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypeGuard

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from .config import MusicConfig
from .errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError

ModelChoice = Literal["fast", "expressive"]

_EXPRESSIVE_REPO = "mlx-community/gemma-3-1b-it-qat-8bit"
_EXPRESSIVE_DIR = "gemma-3-1b-it-qat-8bit"
_DEFAULT_MAX_RETRIES = 3
_LLM_MAX_TOKENS = 3_000
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_EMBEDDING_DIR = Path("models") / _EMBEDDING_MODEL_NAME
_LOGGER = logging.getLogger("latentscore.models")


class ModelForGeneratingMusicConfig(Protocol):
    async def generate(self, vibe: str) -> MusicConfig: ...


ModelSpec = ModelChoice | ModelForGeneratingMusicConfig


def _is_model(obj: object) -> TypeGuard[ModelForGeneratingMusicConfig]:
    return hasattr(obj, "generate")


FEW_SHOT_EXAMPLES = """
Example 1:
Vibe: "dark ambient underwater cave with bioluminescence"
{
  "tempo": "very_slow",
  "root": "d",
  "mode": "dorian",
  "brightness": "very_dark",
  "space": "vast",
  "density": 4,
  "bass": "drone",
  "pad": "dark_sustained",
  "melody": "minimal",
  "rhythm": "none",
  "texture": "shimmer_slow",
  "accent": "none",
  "motion": "slow",
  "attack": "soft",
  "stereo": "wide",
  "depth": true,
  "echo": "heavy",
  "human": "tight",
  "grain": "warm"
}

Example 2:
Vibe: "uplifting sunrise over mountains"
{
  "tempo": "medium",
  "root": "c",
  "mode": "major",
  "brightness": "bright",
  "space": "large",
  "density": 4,
  "bass": "sustained",
  "pad": "warm_slow",
  "melody": "rising",
  "rhythm": "minimal",
  "texture": "shimmer",
  "accent": "bells",
  "motion": "medium",
  "attack": "soft",
  "stereo": "wide",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "clean"
}

Example 3:
Vibe: "cyberpunk nightclub in tokyo"
{
  "tempo": "fast",
  "root": "a",
  "mode": "minor",
  "brightness": "bright",
  "space": "small",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "arp_melody",
  "rhythm": "electronic",
  "texture": "none",
  "accent": "none",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "wide",
  "depth": true,
  "echo": "subtle",
  "human": "robotic",
  "grain": "gritty"
}

Example 4:
Vibe: "peaceful meditation in a zen garden"
{
  "tempo": "very_slow",
  "root": "f",
  "mode": "major",
  "brightness": "medium",
  "space": "large",
  "density": 2,
  "bass": "drone",
  "pad": "ambient_drift",
  "melody": "minimal",
  "rhythm": "none",
  "texture": "breath",
  "accent": "chime",
  "motion": "static",
  "attack": "soft",
  "stereo": "medium",
  "depth": false,
  "echo": "medium",
  "human": "natural",
  "grain": "clean"
}

Example 5:
Vibe: "epic cinematic battle scene"
{
  "tempo": "very_fast",
  "root": "d",
  "mode": "minor",
  "brightness": "medium",
  "space": "medium",
  "density": 6,
  "bass": "pulsing",
  "pad": "cinematic",
  "melody": "rising",
  "rhythm": "soft_four",
  "texture": "none",
  "accent": "bells",
  "motion": "fast",
  "attack": "sharp",
  "stereo": "ultra_wide",
  "depth": true,
  "echo": "subtle",
  "human": "robotic",
  "grain": "warm"
}
"""

_EXPRESSIVE_PROMPT = "\n".join(
    [
        "Role:",
        "You are a world-class music synthesis expert with deep music theory knowledge.",
        "",
        "Task:",
        "- Given a vibe/mood description, generate ONE MusicConfig JSON object.",
        "- Use the source examples as guidance for value choices.",
        "",
        "Output requirements:",
        "- Return only JSON (no markdown, no explanations).",
        "- Use only the keys shown in the examples.",
        "",
        "Few-shot examples:",
        FEW_SHOT_EXAMPLES.strip(),
    ]
)


def build_expressive_prompt() -> str:
    return _EXPRESSIVE_PROMPT


class ExampleConfig(BaseModel):
    vibe: str
    config: MusicConfig

    model_config = ConfigDict(frozen=True, extra="forbid")


FAST_EXAMPLES: tuple[ExampleConfig, ...] = (
    ExampleConfig(
        vibe="dark ambient underwater cave with bioluminescence",
        config=MusicConfig(
            tempo="very_slow",
            root="d",
            mode="dorian",
            brightness="very_dark",
            space="vast",
            density=4,
            bass="drone",
            pad="dark_sustained",
            melody="minimal",
            rhythm="none",
            texture="shimmer_slow",
            accent="none",
            motion="slow",
            attack="soft",
            stereo="wide",
            depth=True,
            echo="heavy",
            human="tight",
            grain="warm",
        ),
    ),
    ExampleConfig(
        vibe="uplifting sunrise over mountains",
        config=MusicConfig(
            tempo="medium",
            root="c",
            mode="major",
            brightness="bright",
            space="large",
            density=4,
            bass="sustained",
            pad="warm_slow",
            melody="rising",
            rhythm="minimal",
            texture="shimmer",
            accent="bells",
            motion="medium",
            attack="soft",
            stereo="wide",
            depth=False,
            echo="medium",
            human="natural",
            grain="clean",
        ),
    ),
    ExampleConfig(
        vibe="cyberpunk nightclub in tokyo",
        config=MusicConfig(
            tempo="fast",
            root="a",
            mode="minor",
            brightness="bright",
            space="small",
            density=6,
            bass="pulsing",
            pad="cinematic",
            melody="arp_melody",
            rhythm="electronic",
            texture="none",
            accent="none",
            motion="fast",
            attack="sharp",
            stereo="wide",
            depth=True,
            echo="subtle",
            human="robotic",
            grain="gritty",
        ),
    ),
    ExampleConfig(
        vibe="peaceful meditation in a zen garden",
        config=MusicConfig(
            tempo="very_slow",
            root="f",
            mode="major",
            brightness="medium",
            space="large",
            density=2,
            bass="drone",
            pad="ambient_drift",
            melody="minimal",
            rhythm="none",
            texture="breath",
            accent="chime",
            motion="static",
            attack="soft",
            stereo="medium",
            depth=False,
            echo="medium",
            human="natural",
            grain="clean",
        ),
    ),
    ExampleConfig(
        vibe="epic cinematic battle scene",
        config=MusicConfig(
            tempo="very_fast",
            root="d",
            mode="minor",
            brightness="medium",
            space="medium",
            density=6,
            bass="pulsing",
            pad="cinematic",
            melody="rising",
            rhythm="soft_four",
            texture="none",
            accent="bells",
            motion="fast",
            attack="sharp",
            stereo="ultra_wide",
            depth=True,
            echo="subtle",
            human="robotic",
            grain="warm",
        ),
    ),
)


def _heuristic_config(vibe: str) -> MusicConfig:
    vibe_lower = vibe.lower()
    density_sparse = 3
    density_full = 6

    def _has_any(words: Sequence[str]) -> bool:
        return any(word in vibe_lower for word in words)

    config = MusicConfig()
    match True:
        case _ if _has_any(("dark", "sad", "night", "mysterious")):
            config.root = "d"
            config.mode = "dorian"
            config.brightness = "dark"
        case _ if _has_any(("happy", "bright", "joy")):
            config.root = "c"
            config.mode = "major"
            config.brightness = "bright"
        case _ if _has_any(("epic", "cinematic", "powerful")):
            config.root = "d"
            config.mode = "minor"
            config.brightness = "medium"
            config.depth = True
        case _ if _has_any(("indian", "spiritual", "meditation")):
            config.root = "d"
            config.mode = "dorian"
            config.human = "natural"
            config.grain = "warm"
        case _:
            pass

    match True:
        case _ if _has_any(("slow", "calm", "meditation")):
            config.tempo = "slow"
        case _ if _has_any(("fast", "energy", "drive")):
            config.tempo = "fast"
        case _:
            config.tempo = "medium"

    match True:
        case _ if _has_any(("vast", "space", "underwater", "cave")):
            config.space = "vast"
            config.echo = "heavy"
        case _ if _has_any(("intimate", "close")):
            config.space = "small"
            config.echo = "subtle"
        case _:
            pass

    match True:
        case _ if _has_any(("electronic", "synth")):
            config.rhythm = "electronic"
            config.attack = "sharp"
        case _ if _has_any(("ambient", "peaceful")):
            config.rhythm = "none"
            config.attack = "soft"
        case _:
            pass

    match True:
        case _ if _has_any(("minimal", "sparse")):
            config.density = density_sparse
            config.melody = "minimal"
        case _ if _has_any(("rich", "full", "lush")):
            config.density = density_full
        case _:
            pass

    return config


class FastEmbeddingModel:
    """Embeddings-backed config selection with a heuristic fallback."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir

    async def generate(self, vibe: str) -> MusicConfig:
        try:
            return await asyncio.to_thread(self._embed_and_select, vibe)
        except Exception as exc:  # pragma: no cover - runtime fallback
            _LOGGER.warning("Fast model fallback: %s", exc, exc_info=True)
            return _heuristic_config(vibe)

    @functools.lru_cache(maxsize=1)
    def _load_encoder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("sentence-transformers not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError("sentence-transformers is not installed") from exc

        model_dir = self._model_dir
        if model_dir is None:
            model_dir = _LOCAL_EMBEDDING_DIR if _LOCAL_EMBEDDING_DIR.exists() else None

        return SentenceTransformer(str(model_dir) if model_dir else _EMBEDDING_MODEL_NAME)

    @functools.lru_cache(maxsize=1)
    def _example_matrix(self) -> NDArray[np.float32]:
        encoder = self._load_encoder()
        texts = [example.vibe for example in FAST_EXAMPLES]
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_and_select(self, vibe: str) -> MusicConfig:
        encoder = self._load_encoder()
        example_matrix = self._example_matrix()
        query = encoder.encode([vibe], normalize_embeddings=True)
        query_vec = np.asarray(query, dtype=np.float32)[0]
        scores = example_matrix @ query_vec
        best_idx = int(np.argmax(scores))
        return FAST_EXAMPLES[best_idx].config


class ExpressiveMlxModel:
    """Local MLX-backed model using the LLM prompt from the synth prototype."""

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        allow_download: bool = True,
    ) -> None:
        self._model_dir = model_dir
        self._max_retries = max_retries
        self._allow_download = allow_download

    async def generate(self, vibe: str) -> MusicConfig:
        return await asyncio.to_thread(self._generate_sync, vibe)

    def _resolve_model_dir(self) -> Path:
        if self._model_dir is not None:
            return self._model_dir
        base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", ""))
        if not base_dir:
            base_dir = Path.home() / ".cache" / "latentscore" / "models"
        return base_dir / _EXPRESSIVE_DIR

    def _download_expressive(self, model_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError("huggingface_hub is not installed") from exc

        _LOGGER.info(
            "Downloading model weights (~1.2GB). This happens once and is cached at %s.",
            model_dir.parent,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(_EXPRESSIVE_REPO, local_dir=str(model_dir))

    @functools.lru_cache(maxsize=1)
    def _load_model(self) -> Any:
        try:
            import mlx_lm  # type: ignore[import]
            import outlines  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("MLX dependencies not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError("MLX dependencies are not installed") from exc

        model_dir = self._resolve_model_dir()
        if not model_dir.exists():
            if not self._allow_download:
                raise ModelNotAvailableError(
                    f"Expressive model missing at {model_dir}. "
                    "Run `latentscore download expressive` or set LATENTSCORE_MODEL_DIR."
                )
            self._download_expressive(model_dir)

        loaded = mlx_lm.load(str(model_dir))
        model, tokenizer, *_ = loaded
        assert model is not None
        assert tokenizer is not None
        model_any: Any = model
        tokenizer_any: Any = tokenizer
        return outlines.from_mlxlm(model_any, tokenizer_any)

    def _build_prompt(self, vibe: str) -> str:
        return (
            f"{build_expressive_prompt()}\n\n"
            '<input description="The vibe/mood description to generate configs for">\n'
            f'Vibe: "{vibe}"\n'
            "</input>\n\n"
            "<output>\n"
        )

    def _generate_sync(self, vibe: str) -> MusicConfig:
        model = self._load_model()
        last_error: Exception | None = None
        for _ in range(self._max_retries):
            prompt = self._build_prompt(vibe)
            try:
                raw = model(prompt, output_type=MusicConfig, max_tokens=_LLM_MAX_TOKENS)
                return MusicConfig.model_validate_json(raw)
            except ValidationError as exc:
                _LOGGER.warning("Expressive model returned invalid JSON: %s", exc, exc_info=True)
                last_error = exc
            except Exception as exc:  # pragma: no cover - model provider errors
                _LOGGER.warning("Expressive model inference failed: %s", exc, exc_info=True)
                last_error = exc

        message = f"Failed to generate config for vibe {vibe!r}."
        raise LLMInferenceError(message) from last_error


def resolve_model(
    model: ModelSpec,
) -> ModelForGeneratingMusicConfig:
    model_obj: object = model
    match model_obj:
        case "expressive":
            return ExpressiveMlxModel()
        case "fast":
            return FastEmbeddingModel()
        case _ if _is_model(model):
            return model
        case _:
            raise ConfigGenerateError(f"Unknown model choice: {model!r}")
</file>

<file path="synth.py">
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""
Architecture:

1. Primitives: oscillators, envelopes, filters, effects
2. Patterns: pre-baked layer templates (bass, pad, melody, rhythm, texture, accent)
3. Assembler: config → audio conversion with V2 parameter transforms
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable, TypeAlias, cast

import numpy as np
import soundfile as sf  # type: ignore[import]
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.signal import butter, decimate, lfilter  # type: ignore[import]

from .config import (
    AccentStyle,
    AttackStyle,
    BassStyle,
    DensityLevel,
    GrainStyle,
    MelodyStyle,
    ModeName,
    PadStyle,
    RhythmStyle,
    RootNote,
    TextureStyle,
)

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100

# Root note frequencies (octave 4) - mapping proxy for immutability
NOTE_FREQS: Mapping[RootNote, float] = MappingProxyType(
    {
        "c": 261.63,
        "c#": 277.18,
        "d": 293.66,
        "d#": 311.13,
        "e": 329.63,
        "f": 349.23,
        "f#": 369.99,
        "g": 392.00,
        "g#": 415.30,
        "a": 440.00,
        "a#": 466.16,
        "b": 493.88,
    }
)

# Root to semitone offset
ROOT_SEMITONES: Mapping[RootNote, int] = MappingProxyType(
    {
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }
)

# Mode intervals (semitones from root) - tuples for JIT
MODE_INTERVALS: Mapping[ModeName, tuple[int, ...]] = MappingProxyType(
    {
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10),
        "dorian": (0, 2, 3, 5, 7, 9, 10),
        "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    }
)

# V2 parameter mappings
ATTACK_MULT: Mapping[AttackStyle, float] = MappingProxyType(
    {"soft": 2.5, "medium": 1.0, "sharp": 0.3}
)
GRAIN_OSC: Mapping[GrainStyle, str] = MappingProxyType(
    {"clean": "sine", "warm": "triangle", "gritty": "sawtooth"}
)

# Density → active layers (tuples)
DENSITY_LAYERS: Mapping[DensityLevel, tuple[str, ...]] = MappingProxyType(
    {
        2: ("bass", "pad"),
        3: ("bass", "pad", "melody"),
        4: ("bass", "pad", "melody", "rhythm"),
        5: ("bass", "pad", "melody", "rhythm", "texture"),
        6: ("bass", "pad", "melody", "rhythm", "texture", "accent"),
    }
)

FloatArray: TypeAlias = NDArray[np.float64]
OscFn: TypeAlias = Callable[[float, float, int, float], FloatArray]


# =============================================================================
# PART 1: SYNTHESIS PRIMITIVES
# =============================================================================


def freq_from_note(root: RootNote, semitones: int = 0, octave: int = 4) -> float:
    """Get frequency for a note."""
    root_value: RootNote = root
    if root_value not in NOTE_FREQS:
        raise ValueError(f"Unknown root note: {root}. Valid: {list(NOTE_FREQS.keys())}")
    base_freq = NOTE_FREQS[root_value]
    octave_shift = octave - 4
    return base_freq * (2**octave_shift) * (2 ** (semitones / 12))


def generate_sine(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate sine wave."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return amp * np.sin(2 * np.pi * freq * t)


def generate_triangle(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3
) -> FloatArray:
    """Generate triangle wave."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return amp * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - amp


def generate_sawtooth(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased sawtooth using 4-point PolyBLEP + oversampling."""

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = 2.0 * t - 1.0
    correction = np.zeros(num_samples_high)

    # 4-point PolyBLEP correction
    # Region 1: 0 <= t < dt
    m1 = t < dt
    t1 = t[m1] / dt
    correction[m1] = t1 * t1 * (2 * t1 - 3) + 1

    # Region 2: dt <= t < 2*dt
    m2 = (t >= dt) & (t < 2 * dt)
    t2 = t[m2] / dt - 1
    correction[m2] = t2 * t2 * (2 * t2 - 3)

    # Region 3: 1-2*dt < t <= 1-dt
    m3 = (t > 1 - 2 * dt) & (t <= 1 - dt)
    t3 = (t[m3] - 1) / dt + 1
    correction[m3] = t3 * t3 * (2 * t3 + 3)

    # Region 4: 1-dt < t < 1
    m4 = t > 1 - dt
    t4 = (t[m4] - 1) / dt
    correction[m4] = t4 * t4 * (2 * t4 + 3) + 1

    signal_high = naive - correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    output = amp * signal
    assert isinstance(output, np.ndarray)
    return cast(FloatArray, output)


def generate_square(
    freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 0.3, oversample: int = 2
) -> FloatArray:
    """Generate anti-aliased square wave using 4-point PolyBLEP + oversampling."""

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    t = np.linspace(0, duration * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = np.where(t < 0.5, 1.0, -1.0)
    correction = np.zeros(num_samples_high)

    def apply_4pt_blep(phase: FloatArray, corr: FloatArray, sign: float) -> None:
        """Apply 4-point PolyBLEP correction at discontinuity."""
        # Region 1: 0 <= phase < dt
        m1 = phase < dt
        t1 = phase[m1] / dt
        corr[m1] += sign * (t1 * t1 * (2 * t1 - 3) + 1)

        # Region 2: dt <= phase < 2*dt
        m2 = (phase >= dt) & (phase < 2 * dt)
        t2 = phase[m2] / dt - 1
        corr[m2] += sign * (t2 * t2 * (2 * t2 - 3))

        # Region 3: 1-2*dt < phase <= 1-dt
        m3 = (phase > 1 - 2 * dt) & (phase <= 1 - dt)
        t3 = (phase[m3] - 1) / dt + 1
        corr[m3] += sign * (t3 * t3 * (2 * t3 + 3))

        # Region 4: 1-dt < phase < 1
        m4 = phase > 1 - dt
        t4 = (phase[m4] - 1) / dt
        corr[m4] += sign * (t4 * t4 * (2 * t4 + 3) + 1)

    # Rising edge at phase = 0
    assert isinstance(t, np.ndarray)
    t_array = cast(FloatArray, t)
    apply_4pt_blep(t_array, correction, 1.0)

    # Falling edge at phase = 0.5
    t_shifted = (t + 0.5) % 1.0
    assert isinstance(t_shifted, np.ndarray)
    t_shifted_array = cast(FloatArray, t_shifted)
    apply_4pt_blep(t_shifted_array, correction, -1.0)

    signal_high = naive + correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    output = amp * signal
    assert isinstance(output, np.ndarray)
    return cast(FloatArray, output)


def generate_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 0.1) -> FloatArray:
    """Generate white noise."""
    return amp * np.random.randn(int(sr * duration))


OSC_FUNCTIONS: Mapping[str, OscFn] = MappingProxyType(
    {
        "sine": generate_sine,
        "triangle": generate_triangle,
        "sawtooth": generate_sawtooth,
        "square": generate_square,
    }
)


def apply_adsr(
    signal: FloatArray,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply ADSR envelope to signal."""
    # Minimum times to prevent clicks (5ms attack, 10ms release)
    attack = max(attack, 0.005)
    release = max(release, 0.01)

    total = len(signal)
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, total - a_samples - d_samples - r_samples)

    envelope = np.concatenate(
        (
            np.linspace(0, 1, max(1, a_samples)),
            np.linspace(1, sustain, max(1, d_samples)),
            np.ones(max(1, s_samples)) * sustain,
            np.linspace(sustain, 0, max(1, r_samples)),
        )
    )

    # Match length
    if len(envelope) < total:
        envelope = np.pad(envelope, (0, total - len(envelope)))
    else:
        envelope = envelope[:total]

    return signal * envelope


def apply_lowpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply lowpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    coeffs = butter(2, normalized, btype="low", output="ba")
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 2
    b_raw, a_raw = coeffs
    assert isinstance(b_raw, np.ndarray)
    assert isinstance(a_raw, np.ndarray)
    b = cast(FloatArray, b_raw)
    a = cast(FloatArray, a_raw)
    filtered = lfilter(b, a, signal)
    assert isinstance(filtered, np.ndarray)
    return cast(FloatArray, filtered)


def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply highpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    coeffs = butter(2, normalized, btype="high", output="ba")
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 2
    b_raw, a_raw = coeffs
    assert isinstance(b_raw, np.ndarray)
    assert isinstance(a_raw, np.ndarray)
    b = cast(FloatArray, b_raw)
    a = cast(FloatArray, a_raw)
    filtered = lfilter(b, a, signal)
    assert isinstance(filtered, np.ndarray)
    return cast(FloatArray, filtered)


def apply_delay(
    signal: FloatArray, delay_time: float, feedback: float, wet: float, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Apply delay effect."""
    delay_samples = int(delay_time * sr)
    output = signal.copy()

    for i in range(1, 5):  # 5 delay taps
        offset = delay_samples * i
        if offset < len(signal):
            delayed = np.zeros_like(signal)
            delayed[offset:] = signal[:-offset] * (feedback**i) * wet
            output += delayed

    return output


def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Simple reverb via multiple delays."""
    output = signal.copy()

    # Multiple delay lines at prime-ish intervals (tuple)
    delays = (0.029, 0.037, 0.041, 0.053, 0.067)

    for i, delay in enumerate(delays):
        delay_samples = int(delay * size * sr)
        if delay_samples < len(signal) and delay_samples > 0:
            reverb = np.zeros_like(signal)
            reverb[delay_samples:] = signal[:-delay_samples] * room * (0.7**i)
            output += reverb

    return output


def generate_lfo(duration: float, rate: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Generate LFO signal (0 to 1 range)."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)


def apply_humanize(signal: FloatArray, amount: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply subtle timing/amplitude humanization."""
    if amount <= 0:
        return signal

    # Subtle amplitude variation
    amp_lfo = 1.0 + (np.random.randn(len(signal)) * amount * 0.1)
    amp_lfo = np.clip(amp_lfo, 0.9, 1.1)

    return signal * amp_lfo


def add_note(signal: FloatArray, note: FloatArray, start_index: int) -> None:
    """Safely adds a note to the signal buffer, clipping if necessary."""
    if start_index >= len(signal):
        return

    end_index = start_index + len(note)

    if end_index <= len(signal):
        signal[start_index:end_index] += note
    else:
        # Clip the note to fit the remaining signal space
        available = len(signal) - start_index
        clipped = note[:available].copy()

        # Apply quick fade-out to prevent click from abrupt cutoff
        fade_samples = min(int(SAMPLE_RATE * 0.01), available // 4)  # 10ms max
        if fade_samples > 1:
            clipped[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        signal[start_index:] += clipped


# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================


class SynthParams(BaseModel):
    """Parameters passed to all synthesis functions."""

    root: RootNote = "c"
    mode: ModeName = "minor"
    brightness: float = 0.5
    space: float = 0.6
    duration: float = 16.0
    tempo: float = 0.35

    # V2 parameters
    motion: float = 0.5
    attack: AttackStyle = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: GrainStyle = "clean"

    model_config = ConfigDict(extra="forbid")

    @property
    def attack_mult(self) -> float:
        return ATTACK_MULT.get(self.attack, 1.0)

    @property
    def osc_type(self) -> str:
        return GRAIN_OSC.get(self.grain, "sine")

    @property
    def echo_mult(self) -> float:
        return self.echo / 0.5  # Normalize around 0.5

    @property
    def pan_width(self) -> float:
        return self.stereo * 0.5

    def get_scale_freq(self, degree: int, octave: int = 4) -> float:
        """Get frequency for a scale degree in the current mode."""
        intervals = MODE_INTERVALS.get(self.mode, MODE_INTERVALS["minor"])
        semitone = intervals[degree % len(intervals)] + (12 * (degree // len(intervals)))
        return freq_from_note(self.root, semitone, octave)


PatternFn: TypeAlias = Callable[[SynthParams], FloatArray]


# -----------------------------------------------------------------------------
# BASS PATTERNS
# -----------------------------------------------------------------------------


def bass_drone(params: SynthParams) -> FloatArray:
    """Sustained drone bass."""
    sr = SAMPLE_RATE
    dur = params.duration
    freq = params.get_scale_freq(0, 2)

    signal = generate_sine(freq, dur, sr, 0.35)
    signal = apply_lowpass(signal, 80 * params.brightness + 20, sr)
    signal = apply_adsr(signal, 2.0 * params.attack_mult, 0.5, 0.95, 3.0 * params.attack_mult, sr)
    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)

    return signal


def bass_sustained(params: SynthParams) -> FloatArray:
    """Long sustained notes with slow movement."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Root and fifth alternating slowly (tuple)
    pattern = (0, 0, 4, 0)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 2)
        start = int(i * note_dur * sr)

        note = generate_sine(freq, note_dur, sr, 0.32)
        note = apply_lowpass(note, 100 * params.brightness + 30, sr)
        note = apply_adsr(note, 0.8 * params.attack_mult, 0.3, 0.85, 1.5 * params.attack_mult, sr)

        add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.4, params.space, sr)
    return signal


def bass_pulsing(params: SynthParams) -> FloatArray:
    """Rhythmic pulsing bass."""
    sr = SAMPLE_RATE
    dur = params.duration

    # 8 pulses per cycle
    num_pulses = 8
    pulse_dur = dur / num_pulses
    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        freq = params.get_scale_freq(0, 2)
        start = int(i * pulse_dur * sr)

        note = generate_sine(freq, pulse_dur * 0.8, sr, 0.35)
        note = apply_lowpass(note, 90 * params.brightness + 20, sr)
        note = apply_adsr(note, 0.02 * params.attack_mult, 0.1, 0.6, 0.3 * params.attack_mult, sr)

        add_note(signal, note, start)

    return signal


def bass_walking(params: SynthParams) -> FloatArray:
    """Walking bass line."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Walking pattern (tuple)
    pattern = (0, 2, 4, 2, 0, 2, 4, 4)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 2)
        start = int(i * note_dur * sr)

        note = generate_triangle(freq, note_dur * 0.9, sr, 0.30)
        note = apply_lowpass(note, 120 * params.brightness + 40, sr)
        note = apply_adsr(note, 0.05 * params.attack_mult, 0.15, 0.7, 0.25 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.8, sr)
    return signal


def bass_fifth_drone(params: SynthParams) -> FloatArray:
    """Root + fifth drone."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Root
    root_freq = params.get_scale_freq(0, 2)
    root = generate_sine(root_freq, dur, sr, 0.28)
    root = apply_lowpass(root, 70 * params.brightness + 20, sr)
    root = apply_adsr(root, 2.5 * params.attack_mult, 0.5, 0.95, 3.0 * params.attack_mult, sr)

    # Fifth
    fifth_freq = params.get_scale_freq(4, 2)
    fifth = generate_sine(fifth_freq, dur, sr, 0.18)
    fifth = apply_lowpass(fifth, 100 * params.brightness + 30, sr)
    fifth = apply_adsr(fifth, 3.0 * params.attack_mult, 0.5, 0.9, 3.0 * params.attack_mult, sr)

    signal = root + fifth
    signal = apply_reverb(signal, params.space * 0.5, params.space, sr)
    return signal


def bass_sub_pulse(params: SynthParams) -> FloatArray:
    """Deep sub-bass pulse."""
    sr = SAMPLE_RATE
    dur = params.duration

    freq = params.get_scale_freq(0, 1)  # Very low octave

    # Slow pulse (4 per cycle)
    num_pulses = 4
    pulse_dur = dur / num_pulses
    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        start = int(i * pulse_dur * sr)

        note = generate_sine(freq, pulse_dur * 0.95, sr, 0.4)
        note = apply_lowpass(note, 50, sr)
        note = apply_adsr(note, 0.3 * params.attack_mult, 0.2, 0.9, 0.8 * params.attack_mult, sr)

        add_note(signal, note, start)

    return signal


BASS_PATTERNS: Mapping[BassStyle, PatternFn] = MappingProxyType(
    {
        "drone": bass_drone,
        "sustained": bass_sustained,
        "pulsing": bass_pulsing,
        "walking": bass_walking,
        "fifth_drone": bass_fifth_drone,
        "sub_pulse": bass_sub_pulse,
        "octave": bass_sustained,
        "arp_bass": bass_pulsing,
    }
)


# -----------------------------------------------------------------------------
# PAD PATTERNS
# -----------------------------------------------------------------------------


def pad_warm_slow(params: SynthParams) -> FloatArray:
    """Warm, slowly evolving pad."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    signal = np.zeros(int(sr * dur))

    # Stack root, 3rd, 5th (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 3)
        tone = osc(freq, dur, sr, 0.15)

        # Slow filter movement (affected by motion)
        lfo_rate = 0.1 / (params.motion + 0.1)
        lfo = generate_lfo(dur, lfo_rate, sr)

        # Apply moving filter
        base_cutoff = 300 * params.brightness + 100
        tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
        tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
        tone = tone_low * (1 - lfo) + tone_high * lfo

        tone = apply_adsr(tone, 1.5 * params.attack_mult, 0.8, 0.85, 2.5 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_humanize(signal, params.human, sr)

    return signal


def pad_dark_sustained(params: SynthParams) -> FloatArray:
    """Dark, heavy sustained pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Minor chord voicing (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 3)
        tone = generate_sawtooth(freq, dur, sr, 0.12)
        tone = apply_lowpass(tone, 200 * params.brightness + 80, sr)
        tone = apply_adsr(tone, 2.0 * params.attack_mult, 1.0, 0.9, 3.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    return signal


def pad_cinematic(params: SynthParams) -> FloatArray:
    """Big, cinematic pad with movement."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Wider voicing with octave doubling (tuple of tuples)
    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for degree, octave in voicings:
        freq = params.get_scale_freq(degree, octave)

        # Mix oscillators
        tone = generate_sawtooth(freq, dur, sr, 0.08)
        tone += generate_triangle(freq * 1.002, dur, sr, 0.06)  # Slight detune

        tone = apply_lowpass(tone, 400 * params.brightness + 150, sr)
        tone = apply_adsr(tone, 1.8 * params.attack_mult, 0.8, 0.88, 2.8 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * params.echo_mult, 0.3 * params.echo_mult, sr)

    return signal


def pad_thin_high(params: SynthParams) -> FloatArray:
    """Thin, high pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    for degree in (0, 4):  # Just root and fifth, high
        freq = params.get_scale_freq(degree, 4)
        tone = generate_sine(freq, dur, sr, 0.12)
        tone = apply_lowpass(tone, 800 * params.brightness + 200, sr)
        tone = apply_highpass(tone, 200, sr)
        tone = apply_adsr(tone, 1.2 * params.attack_mult, 0.6, 0.8, 2.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.75, params.space * 0.95, sr)
    return signal


def pad_ambient_drift(params: SynthParams) -> FloatArray:
    """Slowly drifting ambient pad."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Evolving chord (tuple of tuples)
    chord_dur = dur / 4
    chord_progressions = (
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 2, 4),
    )

    for i, chord in enumerate(chord_progressions):
        start = int(i * chord_dur * sr)

        for degree in chord:
            freq = params.get_scale_freq(degree, 3)
            tone = generate_sine(freq, chord_dur * 1.2, sr, 0.14)  # Overlap
            tone = apply_lowpass(tone, 350 * params.brightness + 100, sr)
            tone = apply_adsr(
                tone, 1.5 * params.attack_mult, 0.5, 0.85, 2.0 * params.attack_mult, sr
            )

            add_note(signal, tone, start)

    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)
    signal = apply_delay(signal, 0.5, 0.4 * params.echo_mult, 0.35 * params.echo_mult, sr)

    return signal


def pad_stacked_fifths(params: SynthParams) -> FloatArray:
    """Fifths stacked for a powerful sound."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    signal = np.zeros(int(sr * dur))

    # Stack fifths (tuple of tuples)
    voicings = ((0, 3), (4, 3), (0, 4), (4, 4))
    for degree, octave in voicings:
        freq = params.get_scale_freq(degree, octave)
        tone = osc(freq, dur, sr, 0.10)
        tone = apply_lowpass(tone, 500 * params.brightness + 150, sr)
        tone = apply_adsr(tone, 1.3 * params.attack_mult, 0.7, 0.88, 2.2 * params.attack_mult, sr)
        signal += tone

    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.85, sr)
    return signal


PAD_PATTERNS: Mapping[PadStyle, PatternFn] = MappingProxyType(
    {
        "warm_slow": pad_warm_slow,
        "dark_sustained": pad_dark_sustained,
        "cinematic": pad_cinematic,
        "thin_high": pad_thin_high,
        "ambient_drift": pad_ambient_drift,
        "stacked_fifths": pad_stacked_fifths,
        "bright_open": pad_thin_high,
    }
)


# -----------------------------------------------------------------------------
# MELODY PATTERNS
# -----------------------------------------------------------------------------


def melody_contemplative(params: SynthParams) -> FloatArray:
    """Slow, contemplative melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Sparse melody pattern (tuple), -1 = rest
    pattern = (0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.5, sr, 0.18)
        note = apply_lowpass(note, 800 * params.brightness + 200, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.3, 0.6, 0.8 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.35, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def melody_rising(params: SynthParams) -> FloatArray:
    """Ascending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Rising pattern (tuple)
    pattern = (0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.8, sr, 0.16)
        note = apply_lowpass(note, 900 * params.brightness + 250, sr)
        note = apply_adsr(note, 0.08 * params.attack_mult, 0.25, 0.55, 0.9 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.33, 0.35 * params.echo_mult, 0.28 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.55, params.space * 0.75, sr)

    return signal


def melody_falling(params: SynthParams) -> FloatArray:
    """Descending melodic line."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Falling pattern (tuple)
    pattern = (6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 1.6, sr, 0.17)
        note = apply_lowpass(note, 850 * params.brightness + 220, sr)
        note = apply_adsr(note, 0.1 * params.attack_mult, 0.28, 0.52, 0.85 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.38, 0.38 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.58, params.space * 0.78, sr)

    return signal


def melody_minimal(params: SynthParams) -> FloatArray:
    """Very sparse, minimal melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_sine)

    # Extremely sparse (tuple)
    pattern = (4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = params.get_scale_freq(degree, 5)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 2.5, sr, 0.20)
        note = apply_lowpass(note, 600 * params.brightness + 150, sr)
        note = apply_adsr(note, 0.15 * params.attack_mult, 0.4, 0.5, 1.2 * params.attack_mult, sr)
        note = apply_humanize(note, params.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.5, 0.45 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.7, params.space * 0.9, sr)

    return signal


def melody_ornamental(params: SynthParams) -> FloatArray:
    """Ornamental melody with grace notes."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Pattern with ornaments (tuples)
    main_notes = (0, 4, 2, 0)
    grace_offsets = (2, 5, 3, 2)
    note_dur = dur / (len(main_notes) * 2)
    signal = np.zeros(int(sr * dur))

    for i, (main, grace) in enumerate(zip(main_notes, grace_offsets)):
        start = int(i * 2 * note_dur * sr)

        # Grace note (quick)
        grace_freq = params.get_scale_freq(grace, 5)
        grace_note = osc(grace_freq, note_dur * 0.15, sr, 0.10)
        grace_note = apply_lowpass(grace_note, 1000 * params.brightness, sr)
        grace_note = apply_adsr(grace_note, 0.01, 0.05, 0.3, 0.1, sr)

        # Main note
        main_freq = params.get_scale_freq(main, 5)
        main_note = osc(main_freq, note_dur * 1.5, sr, 0.18)
        main_note = apply_lowpass(main_note, 900 * params.brightness + 200, sr)
        main_note = apply_adsr(
            main_note, 0.08 * params.attack_mult, 0.3, 0.55, 0.8 * params.attack_mult, sr
        )
        main_note = apply_humanize(main_note, params.human, sr)

        grace_start = start
        main_start = start + int(note_dur * 0.15 * sr)

        add_note(signal, grace_note, grace_start)
        add_note(signal, main_note, main_start)

    signal = apply_delay(signal, 0.3, 0.35 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)

    return signal


def melody_arp(params: SynthParams) -> FloatArray:
    """Arpeggiated melody."""
    sr = SAMPLE_RATE
    dur = params.duration
    osc = OSC_FUNCTIONS.get(params.osc_type, generate_triangle)

    # Fast arpeggio pattern (tuple * 8)
    base_pattern = (0, 2, 4, 2)
    pattern = base_pattern * 8
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = params.get_scale_freq(degree, 4)
        start = int(i * note_dur * sr)

        note = osc(freq, note_dur * 0.8, sr, 0.14)
        note = apply_lowpass(note, 1200 * params.brightness + 300, sr)
        note = apply_adsr(note, 0.02 * params.attack_mult, 0.1, 0.4, 0.15 * params.attack_mult, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.25, 0.3 * params.echo_mult, 0.25 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.4, params.space * 0.6, sr)

    return signal


MELODY_PATTERNS: Mapping[MelodyStyle, PatternFn] = MappingProxyType(
    {
        "contemplative": melody_contemplative,
        "contemplative_minor": melody_contemplative,
        "rising": melody_rising,
        "falling": melody_falling,
        "minimal": melody_minimal,
        "ornamental": melody_ornamental,
        "arp_melody": melody_arp,
        "call_response": melody_contemplative,
        "heroic": melody_rising,
    }
)


# -----------------------------------------------------------------------------
# RHYTHM PATTERNS
# -----------------------------------------------------------------------------


def rhythm_none(params: SynthParams) -> FloatArray:
    """No rhythm."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def rhythm_minimal(params: SynthParams) -> FloatArray:
    """Minimal rhythm - just occasional hits."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Sparse kicks (tuple)
    pattern = (1, 0, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        # Kick: sine with pitch envelope
        kick_dur = 0.15
        kick = generate_sine(60, kick_dur, sr, 0.35)
        pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 150, sr)
        kick = apply_humanize(kick, params.human, sr)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.3, params.space * 0.5, sr)
    return signal


def rhythm_heartbeat(params: SynthParams) -> FloatArray:
    """Heartbeat-like rhythm."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Double-hit pattern (tuple)
    pattern = (1, 1, 0, 0, 0, 0, 0, 0) * 2
    hit_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, hit in enumerate(pattern):
        if not hit:
            continue

        start = int(i * hit_dur * sr)

        kick_dur = 0.12
        kick = generate_sine(55, kick_dur, sr, 0.32)
        pitch_env = np.exp(-np.linspace(0, 10, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 120, sr)

        add_note(signal, kick, start)

    return signal


def rhythm_soft_four(params: SynthParams) -> FloatArray:
    """Soft four-on-the-floor."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Kick on every beat
    num_beats = 8
    beat_dur = dur / num_beats
    signal = np.zeros(int(sr * dur))

    for i in range(num_beats):
        start = int(i * beat_dur * sr)

        # Soft kick
        kick_dur = 0.18
        kick = generate_sine(50, kick_dur, sr, 0.28)
        pitch_env = np.exp(-np.linspace(0, 6, len(kick)))
        kick = kick * pitch_env
        kick = apply_lowpass(kick, 100, sr)
        kick = apply_humanize(kick, params.human, sr)

        add_note(signal, kick, start)

    signal = apply_reverb(signal, params.space * 0.25, params.space * 0.4, sr)
    return signal


def rhythm_hats_only(params: SynthParams) -> FloatArray:
    """Just hi-hats."""
    sr = SAMPLE_RATE
    dur = params.duration

    # 16th note hats
    num_hits = 32
    hit_dur = dur / num_hits
    signal = np.zeros(int(sr * dur))

    for i in range(num_hits):
        start = int(i * hit_dur * sr)

        # Hi-hat: filtered noise
        hat_dur = 0.05
        hat = generate_noise(hat_dur, sr, 0.08)
        hat = apply_highpass(hat, 6000, sr)
        hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.03, sr)
        hat = apply_humanize(hat, params.human, sr)

        add_note(signal, hat, start)

    return signal


def rhythm_electronic(params: SynthParams) -> FloatArray:
    """Electronic beat."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))
    beat_dur = dur / 16

    # Patterns (tuples)
    kick_pattern = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    hat_pattern = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)

    for i in range(16):
        start = int(i * beat_dur * sr)

        if kick_pattern[i]:
            kick = generate_sine(45, 0.2, sr, 0.35)
            pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 100, sr)
            add_note(signal, kick, start)

        if hat_pattern[i]:
            hat = generate_noise(0.06, sr, 0.06)
            hat = apply_highpass(hat, 7000, sr)
            hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.04, sr)
            add_note(signal, hat, start)

    return signal


RHYTHM_PATTERNS: Mapping[RhythmStyle, PatternFn] = MappingProxyType(
    {
        "none": rhythm_none,
        "minimal": rhythm_minimal,
        "heartbeat": rhythm_heartbeat,
        "soft_four": rhythm_soft_four,
        "hats_only": rhythm_hats_only,
        "electronic": rhythm_electronic,
        "kit_light": rhythm_minimal,
        "kit_medium": rhythm_soft_four,
        "military": rhythm_soft_four,
        "tabla_essence": rhythm_heartbeat,
        "brush": rhythm_minimal,
    }
)


# -----------------------------------------------------------------------------
# TEXTURE PATTERNS
# -----------------------------------------------------------------------------


def texture_none(params: SynthParams) -> FloatArray:
    """No texture."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def texture_shimmer(params: SynthParams) -> FloatArray:
    """High, shimmering texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # High sine clusters with amplitude modulation (tuple)
    for degree in (0, 2, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.04)

        # Amplitude modulation for shimmer
        lfo_rate = 2.0 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.5 + 0.5 * lfo)

        tone = apply_adsr(tone, 0.5 * params.attack_mult, 0.3, 0.8, 1.5 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.4, 0.5 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.8, params.space, sr)

    return signal


def texture_shimmer_slow(params: SynthParams) -> FloatArray:
    """Slow, gentle shimmer."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    for degree in (0, 4):
        freq = params.get_scale_freq(degree, 6)
        tone = generate_sine(freq, dur, sr, 0.035)

        # Very slow amplitude modulation
        lfo_rate = 0.5 / (params.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.4 + 0.6 * lfo)

        tone = apply_adsr(tone, 1.0 * params.attack_mult, 0.5, 0.85, 2.0 * params.attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.5, 0.55 * params.echo_mult, 0.45 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.85, params.space, sr)

    return signal


def texture_vinyl_crackle(params: SynthParams) -> FloatArray:
    """Vinyl crackle texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Sparse noise impulses
    signal = np.zeros(int(sr * dur))

    num_crackles = int(dur * 20)  # ~20 crackles per second

    for _ in range(num_crackles):
        pos = np.random.randint(0, len(signal) - 100)
        crackle = generate_noise(0.002, sr, np.random.uniform(0.01, 0.04))
        crackle = apply_highpass(crackle, 2000, sr)

        add_note(signal, crackle, pos)

    # Soft background hiss
    hiss = generate_noise(dur, sr, 0.008)
    hiss = apply_lowpass(hiss, 8000, sr)
    hiss = apply_highpass(hiss, 1000, sr)
    signal += hiss

    return signal


def texture_breath(params: SynthParams) -> FloatArray:
    """Breathing texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    # Filtered noise with slow envelope
    signal = generate_noise(dur, sr, 0.06)

    # Bandpass around a note frequency
    freq = params.get_scale_freq(0, 3)
    signal = apply_lowpass(signal, freq * 2, sr)
    signal = apply_highpass(signal, freq * 0.5, sr)

    # Breathing envelope (slow LFO)
    breath_rate = 0.2 / (params.motion + 0.1)
    lfo = generate_lfo(dur, breath_rate, sr)
    signal = signal * lfo

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)

    return signal


def texture_stars(params: SynthParams) -> FloatArray:
    """Twinkling stars texture."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Random high plinks
    num_stars = int(dur * 3)  # ~3 per second

    # Scale degrees for stars (tuple)
    star_degrees = (0, 2, 4, 5)

    for _ in range(num_stars):
        pos = np.random.randint(0, len(signal) - sr)

        degree = np.random.choice(star_degrees)
        freq = params.get_scale_freq(degree, 6)

        star = generate_sine(freq, 0.3, sr, np.random.uniform(0.02, 0.05))
        star = apply_adsr(star, 0.01, 0.1, 0.1, 0.2, sr)

        add_note(signal, star, pos)

    signal = apply_delay(signal, 0.4, 0.5 * params.echo_mult, 0.4 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.9, params.space, sr)

    return signal


TEXTURE_PATTERNS: Mapping[TextureStyle, PatternFn] = MappingProxyType(
    {
        "none": texture_none,
        "shimmer": texture_shimmer,
        "shimmer_slow": texture_shimmer_slow,
        "vinyl_crackle": texture_vinyl_crackle,
        "breath": texture_breath,
        "stars": texture_stars,
        "glitch": texture_shimmer,
        "noise_wash": texture_breath,
        "crystal": texture_stars,
        "pad_whisper": texture_breath,
    }
)


# -----------------------------------------------------------------------------
# ACCENT PATTERNS
# -----------------------------------------------------------------------------


def accent_none(params: SynthParams) -> FloatArray:
    """No accent."""
    return np.zeros(int(SAMPLE_RATE * params.duration))


def accent_bells(params: SynthParams) -> FloatArray:
    """Bell-like accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Sparse bell hits (tuple)
    pattern = (0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 5)

        # Bell: mix of harmonics with fast decay
        bell_dur = 0.8
        bell = generate_sine(freq, bell_dur, sr, 0.12)
        bell += generate_sine(freq * 2.0, bell_dur, sr, 0.06)
        bell += generate_sine(freq * 3.0, bell_dur, sr, 0.03)
        bell = apply_adsr(bell, 0.005 * params.attack_mult, 0.2, 0.1, 0.6 * params.attack_mult, sr)
        bell = apply_humanize(bell, params.human, sr)

        add_note(signal, bell, start)

    signal = apply_reverb(signal, params.space * 0.6, params.space * 0.8, sr)
    return signal


def accent_pluck(params: SynthParams) -> FloatArray:
    """Plucked string accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Pattern (tuple)
    pattern = (0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        freq = params.get_scale_freq(degree, 4)

        # Pluck: sharp attack, quick decay
        pluck = generate_triangle(freq, 0.5, sr, 0.15)
        pluck = apply_lowpass(pluck, 1500 * params.brightness + 400, sr)
        pluck = apply_adsr(
            pluck, 0.003 * params.attack_mult, 0.15, 0.05, 0.4 * params.attack_mult, sr
        )
        pluck = apply_humanize(pluck, params.human, sr)

        add_note(signal, pluck, start)

    signal = apply_reverb(signal, params.space * 0.5, params.space * 0.7, sr)
    return signal


def accent_chime(params: SynthParams) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = params.duration

    signal = np.zeros(int(sr * dur))

    # Random chime hits
    num_chimes = int(dur * 1.5)

    # Chime degrees (tuple)
    chime_degrees = (0, 2, 4, 5, 6)

    for _ in range(num_chimes):
        pos = np.random.randint(0, len(signal) - sr)

        degree = np.random.choice(chime_degrees)
        freq = params.get_scale_freq(degree, 5)

        chime_dur = 1.2
        chime = generate_sine(freq, chime_dur, sr, np.random.uniform(0.06, 0.12))
        chime += generate_sine(freq * 2.0, chime_dur, sr, 0.03)
        chime = apply_adsr(chime, 0.002, 0.3, 0.05, 0.9, sr)

        add_note(signal, chime, pos)

    signal = apply_delay(signal, 0.3, 0.4 * params.echo_mult, 0.3 * params.echo_mult, sr)
    signal = apply_reverb(signal, params.space * 0.75, params.space * 0.9, sr)

    return signal


ACCENT_PATTERNS: Mapping[AccentStyle, PatternFn] = MappingProxyType(
    {
        "none": accent_none,
        "bells": accent_bells,
        "bells_dense": accent_bells,
        "pluck": accent_pluck,
        "chime": accent_chime,
        "blip": accent_bells,
        "blip_random": accent_chime,
        "brass_hit": accent_bells,
        "wind": accent_chime,
        "arp_accent": accent_pluck,
        "piano_note": accent_pluck,
    }
)


# =============================================================================
# PART 3: ASSEMBLER - CONFIG → AUDIO
# =============================================================================


class MusicConfig(BaseModel):
    """Complete V1/V2 configuration."""

    tempo: float = 0.35
    root: RootNote = "c"
    mode: ModeName = "minor"
    brightness: float = 0.5
    space: float = 0.6
    density: DensityLevel = 5

    # Layer selections
    bass: BassStyle = "drone"
    pad: PadStyle = "warm_slow"
    melody: MelodyStyle = "contemplative"
    rhythm: RhythmStyle = "minimal"
    texture: TextureStyle = "shimmer"
    accent: AccentStyle = "bells"

    # V2 parameters
    motion: float = 0.5
    attack: AttackStyle = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: GrainStyle = "clean"

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _flatten_layers(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        merged = dict(data)
        layers = merged.pop("layers", None)
        if isinstance(layers, Mapping):
            for key, value in layers.items():
                merged.setdefault(key, value)
        return merged

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MusicConfig":
        """Create config from dict (e.g., from JSON)."""
        return cls.model_validate(d)


def assemble(config: MusicConfig, duration: float = 16.0, normalize: bool = True) -> FloatArray:
    """
    Assemble all layers into final audio.

    This is the core function that converts a config into audio.

    Args:
        config: MusicConfig object
        duration: Duration in seconds
        normalize: If True, maximizes volume. Set False for automation chunks
                   to preserve relative dynamics.
    """
    sr = SAMPLE_RATE

    # Build synth params
    params = SynthParams(
        root=config.root,
        mode=config.mode,
        brightness=config.brightness,
        space=config.space,
        duration=duration,
        tempo=config.tempo,
        motion=config.motion,
        attack=config.attack,
        stereo=config.stereo,
        depth=config.depth,
        echo=config.echo,
        human=config.human,
        grain=config.grain,
    )

    # Determine active layers based on density
    active_layers = DENSITY_LAYERS.get(config.density, DENSITY_LAYERS[5])

    # Initialize output
    output = np.zeros(int(sr * duration))

    # Generate each layer
    if "bass" in active_layers:
        bass_fn = BASS_PATTERNS.get(config.bass, bass_drone)
        output += bass_fn(params)

    if config.depth:
        # Add sub-bass layer
        sub_params = params.model_copy(update={"duration": duration})
        output += bass_sub_pulse(sub_params) * 0.6

    if "pad" in active_layers:
        pad_fn = PAD_PATTERNS.get(config.pad, pad_warm_slow)
        output += pad_fn(params)

    if "melody" in active_layers:
        melody_fn = MELODY_PATTERNS.get(config.melody, melody_contemplative)
        output += melody_fn(params)

    if "rhythm" in active_layers and config.rhythm != "none":
        rhythm_fn = RHYTHM_PATTERNS.get(config.rhythm, rhythm_none)
        output += rhythm_fn(params)

    if "texture" in active_layers and config.texture != "none":
        texture_fn = TEXTURE_PATTERNS.get(config.texture, texture_none)
        output += texture_fn(params)

    if "accent" in active_layers and config.accent != "none":
        accent_fn = ACCENT_PATTERNS.get(config.accent, accent_none)
        output += accent_fn(params)

    # Normalize only if requested
    if normalize:
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.85

    return output


def config_to_audio(config: MusicConfig, output_path: str, duration: float = 16.0) -> str:
    """
    Convert a MusicConfig to an audio file.

    Args:
        config: MusicConfig object
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    audio = assemble(config, duration)
    assert callable(sf.write)
    write_audio = sf.write
    write_audio(output_path, audio, SAMPLE_RATE)  # type: ignore[reportUnknownMemberType]
    return output_path


def dict_to_audio(
    config_dict: Mapping[str, Any],
    output_path: str,
    duration: float = 16.0,
) -> str:
    """
    Convert a config dict (from JSON) directly to audio.

    Args:
        config_dict: Mapping with config values
        output_path: Path to save the WAV file
        duration: Duration in seconds

    Returns:
        Path to the saved file
    """
    config = MusicConfig.from_dict(config_dict)
    return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 4: CONVENIENCE FUNCTIONS
# =============================================================================


def generate_from_vibe(vibe: str, output_path: str = "output.wav", duration: float = 16.0) -> str:
    """
    Generate audio from a vibe description.

    This is a placeholder - in production, this would call an LLM to generate the config.
    For now, uses simple keyword matching.
    """
    vibe_lower = vibe.lower()

    # Simple keyword-based config generation
    config = MusicConfig()

    # Root/mode selection
    if any(w in vibe_lower for w in ("dark", "sad", "night", "mysterious")):
        config.root = "d"
        config.mode = "dorian"
        config.brightness = 0.35
    elif any(w in vibe_lower for w in ("happy", "bright", "joy")):
        config.root = "c"
        config.mode = "major"
        config.brightness = 0.65
    elif any(w in vibe_lower for w in ("epic", "cinematic", "powerful")):
        config.root = "d"
        config.mode = "minor"
        config.brightness = 0.5
        config.depth = True
    elif any(w in vibe_lower for w in ("indian", "spiritual", "meditation")):
        config.root = "d"
        config.mode = "dorian"
        config.human = 0.2
        config.grain = "warm"
    else:
        config.root = "c"
        config.mode = "minor"

    # Tempo
    if any(w in vibe_lower for w in ("slow", "calm", "meditation")):
        config.tempo = 0.30
    elif any(w in vibe_lower for w in ("fast", "energy", "drive")):
        config.tempo = 0.45
    else:
        config.tempo = 0.36

    # Space
    if any(w in vibe_lower for w in ("vast", "space", "underwater", "cave")):
        config.space = 0.85
        config.echo = 0.75
    elif any(w in vibe_lower for w in ("intimate", "close")):
        config.space = 0.4
        config.echo = 0.3

    # Layers based on keywords
    if any(w in vibe_lower for w in ("electronic", "synth")):
        config.rhythm = "electronic"
        config.attack = "sharp"
    elif any(w in vibe_lower for w in ("ambient", "peaceful")):
        config.rhythm = "none"
        config.attack = "soft"

    if any(w in vibe_lower for w in ("minimal", "sparse")):
        config.density = 3
        config.melody = "minimal"
    elif any(w in vibe_lower for w in ("rich", "full", "lush")):
        config.density = 6

    return config_to_audio(config, output_path, duration)


# =============================================================================
# PART 5: TRANSITION FUNCTIONS
# =============================================================================


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 returns a, t=1 returns b"""
    return a + (b - a) * t


def _clamp_density(value: float) -> DensityLevel:
    raw = int(round(value))
    match raw:
        case 2:
            return 2
        case 3:
            return 3
        case 4:
            return 4
        case 5:
            return 5
        case 6:
            return 6
        case _:
            return 2 if raw < 2 else 6


def interpolate_configs(config_a: MusicConfig, config_b: MusicConfig, t: float) -> MusicConfig:
    """
    Interpolate between two configs.
    t=0.0 → config_a
    t=1.0 → config_b
    """
    return MusicConfig(
        tempo=lerp(config_a.tempo, config_b.tempo, t),
        root=config_a.root if t < 0.5 else config_b.root,
        mode=config_a.mode if t < 0.5 else config_b.mode,
        brightness=lerp(config_a.brightness, config_b.brightness, t),
        space=lerp(config_a.space, config_b.space, t),
        density=_clamp_density(lerp(config_a.density, config_b.density, t)),
        # Layer selections: staggered switching
        bass=config_a.bass if t < 0.4 else config_b.bass,
        pad=config_a.pad if t < 0.5 else config_b.pad,
        melody=config_a.melody if t < 0.6 else config_b.melody,
        rhythm=config_a.rhythm if t < 0.5 else config_b.rhythm,
        texture=config_a.texture if t < 0.7 else config_b.texture,
        accent=config_a.accent if t < 0.8 else config_b.accent,
        # V2 parameters
        motion=lerp(config_a.motion, config_b.motion, t),
        attack=config_a.attack if t < 0.5 else config_b.attack,
        stereo=lerp(config_a.stereo, config_b.stereo, t),
        depth=config_a.depth if t < 0.5 else config_b.depth,
        echo=lerp(config_a.echo, config_b.echo, t),
        human=lerp(config_a.human, config_b.human, t),
        grain=config_a.grain if t < 0.5 else config_b.grain,
    )


def morph_audio(
    config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0, segments: int = 8
) -> FloatArray:
    """
    Generate audio that morphs from config_a to config_b over duration.
    """
    segment_duration = duration / segments
    output = []

    for i in range(segments):
        t = i / (segments - 1) if segments > 1 else 0.0
        interpolated = interpolate_configs(config_a, config_b, t)
        # Don't normalize individual segments — preserve relative dynamics
        segment = assemble(interpolated, segment_duration, normalize=False)
        output.append(segment)

    result = np.concatenate(output)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result


def crossfade(audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int) -> FloatArray:
    """Crossfade between two audio arrays at the midpoint."""
    min_len = min(len(audio_a), len(audio_b))
    mid = min_len // 2
    half_cf = crossfade_samples // 2

    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)

    result = np.concatenate(
        (
            audio_a[: mid - half_cf],
            audio_a[mid - half_cf : mid + half_cf] * fade_out
            + audio_b[mid - half_cf : mid + half_cf] * fade_in,
            audio_b[mid + half_cf : min_len],
        )
    )

    return result


def transition(config_a: MusicConfig, config_b: MusicConfig, duration: float = 60.0) -> FloatArray:
    """Generate with crossfade transition."""
    # Don't normalize individual tracks — normalize after crossfade
    audio_a = assemble(config_a, duration, normalize=False)
    audio_b = assemble(config_b, duration, normalize=False)

    crossfade_duration = 4.0  # seconds
    crossfade_samples = int(crossfade_duration * SAMPLE_RATE)

    result = crossfade(audio_a, audio_b, crossfade_samples)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result


def generate_tween_with_automation(
    config_a: MusicConfig,
    config_b: MusicConfig,
    duration: float = 120.0,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.05,
) -> FloatArray:
    """
    Generate audio with automated parameters using Cached Block Processing.

    Performance: ~8x faster than per-chunk generation.
    Trade-off: Automation updates occur every 16s (Pattern Length) instead of every 2s.
    """
    sr = SAMPLE_RATE
    num_chunks = int(np.ceil(duration / chunk_seconds))  # ceil ensures we cover full duration
    overlap_samples = int(overlap_seconds * sr)

    PATTERN_LEN = 16.0
    chunks_per_pattern = int(PATTERN_LEN / chunk_seconds)

    chunk_len_sec = chunk_seconds + overlap_seconds
    chunk_samples = int(chunk_len_sec * sr)

    # Initialize output buffer
    output = np.zeros(int(duration * sr))

    cached_pattern: FloatArray | None = None
    cached_pattern_idx: int = -1

    for i in range(num_chunks):
        # 1. Determine which 16s Block we are in
        pattern_idx = i // chunks_per_pattern

        # 2. Check Cache
        if pattern_idx != cached_pattern_idx:
            # Interpolate parameters for this specific 16s block
            # Note: This "steps" the parameters every 16s.
            t = (pattern_idx * chunks_per_pattern) / max(1, num_chunks - 1)
            t = np.clip(t, 0.0, 1.0)
            t_eased = 0.5 - 0.5 * np.cos(t * np.pi)

            config = interpolate_configs(config_a, config_b, t_eased)

            # Generate the FULL 16s block
            # normalize=False preserves relative dynamics between blocks
            cached_pattern = assemble(config, PATTERN_LEN, normalize=False)
            cached_pattern_idx = pattern_idx

        # Ensure cache exists (typing safety)
        if cached_pattern is None:
            continue

        # 3. Slice the 2s window + overlap
        local_chunk_idx = i % chunks_per_pattern

        # Calculate indices relative to the cached pattern
        start_idx = int(local_chunk_idx * chunk_seconds * sr)
        end_idx = start_idx + chunk_samples

        # Handle wrapping (Circular Buffer logic)
        pattern_len_samples = len(cached_pattern)

        if end_idx <= pattern_len_samples:
            chunk = cached_pattern[start_idx:end_idx].copy()
        else:
            # Wrap around to start
            # part_a: from start_idx to end of buffer
            # part_b: from 0 to remainder
            part_a = cached_pattern[start_idx:]
            remainder = end_idx - pattern_len_samples
            part_b = cached_pattern[:remainder]
            chunk = np.concatenate((part_a, part_b))

        # 4. Apply Crossfade Envelopes (Overlap-Add)
        if i > 0:
            chunk[:overlap_samples] *= np.linspace(0.0, 1.0, overlap_samples)

        if i < num_chunks - 1:
            chunk[-overlap_samples:] *= np.linspace(1.0, 0.0, overlap_samples)

        # 5. Add to Main Output
        out_start = int(i * chunk_seconds * sr)
        out_end = min(out_start + len(chunk), len(output))
        available = out_end - out_start

        if available > 0:
            output[out_start:out_end] += chunk[:available]

    # Global Normalization
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.85

    return output


# =============================================================================
# MAIN - Demo
# =============================================================================

if __name__ == "__main__":
    print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
    print("=" * 50)

    # Example 3: Bubblegum, sad, dead (from llm_to_synth.py, see @file_context_1)
    import json

    # These examples mirror the demo config blocks captured in @file_context_0:
    demo_configs = [
        {
            # "justification": "The 'Bubblegum'-style sound is characterized by a bright, slightly distorted, and playful melody with a prominent bass line and a smooth, evolving pad.",
            "tempo": 0.5,
            "root": "a",
            "mode": "major",
            "brightness": 0.75,
            "space": 0.5,
            "density": 2,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "none",
            "texture": "shimmer",
            "accent": "bells",
            "motion": 0.5,
            "attack": "medium",
            "stereo": 0.5,
            "depth": False,
            "echo": 0.25,
            "human": 0.5,
            "grain": "clean",
        },
        {
            # "justification": "The sad vibe is characterized by a low-key sound and a sense of melancholy. The low intensity and muted tones create a feeling of quiet sorrow. The 'low' intensity of the bass and the 'dark' tone of the pad support this feeling. The 'soft' attack and 'warm' tone of the pad create a sense of longing and a feeling of sadness.",
            "tempo": 0.25,
            "root": "e",
            "mode": "minor",
            "brightness": 0.25,
            "space": 0.25,
            "density": 5,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "soft_four",
            "texture": "shimmer_slow",
            "accent": "bells",
            "motion": 0.25,
            "attack": "soft",
            "stereo": 0.25,
            "depth": True,
            "echo": 0.5,
            "human": 0.5,
            "grain": "gritty",
        },
        {
            # "justification": "The 'dead', a low volume synth with a simple sine wave and a low pitch.",
            "tempo": 0.25,
            "root": "c",
            "mode": "minor",
            "brightness": 0.25,
            "space": 0.25,
            "density": 2,
            "bass": "drone",
            "pad": "warm_slow",
            "melody": "contemplative",
            "rhythm": "none",
            "texture": "shimmer_slow",
            "accent": "bells",
            "motion": 0.25,
            "attack": "soft",
            "stereo": 0.25,
            "depth": False,
            "echo": 0.25,
            "human": 0.25,
            "grain": "clean",
        },
    ]
    demo_names = ["bubblegum.wav", "sad.wav", "dead.wav"]
    demo_vibes = ["Bubblegum", "sad", "dead"]

    for vibe, fname, config_dict in zip(demo_vibes, demo_names, demo_configs):
        print(f"\n{'=' * 60}")
        print(f"Vibe: {vibe}")
        print("=" * 60)
        config = MusicConfig.from_dict(config_dict)
        print(json.dumps(config_dict, indent=2))
        config_to_audio(config, fname, duration=20.0)
        print(f"   Saved: {fname}")

    print("\nAll demo audio exported.")


# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 1: Direct config
#     print("\n1. Generating from direct config (Indian Wedding)...")
#     config = MusicConfig(
#         tempo=0.36,
#         root="d",
#         mode="dorian",
#         brightness=0.5,
#         space=0.75,
#         density=5,
#         bass="drone",
#         pad="warm_slow",
#         melody="ornamental",
#         rhythm="minimal",
#         texture="shimmer_slow",
#         accent="pluck",
#         motion=0.5,
#         attack="soft",
#         stereo=0.65,
#         depth=True,
#         echo=0.55,
#         human=0.18,
#         grain="warm",
#     )
#     config_to_audio(config, "indian_wedding.wav", duration=20.0)
#     print("   Saved: indian_wedding.wav")

#     # Example 2: From dict (simulating JSON from LLM)
#     print("\n2. Generating from dict (Dark Electronic)...")
#     dark_config = {
#         "tempo": 0.42,
#         "root": "a",
#         "mode": "minor",
#         "brightness": 0.4,
#         "space": 0.6,
#         "density": 6,
#         "layers": {
#             "bass": "pulsing",
#             "pad": "dark_sustained",
#             "melody": "arp_melody",
#             "rhythm": "electronic",
#             "texture": "shimmer",
#             "accent": "bells",
#         },
#         "motion": 0.65,
#         "attack": "sharp",
#         "stereo": 0.7,
#         "depth": True,
#         "echo": 0.5,
#         "human": 0.0,
#         "grain": "gritty",
#     }
#     dict_to_audio(dark_config, "dark_electronic.wav", duration=20.0)
#     print("   Saved: dark_electronic.wav")

#     # Example 3: From vibe string
#     print("\n3. Generating from vibe string (Underwater Cave)...")
#     generate_from_vibe(
#         "slow, peaceful, underwater cave, bioluminescence", "underwater_cave.wav", duration=20.0
#     )
#     print("   Saved: underwater_cave.wav")

#     print("\n" + "=" * 50)

#     # Example 4: Morph between two configs
#     print("\n4. Generating morph (Morning → Evening)...")

#     morning = MusicConfig(
#         tempo=0.30,
#         root="c",
#         mode="major",
#         brightness=0.6,
#         space=0.8,
#         bass="drone",
#         pad="warm_slow",
#         melody="minimal",
#         motion=0.3,
#         attack="soft",
#         echo=0.7,
#     )

#     evening = MusicConfig(
#         tempo=0.42,
#         root="a",
#         mode="minor",
#         brightness=0.45,
#         space=0.6,
#         bass="pulsing",
#         pad="dark_sustained",
#         melody="arp_melody",
#         motion=0.65,
#         attack="medium",
#         echo=0.5,
#     )

#     # Generate 2-minute morph with overlap-add (no volume dips)
#     audio = generate_tween_with_automation(morning, evening, duration=120.0)
#     # audio = assemble(morning, duration=30.0)  # Just morning, no morphing

#     sf.write("morning_to_evening.wav", audio, SAMPLE_RATE)
#     print("   Saved: morning_to_evening.wav")

#     print("\nDone! Generated 4 audio files.")
</file>

</files>
