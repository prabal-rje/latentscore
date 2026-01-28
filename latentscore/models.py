from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import platform
from dataclasses import dataclass
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import build_config_generation_prompt

from .config import MusicConfig, MusicConfigPrompt, MusicConfigPromptPayload
from .errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError

ModelChoice = Literal["fast", "expressive", "local"]
MODEL_CHOICES: tuple[ModelChoice, ...] = ("fast", "expressive", "local")
EXTERNAL_PREFIX = "external:"

_EXPRESSIVE_REPO = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_REPO",
    "guprab/latentscore-gemma3-270m-v5-merged",
)
_EXPRESSIVE_DIR = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_DIR",
    "latentscore-gemma3-270m-v5-merged",
)
_EXPRESSIVE_MLX_REPO = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_MLX_REPO",
    "guprab/latentscore-gemma3-270m-v5-mlx-4bit",
)
_EXPRESSIVE_MLX_DIR = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_MLX_DIR",
    "latentscore-gemma3-270m-v5-mlx-4bit",
)
_EXPRESSIVE_GGUF_REPO = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_GGUF_REPO",
    "guprab/latentscore-gemma3-270m-v5-gguf",
)
_EXPRESSIVE_GGUF_FILE = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_GGUF_FILE",
    "latentscore-gemma3-270m-v5-q4_k_m.gguf",
)
_GGUF_CHAT_FORMAT = os.environ.get("LATENTSCORE_GGUF_CHAT_FORMAT", "gemma")
_GGUF_PATH = os.environ.get("LATENTSCORE_GGUF_PATH", "").strip()
_DEFAULT_MAX_RETRIES = 3
_LLM_MAX_TOKENS = 3_000
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_EMBEDDING_DIR = Path("models") / _EMBEDDING_MODEL_NAME
_EMBED_MAP_REPO = os.environ.get(
    "LATENTSCORE_EMBED_MAP_REPO",
    "guprab/latentscore-data",
)
_EMBED_MAP_FILE = os.environ.get(
    "LATENTSCORE_EMBED_MAP_FILE",
    "2026-01-26_scored/_progress_embeddings.jsonl",
)
_LOGGER = logging.getLogger("latentscore.models")
_chat_role_warning_emitted = False
_cpu_backend_warning_emitted = False
_mlx_integration_warning_emitted = False

_TRACK_EXAMPLE_PATTERN = re.compile(
    r"\*\*Example\s+(?P<num>\d+)\*\*\s*Input:\s*\"(?P<input>.*?)\"\s*Output:\s*\n\n```json\n(?P<json>.*?)\n```",
    re.DOTALL,
)


def _fast_fallback_enabled() -> bool:
    value = os.environ.get("LATENTSCORE_FAST_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _fast_fallback_preload() -> bool:
    value = os.environ.get("LATENTSCORE_FAST_FALLBACK_PRELOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


class _FallbackModel:
    def __init__(
        self,
        *,
        primary: ModelForGeneratingMusicConfig,
        fallback: ModelForGeneratingMusicConfig,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        if _fast_fallback_preload():
            warmup = getattr(fallback, "warmup", None)
            if callable(warmup):
                try:
                    warmup()
                except Exception as exc:
                    _LOGGER.warning(
                        "Fast fallback warmup failed (%s). Continuing without preload.",
                        type(exc).__name__,
                        exc_info=True,
                    )

    async def generate(self, vibe: str) -> MusicConfig:
        try:
            return await self._primary.generate(vibe)
        except Exception as exc:
            _LOGGER.warning(
                "Primary model failed (%s); falling back to fast model.",
                type(exc).__name__,
                exc_info=True,
            )
            return await self._fallback.generate(vibe)

    async def aclose(self) -> None:
        for model in (self._primary, self._fallback):
            aclose = getattr(model, "aclose", None)
            if callable(aclose):
                result = aclose()
                if asyncio.iscoroutine(result):
                    await result
                continue
            close = getattr(model, "close", None)
            if callable(close):
                close()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


_GGUF_CTX = _env_int("LATENTSCORE_GGUF_CTX", 4096)
_GGUF_GPU_LAYERS = _env_int("LATENTSCORE_GGUF_GPU_LAYERS", 0)
_GGUF_THREADS = _env_int("LATENTSCORE_GGUF_THREADS", 0)
_GGUF_TEMPERATURE = _env_float("LATENTSCORE_GGUF_TEMPERATURE", 0.2)
_GGUF_TOP_P = _env_float("LATENTSCORE_GGUF_TOP_P", 0.9)
_MLX_REPETITION_PENALTY = _env_float("LATENTSCORE_MLX_REPETITION_PENALTY", 1.35)


def _disable_transformers_progress() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging  # type: ignore[import]
    except Exception:
        return
    try:
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        return


_torch_param_patched = False


def _patch_torch_parameter_for_hf() -> None:
    """Ignore newer HF-specific kwargs when using older torch versions."""
    global _torch_param_patched
    if _torch_param_patched:
        return
    try:
        import torch
    except Exception:
        return
    try:
        from inspect import signature

        param_new = cast(
            Any,
            torch.nn.Parameter.__new__,  # type: ignore[reportUnknownMemberType]
        )
        if "_is_hf_initialized" in signature(param_new).parameters:
            _torch_param_patched = True
            return
    except (TypeError, ValueError):
        pass

    orig_new = cast(
        Any,
        torch.nn.Parameter.__new__,  # type: ignore[reportUnknownMemberType]
    )

    def _patched_new(cls: Any, data: Any = None, requires_grad: bool = True, **_kwargs: Any) -> Any:
        return orig_new(cls, data, requires_grad)

    torch.nn.Parameter.__new__ = staticmethod(_patched_new)  # type: ignore[assignment]
    _torch_param_patched = True


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


@dataclass(frozen=True)
class RuntimeInfo:
    os_name: str
    gpu_type: Literal["apple", "nvidia", "cpu"]


def get_runtime_info() -> RuntimeInfo:
    os_name = platform.system().lower()
    if _is_apple_silicon():
        return RuntimeInfo(os_name=os_name, gpu_type="apple")
    try:
        import torch  # type: ignore[import]
    except Exception:
        return RuntimeInfo(os_name=os_name, gpu_type="cpu")
    if torch.cuda.is_available():
        return RuntimeInfo(os_name=os_name, gpu_type="nvidia")
    return RuntimeInfo(os_name=os_name, gpu_type="cpu")


def _should_use_mlx() -> bool:
    return _resolve_backend() == "mlx"


def _should_use_gguf() -> bool:
    return _resolve_backend() == "gguf"


def _resolve_backend() -> Literal["mlx", "transformers", "gguf"]:
    forced = os.environ.get("LATENTSCORE_FORCE_BACKEND", "").strip().lower()
    if forced:
        if forced in {"mlx", "mlxlm"}:
            return "mlx"
        if forced in {"gguf", "llama", "llamacpp", "cpu"}:
            return "gguf"
        if forced in {"transformers", "torch"}:
            return "transformers"

    info = get_runtime_info()
    match (info.os_name, info.gpu_type):
        case ("windows", _):
            return "transformers"
        case ("darwin", "apple"):
            _warn_mlx_integration_once()
            return "transformers"
        case (_, "nvidia"):
            return "transformers"
        case _:
            return "transformers"


class ModelForGeneratingMusicConfig(Protocol):
    async def generate(self, vibe: str) -> MusicConfig: ...


class ExternalModelSpec(BaseModel):
    model: str
    api_key: str | None = None
    litellm_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")


ModelSpec = ModelChoice | ExternalModelSpec | ModelForGeneratingMusicConfig


def _is_model(obj: object) -> TypeGuard[ModelForGeneratingMusicConfig]:
    return hasattr(obj, "generate")


def build_expressive_prompt() -> str:
    return build_config_generation_prompt()


def build_litellm_prompt() -> str:
    return build_config_generation_prompt()


def _warn_chat_roles_once() -> None:
    global _chat_role_warning_emitted
    if _chat_role_warning_emitted:
        return
    _chat_role_warning_emitted = True
    _LOGGER.warning(
        "Using chat roles for local prompts to match system/user separation."
    )


def _warn_cpu_backend_once(message: str) -> None:
    global _cpu_backend_warning_emitted
    if _cpu_backend_warning_emitted:
        return
    _cpu_backend_warning_emitted = True
    _LOGGER.warning(message)


def _warn_mlx_integration_once() -> None:
    global _mlx_integration_warning_emitted
    if _mlx_integration_warning_emitted:
        return
    _mlx_integration_warning_emitted = True
    _LOGGER.warning(
        "Apple Silicon MLX integration is temporarily disabled; "
        "using CPU transformers backend while integration is underway."
    )


def _make_mlx_repetition_penalty_sampler(penalty: float) -> Callable[[Any], Any]:
    import mlx.core as mx  # type: ignore[import]

    seen_tokens: list[int] = []

    def sampler(logprobs: mx.array) -> mx.array:
        nonlocal seen_tokens

        logprobs_flat = cast(list[float], logprobs.squeeze().tolist())

        for tok in set(seen_tokens):
            if logprobs_flat[tok] > 0:
                logprobs_flat[tok] /= penalty
            else:
                logprobs_flat[tok] *= penalty

        penalized = mx.array(logprobs_flat)
        result = mx.argmax(penalized, axis=-1, keepdims=True)
        seen_tokens.append(int(result.item()))
        return result

    return sampler

def _build_chat_prompt(*, system_prompt: str, vibe: str, tokenizer: Any) -> str:
    _warn_chat_roles_once()
    messages = _build_chat_messages(system_prompt=system_prompt, vibe=vibe)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_chat_messages(*, system_prompt: str, vibe: str) -> list[dict[str, str]]:
    _warn_chat_roles_once()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<vibe>{vibe}</vibe>"},
    ]


class _GgufInstructorModel:
    def __init__(self, *, llama: Any, create: Any) -> None:
        self._llama = llama
        self._create = create

    def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        response_model: type[BaseModel],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> BaseModel:
        return self._create(
            messages=list(messages),
            response_model=response_model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_retries=0,
        )


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


@functools.lru_cache(maxsize=1)
def _fast_track_examples() -> tuple[ExampleConfig, ...]:
    try:
        from .prompt_examples import FEW_SHOT_EXAMPLES
    except Exception as exc:  # pragma: no cover - optional import
        _LOGGER.warning("Fast track examples unavailable: %s", exc, exc_info=True)
        return FAST_EXAMPLES

    examples: list[ExampleConfig] = []
    for match in _TRACK_EXAMPLE_PATTERN.finditer(FEW_SHOT_EXAMPLES):
        input_text = match.group("input").strip()
        payload = match.group("json").strip()
        if not input_text or not payload:
            continue
        try:
            wrapper = MusicConfigPromptPayload.model_validate_json(payload)
            config = wrapper.config.to_config()
        except ValidationError:
            try:
                config = MusicConfig.model_validate_json(payload)
            except ValidationError:
                continue
        examples.append(ExampleConfig(vibe=input_text, config=config))

    return tuple(examples) if examples else FAST_EXAMPLES


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

    def warmup(self) -> None:
        _ = self._example_matrix()

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

        _disable_transformers_progress()
        _patch_torch_parameter_for_hf()
        model_dir = self._model_dir
        if model_dir is None:
            model_dir = _LOCAL_EMBEDDING_DIR if _LOCAL_EMBEDDING_DIR.exists() else None

        return SentenceTransformer(str(model_dir) if model_dir else _EMBEDDING_MODEL_NAME)

    @functools.lru_cache(maxsize=1)
    def _examples(self) -> tuple[ExampleConfig, ...]:
        examples, _ = self._embed_map_examples()
        return examples

    @functools.lru_cache(maxsize=1)
    def _example_matrix(self) -> NDArray[np.float32]:
        examples, embed_matrix = self._embed_map_examples()
        if embed_matrix is not None:
            return embed_matrix
        encoder = self._load_encoder()
        texts = [example.vibe for example in examples]
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_and_select(self, vibe: str) -> MusicConfig:
        encoder = self._load_encoder()
        example_matrix = self._example_matrix()
        query = encoder.encode([vibe], normalize_embeddings=True)
        query_vec = np.asarray(query, dtype=np.float32)[0]
        scores = example_matrix @ query_vec
        best_idx = int(np.argmax(scores))
        return self._examples()[best_idx].config

    @functools.lru_cache(maxsize=1)
    def _embed_map_examples(self) -> tuple[tuple[ExampleConfig, ...], NDArray[np.float32] | None]:
        map_path = self._resolve_embed_map_path()
        if map_path is None or not map_path.exists():
            return _fast_track_examples(), None

        examples: list[ExampleConfig] = []
        embeddings: list[list[float]] = []
        try:
            with map_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    vibe = row.get("vibe_original") or row.get("vibe") or row.get("vibe_noisy")
                    config = row.get("config")
                    if not vibe or not isinstance(config, dict):
                        continue
                    try:
                        prompt_config = MusicConfigPrompt.model_validate(config)
                        music_config = prompt_config.to_config()
                    except ValidationError:
                        try:
                            music_config = MusicConfig.model_validate(config)
                        except ValidationError:
                            continue
                    examples.append(ExampleConfig(vibe=str(vibe), config=music_config))
                    embed = row.get("embedding")
                    if isinstance(embed, list):
                        embeddings.append([float(x) for x in embed])
        except OSError as exc:  # pragma: no cover - filesystem edge case
            _LOGGER.warning("Failed reading embedding map: %s", exc, exc_info=True)
            return _fast_track_examples(), None

        if not examples:
            return _fast_track_examples(), None

        matrix: NDArray[np.float32] | None = None
        if len(embeddings) == len(examples):
            mat = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = mat / norms
        return tuple(examples), matrix

    def _resolve_embed_map_path(self) -> Path | None:
        explicit = os.environ.get("LATENTSCORE_EMBED_MAP", "").strip()
        if explicit:
            return Path(explicit).expanduser()

        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
            return None

        base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", "")).expanduser()
        if not base_dir:
            base_dir = Path.home() / ".cache" / "latentscore" / "models"
        cache_dir = base_dir / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            return Path(
                hf_hub_download(
                    repo_id=_EMBED_MAP_REPO,
                    repo_type="dataset",
                    filename=_EMBED_MAP_FILE,
                    cache_dir=str(cache_dir),
                )
            )
        except Exception as exc:  # pragma: no cover - network/IO errors
            _LOGGER.warning(
                "Failed to download embedding map (%s/%s): %s",
                _EMBED_MAP_REPO,
                _EMBED_MAP_FILE,
                exc,
                exc_info=True,
            )
            return None


class ExpressiveMlxModel:
    """Local transformers-backed model using the LLM prompt from the synth prototype."""

    @staticmethod
    def check_dependencies() -> None:
        backend = _resolve_backend()
        if backend == "gguf":
            try:
                import instructor  # type: ignore[import] # noqa: F401
                import llama_cpp  # type: ignore[import] # noqa: F401
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "llama-cpp-python + instructor are required for GGUF inference"
                ) from exc
            return

        try:
            import outlines  # type: ignore[import] # noqa: F401
        except ImportError as exc:
            raise ModelNotAvailableError("outlines is not installed") from exc

        if backend == "mlx":
            try:
                import mlx  # type: ignore[import] # noqa: F401
                import mlx_lm  # type: ignore[import] # noqa: F401
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "mlx and mlx-lm are required on Apple Silicon; install them first"
                ) from exc
            return

        try:
            import transformers  # type: ignore[import]
        except ImportError as exc:
            raise ModelNotAvailableError("transformers is not installed") from exc

        version = transformers.__version__.split(".")[0]
        if version.isdigit() and int(version) >= 5:
            raise ModelNotAvailableError(
                "transformers>=5 is not supported; install transformers<5"
            )
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency mismatch
            raise ModelNotAvailableError(
                "transformers is missing AutoTokenizer; install transformers<5"
            ) from exc
        _ = AutoTokenizer
        # bitsandbytes is optional; we fall back to fp16/fp32 if unavailable.

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

    def warmup(self) -> None:
        _ = self._load_model()

    async def generate(self, vibe: str) -> MusicConfig:
        return await asyncio.to_thread(self._generate_sync, vibe)

    def _resolve_model_dir(self) -> Path:
        if self._model_dir is not None:
            return self._model_dir
        base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", ""))
        if not base_dir:
            base_dir = Path.home() / ".cache" / "latentscore" / "models"
        if _resolve_backend() == "mlx":
            return base_dir / _EXPRESSIVE_MLX_DIR
        return base_dir / _EXPRESSIVE_DIR

    def _resolve_mlx_model_dir(self) -> Path:
        if self._model_dir is not None:
            return self._model_dir
        base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", ""))
        if not base_dir:
            base_dir = Path.home() / ".cache" / "latentscore" / "models"
        return base_dir / _EXPRESSIVE_MLX_DIR

    def _download_expressive(self, model_dir: Path) -> None:
        _disable_transformers_progress()
        os.environ.setdefault("TQDM_DISABLE", "1")
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
        from .spinner import Spinner

        spinner = Spinner("Downloading expressive model (first run)", show_elapsed=True)
        spinner.start()
        try:
            snapshot_download(_EXPRESSIVE_REPO, local_dir=str(model_dir))
        finally:
            spinner.stop()

    def _download_expressive_mlx(self, model_dir: Path) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError("huggingface_hub is not installed") from exc

        _LOGGER.info(
            "Downloading MLX model weights (~200MB). This happens once and is cached at %s.",
            model_dir.parent,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        from .spinner import Spinner

        spinner = Spinner("Downloading expressive MLX model (first run)", show_elapsed=True)
        spinner.start()
        try:
            snapshot_download(_EXPRESSIVE_MLX_REPO, local_dir=str(model_dir))
        finally:
            spinner.stop()

    def _resolve_gguf_path(self) -> Path:
        if _GGUF_PATH:
            return Path(_GGUF_PATH).expanduser()
        if not self._allow_download:
            raise ModelNotAvailableError(
                "GGUF model missing and downloads are disabled. "
                "Set LATENTSCORE_GGUF_PATH or run `latentscore download expressive`."
            )
        return self._download_expressive_gguf()

    def _download_expressive_gguf(self) -> Path:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
            raise ModelNotAvailableError("huggingface_hub is not installed") from exc

        base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", ""))
        if not base_dir:
            base_dir = Path.home() / ".cache" / "latentscore" / "models"
        cache_dir = base_dir / "gguf"
        cache_dir.mkdir(parents=True, exist_ok=True)

        _LOGGER.info(
            "Downloading GGUF model (~235MB). This happens once and is cached at %s.",
            cache_dir,
        )
        from .spinner import Spinner

        spinner = Spinner("Downloading expressive GGUF model (first run)", show_elapsed=True)
        spinner.start()
        try:
            return Path(
                hf_hub_download(
                    repo_id=_EXPRESSIVE_GGUF_REPO,
                    filename=_EXPRESSIVE_GGUF_FILE,
                    cache_dir=str(cache_dir),
                )
            )
        finally:
            spinner.stop()

    def _load_gguf_model(self) -> _GgufInstructorModel:
        import instructor  # type: ignore[import]
        import llama_cpp  # type: ignore[import]

        gguf_path = self._resolve_gguf_path()
        llama_kwargs: dict[str, Any] = {
            "model_path": str(gguf_path),
            "n_ctx": _GGUF_CTX,
            "n_gpu_layers": _GGUF_GPU_LAYERS,
            "chat_format": _GGUF_CHAT_FORMAT,
            "logits_all": True,
            "verbose": False,
        }
        if _GGUF_THREADS > 0:
            llama_kwargs["n_threads"] = _GGUF_THREADS
        if _GGUF_GPU_LAYERS <= 0:
            _warn_cpu_backend_once(
                "Using CPU-only GGUF inference. Set LATENTSCORE_GGUF_GPU_LAYERS>0 to enable GPU offload."
            )
        llama = llama_cpp.Llama(**llama_kwargs)

        create = instructor.patch(
            create=llama.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        return _GgufInstructorModel(llama=llama, create=create)

    @functools.lru_cache(maxsize=1)
    def _load_model(self) -> tuple[Any, Any, str]:
        try:
            self.check_dependencies()
        except ModelNotAvailableError as exc:
            _LOGGER.warning("Expressive model dependencies unavailable: %s", exc, exc_info=True)
            raise

        backend = _resolve_backend()
        if backend == "gguf":
            return self._load_gguf_model(), None, "gguf"

        import outlines  # type: ignore[import]

        if backend == "mlx":
            import mlx_lm  # type: ignore[import]

            model_dir = self._resolve_mlx_model_dir()
            if not model_dir.exists():
                if not self._allow_download:
                    raise ModelNotAvailableError(
                        f"Expressive MLX model missing at {model_dir}. "
                        "Run `latentscore download expressive` or set LATENTSCORE_MODEL_DIR."
                    )
                self._download_expressive_mlx(model_dir)

            model, tokenizer = mlx_lm.load(
                str(model_dir),
                tokenizer_config={"fix_mistral_regex": True},
            )
            return outlines.from_mlxlm(model, tokenizer), tokenizer, "mlx"

        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime = get_runtime_info()
        use_cuda = torch.cuda.is_available() and runtime.os_name != "windows"
        if not use_cuda:
            _warn_cpu_backend_once(
                "Using CPU-only transformers inference. Install CUDA-enabled PyTorch and a supported GPU for faster generation."
            )
        use_bnb = False
        bnb_config = None
        if use_cuda:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[import]
                import bitsandbytes  # type: ignore[import]  # noqa: F401

                use_bnb = True
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                use_bnb = False

        model_dir = self._resolve_model_dir()
        if not model_dir.exists():
            if not self._allow_download:
                raise ModelNotAvailableError(
                    f"Expressive model missing at {model_dir}. "
                    "Run `latentscore download expressive` or set LATENTSCORE_MODEL_DIR."
                )
            self._download_expressive(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        torch_dtype = torch.float16 if use_cuda else torch.float32
        if use_bnb and bnb_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch_dtype,
            )
            model.to("cuda" if use_cuda else "cpu")
        model.eval()
        outlines_any: Any = outlines
        return outlines_any.from_transformers(model, tokenizer), tokenizer, "transformers"

    def _build_prompt(self, tokenizer: Any, vibe: str) -> str:
        return _build_chat_prompt(
            system_prompt=build_expressive_prompt(),
            vibe=vibe,
            tokenizer=tokenizer,
        )

    def _generate_sync_gguf(self, model: _GgufInstructorModel, vibe: str) -> MusicConfig:
        last_error: Exception | None = None
        messages = _build_chat_messages(
            system_prompt=build_expressive_prompt(),
            vibe=vibe,
        )
        for attempt in range(self._max_retries):
            temperature = min(1.0, _GGUF_TEMPERATURE + 0.1 * attempt)
            top_p = min(1.0, _GGUF_TOP_P + 0.05 * attempt)
            try:
                payload = model.generate(
                    messages=messages,
                    response_model=MusicConfigPromptPayload,
                    max_new_tokens=_LLM_MAX_TOKENS,
                    temperature=temperature,
                    top_p=top_p,
                )
                if isinstance(payload, MusicConfigPromptPayload):
                    return payload.config.to_config()
                validated = MusicConfigPromptPayload.model_validate(payload)
                return validated.config.to_config()
            except ValidationError as exc:
                _LOGGER.warning("GGUF model returned invalid JSON: %s", exc, exc_info=True)
                last_error = exc
            except Exception as exc:  # pragma: no cover - model provider errors
                _LOGGER.warning("GGUF model inference failed: %s", exc, exc_info=True)
                last_error = exc

        message = f"Failed to generate config for vibe {vibe!r}."
        raise LLMInferenceError(message) from last_error

    def _generate_sync(self, vibe: str) -> MusicConfig:
        model, tokenizer, backend = self._load_model()
        if backend == "gguf":
            return self._generate_sync_gguf(cast(_GgufInstructorModel, model), vibe)
        last_error: Exception | None = None
        for _ in range(self._max_retries):
            prompt = self._build_prompt(tokenizer, vibe)
            try:
                if getattr(model, "tensor_library_name", "") == "mlx":
                    sampler = None
                    if _MLX_REPETITION_PENALTY and _MLX_REPETITION_PENALTY != 1.0:
                        try:
                            sampler = _make_mlx_repetition_penalty_sampler(
                                _MLX_REPETITION_PENALTY
                            )
                        except Exception as exc:
                            _LOGGER.warning(
                                "MLX repetition sampler unavailable: %s",
                                exc,
                                exc_info=True,
                            )

                    raw = model(
                        prompt,
                        output_type=MusicConfigPromptPayload,
                        max_tokens=_LLM_MAX_TOKENS,
                        sampler=sampler,
                    )
                else:
                    raw = model(
                        prompt,
                        output_type=MusicConfigPromptPayload,
                        max_new_tokens=_LLM_MAX_TOKENS,
                    )
                payload = MusicConfigPromptPayload.model_validate_json(raw)
                return payload.config.to_config()
            except ValidationError as exc:
                _LOGGER.warning("Expressive model returned invalid JSON: %s", exc, exc_info=True)
                last_error = exc
            except Exception as exc:  # pragma: no cover - model provider errors
                _LOGGER.warning("Expressive model inference failed: %s", exc, exc_info=True)
                last_error = exc

        message = f"Failed to generate config for vibe {vibe!r}."
        raise LLMInferenceError(message) from last_error


@functools.lru_cache(maxsize=len(MODEL_CHOICES))
def _resolve_builtin_model(choice: ModelChoice) -> ModelForGeneratingMusicConfig:
    match choice:
        case "expressive":
            return ExpressiveMlxModel()
        case "local":
            return ExpressiveMlxModel()
        case "fast":
            return FastEmbeddingModel()
        case _:
            raise ConfigGenerateError(f"Unknown model choice: {choice!r}")


def _build_external_adapter(
    model: str,
    *,
    api_key: str | None = None,
    litellm_kwargs: Mapping[str, Any] | None = None,
) -> ModelForGeneratingMusicConfig:
    from .providers.litellm import LiteLLMAdapter

    return LiteLLMAdapter(
        model=model,
        api_key=api_key,
        litellm_kwargs=litellm_kwargs,
    )


def resolve_model(
    model: ModelSpec,
) -> ModelForGeneratingMusicConfig:
    if isinstance(model, ExternalModelSpec):
        target = model.model
        if target.startswith(EXTERNAL_PREFIX):
            target = target.removeprefix(EXTERNAL_PREFIX)
        primary = _build_external_adapter(
            target,
            api_key=model.api_key,
            litellm_kwargs=model.litellm_kwargs,
        )
        if _fast_fallback_enabled():
            return _FallbackModel(primary=primary, fallback=_resolve_builtin_model("fast"))
        return primary

    if isinstance(model, str):
        if model.startswith(EXTERNAL_PREFIX):
            primary = _build_external_adapter(model.removeprefix(EXTERNAL_PREFIX))
            if _fast_fallback_enabled():
                return _FallbackModel(primary=primary, fallback=_resolve_builtin_model("fast"))
            return primary
        match model:
            case "expressive" | "local" | "fast":
                primary = _resolve_builtin_model(model)
                if model != "fast" and _fast_fallback_enabled():
                    return _FallbackModel(primary=primary, fallback=_resolve_builtin_model("fast"))
                return primary
            case _:
                raise ConfigGenerateError(f"Unknown model choice: {model!r}")

    if _is_model(model):
        return model

    raise ConfigGenerateError(f"Unknown model choice: {model!r}")
