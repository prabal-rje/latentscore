from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypeGuard

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import MusicConfig
from .errors import ConfigGenerateError, LLMInferenceError, ModelNotAvailableError

ModelChoice = Literal["fast", "expressive", "local"]
MODEL_CHOICES: tuple[ModelChoice, ...] = ("fast", "expressive", "local")
EXTERNAL_PREFIX = "external:"

_EXPRESSIVE_REPO = "mlx-community/gemma-3-1b-it-qat-8bit"
_EXPRESSIVE_DIR = "gemma-3-1b-it-qat-8bit"
_DEFAULT_MAX_RETRIES = 3
_LLM_MAX_TOKENS = 3_000
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_EMBEDDING_DIR = Path("models") / _EMBEDDING_MODEL_NAME
_LOGGER = logging.getLogger("latentscore.models")


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


_TORCH_PARAM_PATCHED = False


def _patch_torch_parameter_for_hf() -> None:
    """Ignore newer HF-specific kwargs when using older torch versions."""
    global _TORCH_PARAM_PATCHED
    if _TORCH_PARAM_PATCHED:
        return
    try:
        import torch
    except Exception:
        return
    try:
        from inspect import signature

        if "_is_hf_initialized" in signature(torch.nn.Parameter.__new__).parameters:
            _TORCH_PARAM_PATCHED = True
            return
    except (TypeError, ValueError):
        pass

    orig_new = torch.nn.Parameter.__new__

    def _patched_new(cls, data=None, requires_grad: bool = True, **_kwargs):
        return orig_new(cls, data, requires_grad)

    torch.nn.Parameter.__new__ = staticmethod(_patched_new)  # type: ignore[assignment]
    _TORCH_PARAM_PATCHED = True


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
        from .spinner import Spinner

        spinner = Spinner("Downloading expressive model (first run)")
        spinner.start()
        try:
            snapshot_download(_EXPRESSIVE_REPO, local_dir=str(model_dir))
        finally:
            spinner.stop()

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
        return _build_external_adapter(
            target,
            api_key=model.api_key,
            litellm_kwargs=model.litellm_kwargs,
        )

    if isinstance(model, str):
        if model.startswith(EXTERNAL_PREFIX):
            return _build_external_adapter(model.removeprefix(EXTERNAL_PREFIX))
        match model:
            case "expressive" | "local" | "fast":
                return _resolve_builtin_model(model)
            case _:
                raise ConfigGenerateError(f"Unknown model choice: {model!r}")

    if _is_model(model):
        return model

    raise ConfigGenerateError(f"Unknown model choice: {model!r}")
