from __future__ import annotations

import asyncio
import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypeGuard

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

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

SYSTEM_PROMPT = f"""
<role description="Detailed description of the role you must embody to successfully complete the task">
You are a WORLD-CLASS music synthesis expert with an eye for detail and a deep understanding of music theory.

You've developed a system that allows you to generate music configurations that match a given vibe/mood description.

Given a vibe/mood description, you previously generated following examples of synth configurations that match a particular vibe.

Study these examples carefully and use THE MOST RELEVANT EXAMPLES TO THE GIVEN "VIBE" as inspiration for your next task.
</role>

<examples description="Examples of synth configurations that match a particular vibe">
{FEW_SHOT_EXAMPLES}
</examples>

<instructions>
1. Generate ONE config for the vibe described in the <input> section.
2. Output ONLY valid JSON for a single MusicConfig object matching the example structure.
3. Your answer should be a single JSON object with only config fields.
</instructions>
"""


@dataclass(frozen=True)
class ExampleConfig:
    vibe: str
    config: MusicConfig


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
            return self._embed_and_select(vibe)
        except Exception as exc:  # pragma: no cover - runtime fallback
            _LOGGER.warning("Fast model fallback: %s", exc)
            return _heuristic_config(vibe)

    @functools.lru_cache(maxsize=1)
    def _load_encoder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
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
            f"{SYSTEM_PROMPT}\n\n"
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
                last_error = exc
            except Exception as exc:  # pragma: no cover - model provider errors
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
