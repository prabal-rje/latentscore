"""Baseline config generators for evaluation comparison."""

from __future__ import annotations

import functools
import json
import math
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (  # noqa: F401 (TypeVar used in generic)
    Any,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    get_args,
)

from common.music_schema import (
    MAX_TITLE_CHARS,
    MAX_TITLE_WORDS,
    AccentStyle,
    AttackStyle,
    BassStyle,
    BrightnessLabel,
    CadenceLabel,
    ChordChangeLabel,
    ChordExtensions,
    ChromaticLabel,
    DensityLevel,
    EchoLabel,
    GrainStyle,
    HarmonyStyle,
    HumanFeelLabel,
    MelodyDensityLabel,
    MelodyEngine,
    MelodyStyle,
    ModeName,
    MotifRepeatLabel,
    MotionLabel,
    MusicConfigPrompt,
    MusicConfigPromptPayload,
    PadStyle,
    Palette,
    PaletteColor,
    PhraseLengthBars,
    RhythmStyle,
    RootNote,
    SpaceLabel,
    StepBiasLabel,
    StereoLabel,
    SwingLabel,
    SyncopationLabel,
    TempoLabel,
    TensionCurve,
    TextureStyle,
    WeightLabel,
)

# Extract valid values from Literal types with explicit type annotations.
# This allows random.choice() to return the correct type instead of Any.
TEMPO_VALUES: Sequence[TempoLabel] = get_args(TempoLabel)
BRIGHTNESS_VALUES: Sequence[BrightnessLabel] = get_args(BrightnessLabel)
MODE_VALUES: Sequence[ModeName] = get_args(ModeName)
ROOT_VALUES: Sequence[RootNote] = get_args(RootNote)
SPACE_VALUES: Sequence[SpaceLabel] = get_args(SpaceLabel)
MOTION_VALUES: Sequence[MotionLabel] = get_args(MotionLabel)
STEREO_VALUES: Sequence[StereoLabel] = get_args(StereoLabel)
ECHO_VALUES: Sequence[EchoLabel] = get_args(EchoLabel)
HUMAN_FEEL_VALUES: Sequence[HumanFeelLabel] = get_args(HumanFeelLabel)
DENSITY_VALUES: Sequence[DensityLevel] = get_args(DensityLevel)
BASS_STYLES: Sequence[BassStyle] = get_args(BassStyle)
PAD_STYLES: Sequence[PadStyle] = get_args(PadStyle)
MELODY_STYLES: Sequence[MelodyStyle] = get_args(MelodyStyle)
RHYTHM_STYLES: Sequence[RhythmStyle] = get_args(RhythmStyle)
TEXTURE_STYLES: Sequence[TextureStyle] = get_args(TextureStyle)
ACCENT_STYLES: Sequence[AccentStyle] = get_args(AccentStyle)
ATTACK_STYLES: Sequence[AttackStyle] = get_args(AttackStyle)
GRAIN_STYLES: Sequence[GrainStyle] = get_args(GrainStyle)
MELODY_ENGINES: Sequence[MelodyEngine] = get_args(MelodyEngine)
TENSION_CURVES: Sequence[TensionCurve] = get_args(TensionCurve)
HARMONY_STYLES: Sequence[HarmonyStyle] = get_args(HarmonyStyle)
CHORD_EXTENSIONS_VALUES: Sequence[ChordExtensions] = get_args(ChordExtensions)
PHRASE_LENGTH_BARS: Sequence[PhraseLengthBars] = get_args(PhraseLengthBars)
MELODY_DENSITY_VALUES: Sequence[MelodyDensityLabel] = get_args(MelodyDensityLabel)
SYNCOPATION_VALUES: Sequence[SyncopationLabel] = get_args(SyncopationLabel)
SWING_VALUES: Sequence[SwingLabel] = get_args(SwingLabel)
MOTIF_REPEAT_VALUES: Sequence[MotifRepeatLabel] = get_args(MotifRepeatLabel)
STEP_BIAS_VALUES: Sequence[StepBiasLabel] = get_args(StepBiasLabel)
CHROMATIC_VALUES: Sequence[ChromaticLabel] = get_args(ChromaticLabel)
CADENCE_VALUES: Sequence[CadenceLabel] = get_args(CadenceLabel)

_TITLE_FALLBACK = "Ambient Mood Study"
_TITLE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_EMBED_MAP_REPO = "guprab/latentscore-data"
_DEFAULT_EMBED_MAP_FILE = "2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl"
_DEFAULT_CLAP_EMBED_MAP_FILE = "2026-01-26_scored/vibe_and_clap_audio_embeddings_to_config_map.jsonl"


def _make_title(vibe: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", vibe)
    tokens = _TITLE_TOKEN_PATTERN.findall(cleaned)
    if not tokens:
        return _TITLE_FALLBACK
    words = tokens[:MAX_TITLE_WORDS]
    title = " ".join(word.capitalize() for word in words)
    if len(title) <= MAX_TITLE_CHARS:
        return title
    while words and len(" ".join(word.capitalize() for word in words)) > MAX_TITLE_CHARS:
        words = words[:-1]
    if not words:
        return _TITLE_FALLBACK
    return " ".join(word.capitalize() for word in words)


CHORD_CHANGE_VALUES: Sequence[ChordChangeLabel] = get_args(ChordChangeLabel)
WEIGHT_VALUES: Sequence[WeightLabel] = get_args(WeightLabel)

# Hex colors for random palettes
HEX_COLORS = [
    "#1a237e",
    "#283593",
    "#303f9f",
    "#3949ab",
    "#3f51b5",  # blues
    "#e65100",
    "#ef6c00",
    "#f57c00",
    "#fb8c00",
    "#ff9800",  # oranges
    "#1b5e20",
    "#2e7d32",
    "#388e3c",
    "#43a047",
    "#4caf50",  # greens
    "#4a148c",
    "#6a1b9a",
    "#7b1fa2",
    "#8e24aa",
    "#9c27b0",  # purples
    "#ff6f00",
    "#ff8f00",
    "#ffa000",
    "#ffb300",
    "#ffc107",  # golds
    "#263238",
    "#37474f",
    "#455a64",
    "#546e7a",
    "#607d8b",  # grays
    "#b71c1c",
    "#c62828",
    "#d32f2f",
    "#e53935",
    "#f44336",  # reds
    "#006064",
    "#00838f",
    "#0097a7",
    "#00acc1",
    "#00bcd4",  # cyans
]


class ConfigBaseline(ABC):
    """Abstract base class for config baselines."""

    @abstractmethod
    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config from a vibe prompt."""
        pass


class RandomConfigBaseline(ConfigBaseline):
    """Baseline that generates random valid configs."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _random_palette(self) -> Palette:
        """Generate a random palette with exactly 5 colors."""
        colors = self.rng.sample(HEX_COLORS, k=5)
        weights = self.rng.sample(list(WEIGHT_VALUES), k=5)
        return Palette(colors=[PaletteColor(hex=c, weight=w) for c, w in zip(colors, weights)])

    def random_palette(self) -> Palette:
        """Public wrapper for palette generation (used by other baselines)."""
        return self._random_palette()

    def _random_config(self) -> MusicConfigPrompt:
        """Generate a random MusicConfigPrompt."""
        _MIN_OCTAVE = 1
        _MAX_OCTAVE_LOW = 4
        _MAX_OCTAVE_HIGH = 8
        min_oct = self.rng.randint(_MIN_OCTAVE, _MAX_OCTAVE_LOW)
        max_oct = self.rng.randint(min_oct + 1, _MAX_OCTAVE_HIGH)
        return MusicConfigPrompt(
            tempo=self.rng.choice(TEMPO_VALUES),
            root=self.rng.choice(ROOT_VALUES),
            mode=self.rng.choice(MODE_VALUES),
            brightness=self.rng.choice(BRIGHTNESS_VALUES),
            space=self.rng.choice(SPACE_VALUES),
            density=self.rng.choice(DENSITY_VALUES),
            bass=self.rng.choice(BASS_STYLES),
            pad=self.rng.choice(PAD_STYLES),
            melody=self.rng.choice(MELODY_STYLES),
            rhythm=self.rng.choice(RHYTHM_STYLES),
            texture=self.rng.choice(TEXTURE_STYLES),
            accent=self.rng.choice(ACCENT_STYLES),
            motion=self.rng.choice(MOTION_VALUES),
            attack=self.rng.choice(ATTACK_STYLES),
            stereo=self.rng.choice(STEREO_VALUES),
            depth=self.rng.choice([True, False]),
            echo=self.rng.choice(ECHO_VALUES),
            human=self.rng.choice(HUMAN_FEEL_VALUES),
            grain=self.rng.choice(GRAIN_STYLES),
            melody_engine=self.rng.choice(MELODY_ENGINES),
            phrase_len_bars=self.rng.choice(PHRASE_LENGTH_BARS),
            melody_density=self.rng.choice(MELODY_DENSITY_VALUES),
            syncopation=self.rng.choice(SYNCOPATION_VALUES),
            swing=self.rng.choice(SWING_VALUES),
            motif_repeat_prob=self.rng.choice(MOTIF_REPEAT_VALUES),
            step_bias=self.rng.choice(STEP_BIAS_VALUES),
            chromatic_prob=self.rng.choice(CHROMATIC_VALUES),
            cadence_strength=self.rng.choice(CADENCE_VALUES),
            register_min_oct=min_oct,
            register_max_oct=max_oct,
            tension_curve=self.rng.choice(TENSION_CURVES),
            harmony_style=self.rng.choice(HARMONY_STYLES),
            chord_change_bars=self.rng.choice(CHORD_CHANGE_VALUES),
            chord_extensions=self.rng.choice(CHORD_EXTENSIONS_VALUES),
        )

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a random MusicConfigPromptPayload (ignores vibe)."""
        _ = vibe  # Intentionally ignored
        return MusicConfigPromptPayload(
            thinking="Random baseline config for evaluation purposes.",
            title=_make_title(vibe),
            config=self._random_config(),
            palettes=[self._random_palette() for _ in range(3)],
        )


# Keyword mappings for rule-based baseline
TEMPO_KEYWORDS: dict[TempoLabel, list[str]] = {
    "very_slow": ["glacial", "frozen", "adagio", "crawl", "statue", "still"],
    "slow": ["slow", "languid", "leisurely", "drift", "gentle", "calm", "peaceful"],
    "fast": ["fast", "quick", "urgent", "rush", "chase", "rapid", "allegro"],
    "very_fast": ["frantic", "presto", "racing", "sprint", "breakneck", "furious"],
}

BRIGHTNESS_KEYWORDS: dict[BrightnessLabel, list[str]] = {
    "very_dark": ["pitch black", "void", "abyss", "lightless", "moonless"],
    "dark": ["dark", "dim", "shadow", "night", "fog", "gray", "overcast", "candle"],
    "bright": ["bright", "sunny", "light", "morning", "golden", "glow"],
    "very_bright": ["blinding", "dazzling", "radiant", "noon", "fluorescent", "gleaming"],
}

MODE_KEYWORDS: dict[ModeName, list[str]] = {
    "minor": [
        "sad",
        "melancholy",
        "grief",
        "loss",
        "mournful",
        "tragic",
        "funeral",
        "somber",
        "bittersweet",
        "minor",
    ],
    "major": [
        "happy",
        "joy",
        "triumph",
        "victory",
        "celebration",
        "hopeful",
        "uplifting",
        "wedding",
        "bright",
        "major",
        "cheerful",
    ],
    "dorian": ["jazz", "dorian", "funky", "groove", "smoky", "cool"],
    "mixolydian": ["folk", "celtic", "country", "mixolydian", "highlands"],
}

SPACE_KEYWORDS: dict[SpaceLabel, list[str]] = {
    "dry": ["dry", "close", "intimate", "dead"],
    "small": ["room", "indoor", "small"],
    "large": ["hall", "cathedral", "large", "vast"],
    "vast": ["space", "infinite", "endless", "cosmos", "universe"],
}


_T = TypeVar("_T", bound=str)


class RuleBasedBaseline(ConfigBaseline):
    """Baseline using keyword matching heuristics."""

    def __init__(self) -> None:
        self.random_baseline = RandomConfigBaseline()

    def _match_keywords(
        self,
        vibe: str,
        keyword_map: Mapping[_T, Sequence[str]],
        default: _T,
    ) -> _T:
        """Find the best matching value based on keywords."""
        vibe_lower = vibe.lower()
        best_match = default
        best_count = 0

        for value, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in vibe_lower)
            if count > best_count:
                best_count = count
                best_match = value

        return best_match

    def _extract_numbers(self, vibe: str) -> list[int]:
        """Extract numbers from vibe text."""
        return [int(n) for n in re.findall(r"\d+", vibe)]

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config using keyword matching."""
        # Start with random config as base
        payload = self.random_baseline.generate(vibe)

        # Default values for keyword matching (typed as Literals)
        _DEFAULT_TEMPO: TempoLabel = "medium"
        _DEFAULT_BRIGHTNESS: BrightnessLabel = "medium"
        _DEFAULT_MODE: ModeName = "minor"
        _DEFAULT_SPACE: SpaceLabel = "medium"

        # Create a mutable copy of the config with overrides
        tempo = cast(TempoLabel, self._match_keywords(vibe, TEMPO_KEYWORDS, _DEFAULT_TEMPO))
        brightness = cast(
            BrightnessLabel,
            self._match_keywords(vibe, BRIGHTNESS_KEYWORDS, _DEFAULT_BRIGHTNESS),
        )
        mode = cast(ModeName, self._match_keywords(vibe, MODE_KEYWORDS, _DEFAULT_MODE))
        space = cast(SpaceLabel, self._match_keywords(vibe, SPACE_KEYWORDS, _DEFAULT_SPACE))

        # Adjust motion based on tempo
        motion: MotionLabel = "medium"
        if tempo in ["very_slow", "slow"]:
            motion = "slow"
        elif tempo in ["fast", "very_fast"]:
            motion = "fast"

        # Adjust rhythm based on tempo and keywords
        rhythm: RhythmStyle = payload.config.rhythm
        if "no drum" in vibe.lower() or "ambient" in vibe.lower():
            rhythm = "none"
        elif tempo == "very_slow":
            rhythm = "none"
        elif tempo == "slow":
            rhythm = "minimal"

        # Create new config with overrides
        new_config = MusicConfigPrompt(
            tempo=tempo,
            brightness=brightness,
            mode=mode,
            space=space,
            motion=motion,
            rhythm=rhythm,
            # Keep the rest from the random baseline
            root=payload.config.root,
            density=payload.config.density,
            bass=payload.config.bass,
            pad=payload.config.pad,
            melody=payload.config.melody,
            texture=payload.config.texture,
            accent=payload.config.accent,
            attack=payload.config.attack,
            stereo=payload.config.stereo,
            depth=payload.config.depth,
            echo=payload.config.echo,
            human=payload.config.human,
            grain=payload.config.grain,
            melody_engine=payload.config.melody_engine,
            phrase_len_bars=payload.config.phrase_len_bars,
            melody_density=payload.config.melody_density,
            syncopation=payload.config.syncopation,
            swing=payload.config.swing,
            motif_repeat_prob=payload.config.motif_repeat_prob,
            step_bias=payload.config.step_bias,
            chromatic_prob=payload.config.chromatic_prob,
            cadence_strength=payload.config.cadence_strength,
            register_min_oct=payload.config.register_min_oct,
            register_max_oct=payload.config.register_max_oct,
            tension_curve=payload.config.tension_curve,
            harmony_style=payload.config.harmony_style,
            chord_change_bars=payload.config.chord_change_bars,
            chord_extensions=payload.config.chord_extensions,
        )

        return MusicConfigPromptPayload(
            thinking=f"Rule-based config for vibe: {vibe[:50]}...",
            title=_make_title(vibe),
            config=new_config,
            palettes=payload.palettes,
        )


class ModeConfigBaseline(ConfigBaseline):
    """Baseline using most common values (mode) from training data."""

    def __init__(self) -> None:
        """Initialize with default mode values."""
        self.random_baseline = RandomConfigBaseline(seed=42)  # Fixed seed for consistency

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        """Generate a config using mode (most common) values (ignores vibe)."""
        _ = vibe  # Intentionally ignored
        # Start with random config for palettes and other fields
        payload = self.random_baseline.generate("")

        # Create config with "mode" (most common) values
        new_config = MusicConfigPrompt(
            tempo="medium",
            brightness="medium",
            space="medium",
            motion="medium",
            stereo="medium",
            echo="medium",
            human="natural",
            root="c",
            mode="minor",
            density=4,
            bass="drone",
            pad="ambient_drift",
            melody="contemplative",
            rhythm="minimal",
            texture="shimmer",
            accent="none",
            attack="medium",
            grain="warm",
            depth=True,
            melody_engine="procedural",
            phrase_len_bars=4,
            melody_density="medium",
            syncopation="light",
            swing="none",
            motif_repeat_prob="sometimes",
            step_bias="balanced",
            chromatic_prob="light",
            cadence_strength="medium",
            register_min_oct=3,
            register_max_oct=5,
            tension_curve="arc",
            harmony_style="ambient",
            chord_change_bars="slow",
            chord_extensions="triads",
        )

        return MusicConfigPromptPayload(
            thinking="Mode baseline using most common values.",
            title=_make_title(vibe),
            config=new_config,
            palettes=payload.palettes,
        )


class _RetrievalExample:
    def __init__(
        self,
        *,
        vibe: str,
        config: MusicConfigPrompt,
        palettes: list[Palette],
    ) -> None:
        self.vibe = vibe
        self.config = config
        self.palettes = palettes


class _EmbeddingLookupIndex:
    def __init__(self, *, embed_map_path: Path) -> None:
        self._embed_map_path = embed_map_path
        self._encoder: Any | None = None
        self._examples: list[_RetrievalExample] = []
        self._matrix: Any | None = None

    def warmup(self) -> None:
        _ = self._load()

    def lookup(self, vibe: str) -> _RetrievalExample:
        index = self._load()
        return index._lookup_impl(vibe)

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - environment mismatch
            raise RuntimeError("sentence-transformers is required for retrieval baseline.") from exc
        self._encoder = cast(Any, SentenceTransformer(_DEFAULT_EMBED_MODEL))
        return cast(Any, self._encoder)

    def _load(self) -> "_EmbeddingLookupIndex":
        if self._matrix is not None and self._examples:
            return self

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment mismatch
            raise RuntimeError("numpy is required for retrieval baseline.") from exc

        from pydantic import ValidationError

        # Reuse random palettes if a row doesn't have valid ones.
        palette_fallback = RandomConfigBaseline(seed=42)

        vibes: list[str] = []
        examples: list[_RetrievalExample] = []
        embeddings: list[list[float]] = []

        with self._embed_map_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row_obj: object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row_obj, dict):
                    continue
                row = cast(dict[str, Any], row_obj)

                # Build the lookup index from non-test splits only (avoid leakage).
                split = row.get("split")
                if isinstance(split, str) and split.upper() == "TEST":
                    continue

                vibe_value = row.get("vibe_original") or row.get("vibe") or row.get("vibe_noisy")
                config_value = row.get("config")
                if not isinstance(vibe_value, str) or not vibe_value.strip():
                    continue
                if not isinstance(config_value, dict):
                    continue

                try:
                    config = MusicConfigPrompt.model_validate(config_value)
                except ValidationError:
                    continue

                palettes_value = row.get("palettes")
                palettes: list[Palette] = []
                if isinstance(palettes_value, list):
                    palettes_list = cast(list[Any], palettes_value)
                    for item in palettes_list:
                        if not isinstance(item, dict):
                            palettes = []
                            break
                        try:
                            palettes.append(Palette.model_validate(item))
                        except ValidationError:
                            palettes = []
                            break

                if len(palettes) != 3:
                    palettes = [palette_fallback.random_palette() for _ in range(3)]

                vibes.append(vibe_value)
                examples.append(
                    _RetrievalExample(
                        vibe=vibe_value,
                        config=config,
                        palettes=palettes,
                    )
                )

                embed_value = row.get("embedding")
                if isinstance(embed_value, list):
                    embed_list = cast(list[Any], embed_value)
                    try:
                        embeddings.append([float(x) for x in embed_list])
                    except (TypeError, ValueError):
                        embeddings.append([])

        if not examples:
            raise RuntimeError(f"No usable examples found in embedding map: {self._embed_map_path}")

        matrix = None
        if len(embeddings) == len(examples) and embeddings and all(embeddings):
            mat = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = mat / norms
        else:
            encoder = self._load_encoder()
            vecs = encoder.encode(vibes, normalize_embeddings=True)
            matrix = np.asarray(vecs, dtype=np.float32)

        self._examples = examples
        self._matrix = matrix
        return self

    def _lookup_impl(self, vibe: str) -> _RetrievalExample:
        if self._matrix is None or not self._examples:
            self._load()
        assert self._matrix is not None
        assert self._examples

        import numpy as np

        encoder = self._load_encoder()
        query = encoder.encode([vibe], normalize_embeddings=True)
        query_vec = np.asarray(query, dtype=np.float32)[0]
        scores = self._matrix @ query_vec
        best_idx = int(np.argmax(scores))
        return self._examples[best_idx]

    def lookup_top_k(self, vibe: str, k: int = 3) -> tuple[list[_RetrievalExample], list[float]]:
        """Return top-k nearest examples with normalized similarity weights."""
        index = self._load()
        return index._lookup_top_k_impl(vibe, k)

    def _lookup_top_k_impl(self, vibe: str, k: int) -> tuple[list[_RetrievalExample], list[float]]:
        if self._matrix is None or not self._examples:
            self._load()
        assert self._matrix is not None
        assert self._examples

        import numpy as np

        encoder = self._load_encoder()
        query = encoder.encode([vibe], normalize_embeddings=True)
        query_vec = np.asarray(query, dtype=np.float32)[0]
        scores = self._matrix @ query_vec

        actual_k = min(k, len(self._examples))
        top_indices = np.argsort(scores)[-actual_k:][::-1]
        top_scores = [float(scores[int(i)]) for i in top_indices]
        if all(s <= 0 for s in top_scores):
            return [self._examples[int(top_indices[0])]], [1.0]
        weights = _log_inverse_weights(top_scores)
        examples = [self._examples[int(i)] for i in top_indices]
        return examples, weights


def _resolve_embed_map_path() -> Path:
    explicit = os.environ.get("LATENTSCORE_EMBED_MAP", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    repo = os.environ.get("LATENTSCORE_EMBED_MAP_REPO", _DEFAULT_EMBED_MAP_REPO)
    filename = os.environ.get("LATENTSCORE_EMBED_MAP_FILE", _DEFAULT_EMBED_MAP_FILE)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - environment mismatch
        raise RuntimeError(
            "huggingface_hub is required to download the embedding map. "
            "Either install huggingface_hub or set LATENTSCORE_EMBED_MAP to a local file."
        ) from exc

    base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", "")).expanduser()
    if not base_dir:
        base_dir = Path.home() / ".cache" / "latentscore" / "models"
    cache_dir = base_dir / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


@functools.lru_cache(maxsize=1)
def _get_retrieval_index() -> _EmbeddingLookupIndex:
    return _EmbeddingLookupIndex(embed_map_path=_resolve_embed_map_path())


class EmbeddingLookupBaseline(ConfigBaseline):
    """Nearest-neighbor lookup baseline using the synthetic embedding map.

    This is the "fast" tier described in the paper: CPU-only retrieval from a
    synthetic vibe-to-config dataset using sentence-transformers embeddings.
    """

    def __init__(self) -> None:
        self._index = _get_retrieval_index()

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        example = self._index.lookup(vibe)
        return MusicConfigPromptPayload(
            thinking="Nearest-neighbor retrieval from embedding map (fast tier).",
            title=_make_title(vibe),
            config=example.config,
            palettes=example.palettes,
        )


# ---------------------------------------------------------------------------
# CLAP audio-embedding lookup baseline (fast_heavy tier)
# ---------------------------------------------------------------------------


class _ClapEmbeddingLookupIndex:
    """Index that matches text queries against pre-computed CLAP audio embeddings."""

    def __init__(self, *, embed_map_path: Path) -> None:
        self._embed_map_path = embed_map_path
        self._clap: Any | None = None
        self._examples: list[_RetrievalExample] = []
        self._matrix: Any | None = None

    def warmup(self) -> None:
        _ = self._load()

    def lookup(self, vibe: str) -> _RetrievalExample:
        index = self._load()
        return index._lookup_impl(vibe)

    def _load_clap(self) -> Any:
        if self._clap is not None:
            return self._clap
        try:
            import laion_clap  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("laion-clap is required for CLAP embedding baseline.") from exc
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        self._clap = model
        return model

    def _load(self) -> "_ClapEmbeddingLookupIndex":
        if self._matrix is not None and self._examples:
            return self

        import numpy as np
        from pydantic import ValidationError

        palette_fallback = RandomConfigBaseline(seed=42)

        examples: list[_RetrievalExample] = []
        embeddings: list[list[float]] = []

        with self._embed_map_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row_obj: object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row_obj, dict):
                    continue
                row = cast(dict[str, Any], row_obj)

                # Exclude TEST split to prevent data leakage.
                split = row.get("split")
                if isinstance(split, str) and split.upper() == "TEST":
                    continue

                vibe_value = row.get("vibe_original") or row.get("vibe") or row.get("vibe_noisy")
                config_value = row.get("config")
                embed_value = row.get("clap_audio_embedding")
                if not isinstance(vibe_value, str) or not vibe_value.strip():
                    continue
                if not isinstance(config_value, dict):
                    continue
                if not isinstance(embed_value, list):
                    continue

                try:
                    config = MusicConfigPrompt.model_validate(config_value)
                except ValidationError:
                    continue

                palettes_value = row.get("palettes")
                palettes: list[Palette] = []
                if isinstance(palettes_value, list):
                    palettes_list = cast(list[Any], palettes_value)
                    for item in palettes_list:
                        if not isinstance(item, dict):
                            palettes = []
                            break
                        try:
                            palettes.append(Palette.model_validate(item))
                        except ValidationError:
                            palettes = []
                            break

                if len(palettes) != 3:
                    palettes = [palette_fallback.random_palette() for _ in range(3)]

                try:
                    embed_list = cast(list[Any], embed_value)
                    embeddings.append([float(x) for x in embed_list])
                except (TypeError, ValueError):
                    continue

                examples.append(
                    _RetrievalExample(vibe=vibe_value, config=config, palettes=palettes)
                )

        if not examples:
            raise RuntimeError(
                f"No usable examples found in CLAP embedding map: {self._embed_map_path}"
            )

        mat = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = mat / norms

        self._examples = examples
        self._matrix = matrix
        return self

    def _lookup_impl(self, vibe: str) -> _RetrievalExample:
        if self._matrix is None or not self._examples:
            self._load()
        assert self._matrix is not None
        assert self._examples

        import numpy as np

        clap = self._load_clap()
        text_embed = clap.get_text_embedding([vibe])
        query_vec = np.asarray(text_embed[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm > 0:
            query_vec = query_vec / norm
        scores = self._matrix @ query_vec
        best_idx = int(np.argmax(scores))
        return self._examples[best_idx]


def _resolve_clap_embed_map_path() -> Path:
    explicit = os.environ.get("LATENTSCORE_CLAP_EMBED_MAP", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    repo = os.environ.get("LATENTSCORE_CLAP_EMBED_MAP_REPO", _DEFAULT_EMBED_MAP_REPO)
    filename = os.environ.get("LATENTSCORE_CLAP_EMBED_MAP_FILE", _DEFAULT_CLAP_EMBED_MAP_FILE)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download the CLAP embedding map."
        ) from exc

    base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", "")).expanduser()
    if not base_dir:
        base_dir = Path.home() / ".cache" / "latentscore" / "models"
    cache_dir = base_dir / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


@functools.lru_cache(maxsize=1)
def _get_clap_retrieval_index() -> _ClapEmbeddingLookupIndex:
    return _ClapEmbeddingLookupIndex(embed_map_path=_resolve_clap_embed_map_path())


class ClapEmbeddingLookupBaseline(ConfigBaseline):
    """Nearest-neighbor lookup using CLAP audio embeddings (fast_heavy tier).

    Matches input text against pre-computed CLAP audio embeddings of rendered
    configs, so the similarity captures what configs *sound* like rather than
    comparing text-to-text.
    """

    def __init__(self) -> None:
        self._index = _get_clap_retrieval_index()

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        example = self._index.lookup(vibe)
        return MusicConfigPromptPayload(
            thinking="CLAP audio-embedding retrieval from rendered configs (fast_heavy tier).",
            title=_make_title(vibe),
            config=example.config,
            palettes=example.palettes,
        )


# ---------------------------------------------------------------------------
# Interpolated embedding baseline: top-K blending
# ---------------------------------------------------------------------------

_INTERP_TOP_K = 3

_BASELINE_ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "tempo": {"very_slow": 0.15, "slow": 0.3, "medium": 0.5, "fast": 0.7, "very_fast": 0.9},
    "brightness": {"very_dark": 0.1, "dark": 0.3, "medium": 0.5, "bright": 0.7, "very_bright": 0.9},
    "space": {"dry": 0.1, "small": 0.3, "medium": 0.5, "large": 0.7, "vast": 0.95},
    "motion": {"static": 0.1, "slow": 0.3, "medium": 0.5, "fast": 0.7, "chaotic": 0.9},
    "stereo": {"mono": 0.0, "narrow": 0.25, "medium": 0.5, "wide": 0.75, "ultra_wide": 1.0},
    "echo": {"none": 0.0, "subtle": 0.25, "medium": 0.5, "heavy": 0.75, "infinite": 0.95},
    "human": {"robotic": 0.0, "tight": 0.15, "natural": 0.3, "loose": 0.5, "drunk": 0.8},
    "melody_density": {
        "very_sparse": 0.15,
        "sparse": 0.30,
        "medium": 0.50,
        "busy": 0.70,
        "very_busy": 0.85,
    },
    "syncopation": {"straight": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8},
    "swing": {"none": 0.0, "light": 0.2, "medium": 0.5, "heavy": 0.8},
    "motif_repeat_prob": {"rare": 0.2, "sometimes": 0.5, "often": 0.8},
    "step_bias": {"step": 0.9, "balanced": 0.7, "leapy": 0.4},
    "chromatic_prob": {"none": 0.0, "light": 0.05, "medium": 0.12, "heavy": 0.25},
    "cadence_strength": {"weak": 0.3, "medium": 0.6, "strong": 0.9},
    "chord_change_bars": {"very_slow": 4.0, "slow": 2.0, "medium": 1.0, "fast": 0.5},
}

_BASELINE_NOMINAL_FIELDS = frozenset(
    {
        "root",
        "mode",
        "bass",
        "pad",
        "melody",
        "rhythm",
        "texture",
        "accent",
        "attack",
        "grain",
        "melody_engine",
        "tension_curve",
        "harmony_style",
        "chord_extensions",
    }
)

_VALID_PHRASE_BARS = (2, 4, 8)
_MIN_DENSITY = 2
_MAX_DENSITY = 6
_MIN_OCTAVE = 1
_MAX_OCTAVE = 8


def _log_inverse_weights(scores: Sequence[float]) -> list[float]:
    """Convert cosine similarities to log-inverse weights: w_i = 1/|log(s_i)|.

    Amplifies small differences in high-similarity regimes where linear
    normalization produces near-uniform weights.
    """
    _EPS = 1e-9
    raw = [1.0 / max(abs(math.log(max(s, _EPS))), _EPS) for s in scores]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [w / total for w in raw]


def _bl_interp_ordinal(
    values: Sequence[str],
    weights: Sequence[float],
    label_map: dict[str, float],
) -> str:
    """Weighted average of ordinal labels via float mapping, snap to nearest."""
    avg = sum(label_map.get(v, 0.5) * w for v, w in zip(values, weights))
    return min(label_map, key=lambda k: abs(label_map[k] - avg))


def _bl_interp_nominal(
    values: Sequence[str],
    weights: Sequence[float],
    rng: random.Random,
) -> str:
    """Weighted random selection among nominal values."""
    return rng.choices(list(values), weights=list(weights), k=1)[0]


class EmbeddingInterpBaseline(ConfigBaseline):
    """Top-K embedding interpolation baseline.

    Finds the K nearest vibes in embedding space and blends their configs
    using similarity-weighted interpolation.
    """

    def __init__(self, top_k: int = _INTERP_TOP_K) -> None:
        self._index = _get_retrieval_index()
        self._top_k = top_k

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        examples, weights = self._index.lookup_top_k(vibe, self._top_k)
        if not examples:
            return EmbeddingLookupBaseline().generate(vibe)

        rng = random.Random(vibe)
        config_data: dict[str, Any] = {}

        # Ordinal fields: weighted avg in float space -> snap to nearest label
        for field, label_map in _BASELINE_ORDINAL_MAPS.items():
            vals = [str(getattr(e.config, field)) for e in examples]
            config_data[field] = _bl_interp_ordinal(vals, weights, label_map)

        # Nominal fields: weighted random selection
        for field in _BASELINE_NOMINAL_FIELDS:
            vals = [str(getattr(e.config, field)) for e in examples]
            config_data[field] = _bl_interp_nominal(vals, weights, rng)

        # density: Literal[2..6] -> weighted avg -> snap
        density_vals = [e.config.density for e in examples]
        density_avg = sum(v * w for v, w in zip(density_vals, weights))
        config_data["density"] = max(_MIN_DENSITY, min(_MAX_DENSITY, round(density_avg)))

        # phrase_len_bars: Literal[2,4,8] -> snap
        plb_vals = [e.config.phrase_len_bars for e in examples]
        plb_avg = sum(v * w for v, w in zip(plb_vals, weights))
        config_data["phrase_len_bars"] = min(_VALID_PHRASE_BARS, key=lambda v: abs(v - plb_avg))

        # register octaves: int, clamp 1-8, ensure min <= max
        min_oct_vals = [e.config.register_min_oct for e in examples]
        max_oct_vals = [e.config.register_max_oct for e in examples]
        min_oct = max(
            _MIN_OCTAVE,
            min(_MAX_OCTAVE, round(sum(v * w for v, w in zip(min_oct_vals, weights)))),
        )
        max_oct = max(
            _MIN_OCTAVE,
            min(_MAX_OCTAVE, round(sum(v * w for v, w in zip(max_oct_vals, weights)))),
        )
        if min_oct > max_oct:
            min_oct, max_oct = max_oct, min_oct
        config_data["register_min_oct"] = min_oct
        config_data["register_max_oct"] = max_oct

        # depth: bool -> weighted probability
        depth_vals = [e.config.depth for e in examples]
        config_data["depth"] = sum(w for v, w in zip(depth_vals, weights) if v) > 0.5

        config = MusicConfigPrompt.model_validate(config_data)
        palettes = examples[0].palettes

        return MusicConfigPromptPayload(
            thinking=f"Top-{self._top_k} embedding interpolation baseline.",
            title=_make_title(vibe),
            config=config,
            palettes=palettes,
        )


# ---------------------------------------------------------------------------
# Semantic matching baseline: zero-shot per-field selection
# ---------------------------------------------------------------------------

_SEMANTIC_SUFFIX: dict[str, str] = {
    "bass": " bass",
    "pad": " pad",
    "melody": " melody",
    "rhythm": " rhythm",
    "texture": " texture",
    "accent": " accent",
    "attack": " attack",
    "grain": " grain",
    "melody_engine": " engine",
    "tension_curve": " tension",
    "harmony_style": " harmony",
}


def _bl_value_to_semantic_text(field: str, value: object) -> str:
    """Convert a config field value to descriptive text for semantic embedding."""
    if field == "root":
        return f"key of {str(value).replace('_sharp', ' sharp')}"
    if field == "density":
        return f"density {value}"
    if field == "phrase_len_bars":
        return f"{value} bar phrase"
    base = str(value).replace("_", " ")
    suffix = _SEMANTIC_SUFFIX.get(field, "")
    return base + suffix


class SemanticMatchBaseline(ConfigBaseline):
    """Zero-shot semantic matching baseline.

    For each config field, picks the enum value whose text embedding
    is most similar to the input vibe. No training data needed.
    """

    def __init__(self) -> None:
        self._encoder: Any | None = None
        self._cached_field_embeddings: dict[str, tuple[list[object], Any]] | None = None

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - environment mismatch
            raise RuntimeError("sentence-transformers is required for semantic baseline.") from exc
        self._encoder = cast(Any, SentenceTransformer(_DEFAULT_EMBED_MODEL))
        return cast(Any, self._encoder)

    def _get_field_embeddings(self) -> dict[str, tuple[list[object], Any]]:
        if self._cached_field_embeddings is not None:
            return self._cached_field_embeddings

        import numpy as np

        encoder = self._load_encoder()

        # All label fields: field_name → (values_sequence, text_formatter)
        label_fields: dict[str, Sequence[object]] = {
            "tempo": TEMPO_VALUES,
            "brightness": BRIGHTNESS_VALUES,
            "space": SPACE_VALUES,
            "motion": MOTION_VALUES,
            "stereo": STEREO_VALUES,
            "echo": ECHO_VALUES,
            "human": HUMAN_FEEL_VALUES,
            "melody_density": MELODY_DENSITY_VALUES,
            "syncopation": SYNCOPATION_VALUES,
            "swing": SWING_VALUES,
            "motif_repeat_prob": MOTIF_REPEAT_VALUES,
            "step_bias": STEP_BIAS_VALUES,
            "chromatic_prob": CHROMATIC_VALUES,
            "cadence_strength": CADENCE_VALUES,
            "chord_change_bars": CHORD_CHANGE_VALUES,
            "root": ROOT_VALUES,
            "mode": MODE_VALUES,
            "bass": BASS_STYLES,
            "pad": PAD_STYLES,
            "melody": MELODY_STYLES,
            "rhythm": RHYTHM_STYLES,
            "texture": TEXTURE_STYLES,
            "accent": ACCENT_STYLES,
            "attack": ATTACK_STYLES,
            "grain": GRAIN_STYLES,
            "melody_engine": MELODY_ENGINES,
            "tension_curve": TENSION_CURVES,
            "harmony_style": HARMONY_STYLES,
            "chord_extensions": CHORD_EXTENSIONS_VALUES,
            "density": DENSITY_VALUES,
            "phrase_len_bars": PHRASE_LENGTH_BARS,
        }

        # Special fields with custom text
        special_fields: dict[str, list[tuple[object, str]]] = {
            "depth": [
                (True, "deep layered sound"),
                (False, "flat simple sound"),
            ],
            "register_min_oct": [(i, f"octave {i} register") for i in range(1, 9)],
            "register_max_oct": [(i, f"octave {i} register") for i in range(1, 9)],
        }

        result: dict[str, tuple[list[object], Any]] = {}

        for field, vals in label_fields.items():
            values: list[object] = list(vals)
            texts = [_bl_value_to_semantic_text(field, v) for v in vals]
            embeddings = encoder.encode(texts, normalize_embeddings=True)
            result[field] = (values, np.asarray(embeddings, dtype=np.float32))

        for field, pairs in special_fields.items():
            values = [p[0] for p in pairs]
            texts = [p[1] for p in pairs]
            embeddings = encoder.encode(texts, normalize_embeddings=True)
            result[field] = (values, np.asarray(embeddings, dtype=np.float32))

        self._cached_field_embeddings = result
        return result

    def generate(self, vibe: str) -> MusicConfigPromptPayload:
        import numpy as np

        encoder = self._load_encoder()
        field_embeddings = self._get_field_embeddings()

        query = encoder.encode([vibe], normalize_embeddings=True)
        query_vec = np.asarray(query, dtype=np.float32)[0]

        config_data: dict[str, Any] = {}
        for field, (values, matrix) in field_embeddings.items():
            scores = matrix @ query_vec
            best_idx = int(np.argmax(scores))
            config_data[field] = values[best_idx]

        # Ensure register_min_oct <= register_max_oct
        min_oct = config_data.get("register_min_oct")
        max_oct = config_data.get("register_max_oct")
        if isinstance(min_oct, int) and isinstance(max_oct, int) and min_oct > max_oct:
            config_data["register_min_oct"] = max_oct
            config_data["register_max_oct"] = min_oct

        config = MusicConfigPrompt.model_validate(config_data)

        palette_gen = RandomConfigBaseline(seed=42)
        return MusicConfigPromptPayload(
            thinking="Zero-shot semantic matching: each field chosen by embedding similarity.",
            title=_make_title(vibe),
            config=config,
            palettes=[palette_gen.random_palette() for _ in range(3)],
        )


# Registry of available baselines
BASELINES: dict[str, type[ConfigBaseline]] = {
    "random": RandomConfigBaseline,
    "rule_based": RuleBasedBaseline,
    "mode": ModeConfigBaseline,
    "retrieval": EmbeddingLookupBaseline,
    "embedding_lookup": EmbeddingLookupBaseline,
    "embedding_interp": EmbeddingInterpBaseline,
    "semantic_match": SemanticMatchBaseline,
    "clap_embedding_lookup": ClapEmbeddingLookupBaseline,
}

_EMBEDDING_LOOKUP_NAMES = frozenset({
    "retrieval", "embedding_lookup", "embedding_interp", "clap_embedding_lookup",
})

EMBEDDING_LOOKUP_WARNING = (
    "The embedding_lookup baseline retrieves configs from a fixed synthetic dataset "
    "(HF: guprab/latentscore-data) generated by a teacher LLM "
    "(gemini/gemini-3-flash-preview). "
    "TEST-split rows are excluded to prevent data leakage."
)


def is_embedding_lookup(name: str) -> bool:
    """Check if a baseline name refers to the embedding lookup baseline."""
    return name in _EMBEDDING_LOOKUP_NAMES


def get_baseline(name: str) -> ConfigBaseline:
    """Get a baseline instance by name."""
    if name not in BASELINES:
        available = ", ".join(BASELINES.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    return BASELINES[name]()


def list_baselines() -> list[str]:
    """List available baseline names."""
    return list(BASELINES.keys())
