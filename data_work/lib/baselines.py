"""Baseline config generators for evaluation comparison."""

from __future__ import annotations

import functools
import json
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, cast, get_args  # noqa: F401 (TypeVar used in generic)

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
_DEFAULT_EMBED_MAP_FILE = "2026-01-26_scored/_progress_embeddings.jsonl"


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
            raise RuntimeError(
                "sentence-transformers is required for retrieval baseline."
            ) from exc
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


# Registry of available baselines
BASELINES: dict[str, type[ConfigBaseline]] = {
    "random": RandomConfigBaseline,
    "rule_based": RuleBasedBaseline,
    "mode": ModeConfigBaseline,
    "retrieval": EmbeddingLookupBaseline,
}


def get_baseline(name: str) -> ConfigBaseline:
    """Get a baseline instance by name."""
    if name not in BASELINES:
        available = ", ".join(BASELINES.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    return BASELINES[name]()


def list_baselines() -> list[str]:
    """List available baseline names."""
    return list(BASELINES.keys())
