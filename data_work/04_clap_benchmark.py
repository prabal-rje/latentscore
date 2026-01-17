"""Benchmark config generators with LAION-CLAP scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

if __package__ is None and __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import (
    LocalHFClient,
    format_prompt_json,
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
)
from data_work.lib.music_prompt import build_music_prompt
from data_work.lib.music_schema import MusicConfigPromptPayload
from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble

LOGGER = logging.getLogger("data_work.clap_benchmark")

DEFAULT_SYSTEM_PROMPT = build_music_prompt()
DEFAULT_VIBE_FIELD = "vibe_original"
DEFAULT_OUTPUT_DIR = Path("data_work/.benchmarks")
DEFAULT_DURATION = 12.0
DEFAULT_MAX_NEW_TOKENS = 512

BAD_CONCEPTS = [
    "bad",
    "terrible",
    "awful",
    "discordant",
    "unharmonious",
    "cacophony",
    "noise",
    "unpleasant",
    "harsh",
    "grating",
    "off-key",
    "out of tune",
    "distorted badly",
    "broken audio",
    "annoying sound",
    "painful to listen to",
    "low quality audio",
]


class BenchmarkSource(BaseModel):
    """Configuration for a benchmark source (model or dataset field)."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    label: str
    model: str | None = None
    field: str | None = None
    api_base: str | None = None


class ClapScore(BaseModel):
    """CLAP scoring result for a single vibe-audio pair."""

    model_config = ConfigDict(extra="forbid")

    audio_text_similarity: float
    audio_bad_similarity: float
    text_bad_similarity: float
    excess_badness: float
    penalty: float
    raw_score: float
    final_reward: float


class BenchmarkResult(BaseModel):
    """Result from benchmarking a single vibe-config pair."""

    model_config = ConfigDict(extra="forbid")

    vibe: str
    model: str
    source_kind: str
    config_field: str | None = None
    id_in_dataset: str | int | None = None
    dataset: str | None = None
    split: str | None = None
    config: dict[str, Any] | None = None
    config_error: str | None = None
    clap_reward: float | None = None
    clap_details: ClapScore | None = None
    audio_path: str | None = None


class ModelSummary(BaseModel):
    """Summary statistics for a model's benchmark results."""

    model_config = ConfigDict(extra="forbid")

    count: int
    mean_clap_reward: float


class ClapScorer:
    def __init__(self) -> None:
        try:
            import laion_clap  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit(
                "laion_clap is required. Install via data_work/requirements.txt."
            ) from exc

        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()
        self._bad_embedding: np.ndarray | None = None

    def _get_bad_embedding(self) -> np.ndarray:
        if self._bad_embedding is None:
            embeddings = self._model.get_text_embedding(BAD_CONCEPTS)
            self._bad_embedding = embeddings.mean(axis=0, keepdims=True)
        return self._bad_embedding

    @staticmethod
    def _softplus(value: float, beta: float = 1.0) -> float:
        return (1.0 / beta) * math.log(1.0 + math.exp(beta * value))

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        numerator = float(a @ b.T)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return numerator / denom

    def score(self, vibe: str, audio_path: str) -> ClapScore:
        """Score audio against vibe text using CLAP embeddings."""
        text_embed = self._model.get_text_embedding([vibe])
        audio_embed = self._model.get_audio_embedding_from_filelist([audio_path])
        bad_embed = self._get_bad_embedding()

        audio_text_sim = self._cosine_sim(audio_embed, text_embed)
        audio_bad_sim = self._cosine_sim(audio_embed, bad_embed)
        text_bad_sim = self._cosine_sim(text_embed, bad_embed)
        excess_badness = audio_bad_sim - text_bad_sim
        penalty = self._softplus(excess_badness, beta=5.0)
        raw_score = self._softplus(audio_text_sim, beta=2.0) - 0.5 * penalty
        reward = float(np.clip(math.exp(raw_score - 1.0), 0.0, 1.0))

        return ClapScore(
            audio_text_similarity=float(audio_text_sim),
            audio_bad_similarity=float(audio_bad_sim),
            text_bad_similarity=float(text_bad_sim),
            excess_badness=float(excess_badness),
            penalty=float(penalty),
            raw_score=float(raw_score),
            final_reward=reward,
        )


def parse_source_entries(values: Sequence[str], kind: str) -> list[BenchmarkSource]:
    sources: list[BenchmarkSource] = []
    for value in values:
        name, _, label = value.partition(":")
        clean_label = label or name
        if not name:
            raise SystemExit(f"Invalid {kind} entry: {value}")
        sources.append(BenchmarkSource(kind=kind, label=clean_label, model=name))
    return sources


def parse_field_entries(values: Sequence[str]) -> list[BenchmarkSource]:
    sources: list[BenchmarkSource] = []
    for value in values:
        name, _, label = value.partition(":")
        clean_label = label or name
        if not name:
            raise SystemExit(f"Invalid dataset field entry: {value}")
        sources.append(BenchmarkSource(kind="dataset", label=clean_label, field=name))
    return sources


def _extract_config(value: Any) -> dict[str, Any]:
    """Extract config dict from various input formats."""
    match value:
        case None:
            raise ValueError("Missing config payload.")
        case str():
            parsed = json.loads(value)
        case BaseModel():
            parsed = value.model_dump()
        case _:
            parsed = value

    match parsed:
        case {"config": dict() as inner_config}:
            return inner_config
        case dict():
            return parsed
        case _:
            raise ValueError("Config payload was not a JSON object.")


def _config_to_audio(config_payload: Mapping[str, Any], duration: float) -> np.ndarray:
    config_dict = dict(config_payload)
    # Try MusicConfigPrompt first (string labels like "sparse", "light")
    # then convert to MusicConfig via to_config() which handles the label->float mapping
    music_config: MusicConfig
    try:
        prompt_config: MusicConfigPrompt = MusicConfigPrompt.model_validate(config_dict)
        music_config = prompt_config.to_config()
    except Exception:
        # Fallback: maybe it's already in MusicConfig format (numeric values)
        music_config = MusicConfig.model_validate(config_dict)
    internal = music_config.to_internal()
    return assemble(internal, duration=duration)


def _safe_filename(value: object) -> str:
    text = str(value)
    return text.replace("/", "__").replace(":", "__").replace(" ", "_").replace("|", "__")


def _iter_rows(path: Path, limit: int, split: str | None) -> Iterable[dict[str, Any]]:
    count = 0
    for row in iter_jsonl(path):
        if split and row.get("split") != split:
            continue
        yield row
        count += 1
        if limit and count >= limit:
            break


async def _generate_litellm_payload(
    *,
    vibe: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    system_prompt: str,
) -> MusicConfigPromptPayload:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": vibe},
    ]
    return await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=MusicConfigPromptPayload,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )


def _write_results(path: Path, rows: Iterable[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _summarize(results: Sequence[BenchmarkResult]) -> dict[str, ModelSummary]:
    """Summarize benchmark results by model."""
    by_model: dict[str, list[float]] = {}
    for result in results:
        if result.clap_reward is not None:
            by_model.setdefault(result.model, []).append(result.clap_reward)

    return {
        model: ModelSummary(
            count=len(scores),
            mean_clap_reward=sum(scores) / len(scores) if scores else 0.0,
        )
        for model, scores in by_model.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark configs with LAION-CLAP scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file containing test rows (e.g., data_work/.processed/TEST.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write benchmark outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of rows to benchmark (0 = all).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name to filter rows (e.g., TEST).",
    )
    parser.add_argument(
        "--vibe-field",
        type=str,
        default=DEFAULT_VIBE_FIELD,
        help="Column name for clean vibe text.",
    )
    parser.add_argument(
        "--dataset-field",
        action="append",
        default=[],
        help=(
            "Use configs from a dataset column. Format: field[:label]. "
            "Example: config_payload:synthetic"
        ),
    )
    parser.add_argument(
        "--litellm-model",
        action="append",
        default=[],
        help="Litellm model to benchmark. Format: model[:label].",
    )
    parser.add_argument(
        "--local-model",
        action="append",
        default=[],
        help="Local HF model path to benchmark. Format: path[:label].",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline type to benchmark (random, rule_based). Format: type[:label].",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for LLM-based config generation.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Audio duration in seconds for each sample.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LiteLLM providers (optional).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Env var name to read API key from.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Override API base URL for LiteLLM models.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load for API keys.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON dict of extra LiteLLM request kwargs.",
    )
    parser.add_argument(
        "--local-device",
        type=str,
        default=None,
        help="Device for local HF models (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--local-max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max tokens to generate for local models.",
    )
    parser.add_argument(
        "--local-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for local models (0 = greedy).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep generated audio files under output-dir/audio.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    load_env_file(args.env_file)

    dataset_sources = parse_field_entries(args.dataset_field)
    litellm_sources = parse_source_entries(args.litellm_model, "litellm")
    local_sources = parse_source_entries(args.local_model, "local")
    baseline_sources = parse_source_entries(args.baseline, "baseline")
    sources = [*dataset_sources, *litellm_sources, *local_sources, *baseline_sources]
    if not sources:
        raise SystemExit(
            "At least one --dataset-field, --litellm-model, --local-model, or --baseline is required."
        )

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.jsonl"
    summary_path = output_dir / "benchmark_summary.json"

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit("--model-kwargs must be valid JSON.") from exc
    if not isinstance(model_kwargs, dict):
        raise SystemExit("--model-kwargs must be a JSON object.")

    litellm_models: list[tuple[str, str | None]] = []
    for source in litellm_sources:
        assert source.model is not None
        model, api_base = normalize_model_and_base(source.model, args.api_base)
        source.model = model
        source.api_base = api_base
        litellm_models.append((model, api_base))

    api_key = resolve_api_key_for_models(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        models=litellm_models,
    )

    local_clients: dict[str, LocalHFClient] = {}
    for source in local_sources:
        assert source.model is not None
        local_clients[source.label] = LocalHFClient(
            source.model,
            device=args.local_device,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
        )

    scorer = ClapScorer()
    results: list[BenchmarkResult] = []

    audio_dir = output_dir / "audio"
    if args.keep_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    for row in _iter_rows(input_path, args.limit, args.split):
        vibe = row.get(args.vibe_field)
        if not isinstance(vibe, str) or not vibe.strip():
            LOGGER.warning("Skipping row without vibe text.")
            continue

        for source in sources:
            config_payload = None
            error = None

            try:
                match source.kind:
                    case "dataset":
                        assert source.field is not None
                        config_payload = _extract_config(row.get(source.field))
                    case "litellm":
                        assert source.model is not None
                        payload = asyncio.run(
                            _generate_litellm_payload(
                                vibe=vibe,
                                model=source.model,
                                api_key=api_key,
                                api_base=source.api_base,
                                model_kwargs=model_kwargs,
                                system_prompt=args.system_prompt,
                            )
                        )
                        config_payload = payload.config.model_dump()
                    case "local":
                        client = local_clients[source.label]
                        prompt = format_prompt_json(args.system_prompt, vibe)
                        payload = client.generate_structured(prompt, MusicConfigPromptPayload)
                        config_payload = payload.config.model_dump()
                    case "baseline":
                        from data_work.lib.baselines import get_baseline

                        assert source.model is not None
                        baseline = get_baseline(source.model)
                        baseline_payload = baseline.generate(vibe)
                        config_payload = baseline_payload.config.model_dump()
                    case _:
                        error = f"Unsupported source kind: {source.kind}"
            except Exception as exc:
                error = str(exc)

            if config_payload is None:
                results.append(
                    BenchmarkResult(
                        model=source.label,
                        source_kind=source.kind,
                        config_field=source.field,
                        config_error=error or "missing_config",
                        vibe=vibe,
                        id_in_dataset=row.get("id_in_dataset"),
                        dataset=row.get("dataset"),
                        split=row.get("split"),
                    )
                )
                continue

            try:
                audio = _config_to_audio(config_payload, args.duration)
                audio_record = None
                if args.keep_audio:
                    safe_label = _safe_filename(source.label)
                    safe_id = _safe_filename(row.get("id_in_dataset"))
                    audio_path = audio_dir / f"{safe_label}_{safe_id}.wav"
                    write_wav(audio_path, audio)
                    audio_file = str(audio_path)
                    audio_record = audio_file
                    clap_metrics = scorer.score(vibe, audio_file)
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        audio_path = Path(tmpdir) / "sample.wav"
                        write_wav(audio_path, audio)
                        audio_file = str(audio_path)
                        clap_metrics = scorer.score(vibe, audio_file)
                results.append(
                    BenchmarkResult(
                        model=source.label,
                        source_kind=source.kind,
                        config_field=source.field,
                        vibe=vibe,
                        id_in_dataset=row.get("id_in_dataset"),
                        dataset=row.get("dataset"),
                        split=row.get("split"),
                        config=config_payload,
                        clap_reward=clap_metrics.final_reward,
                        clap_details=clap_metrics,
                        audio_path=audio_record,
                    )
                )
            except Exception as exc:
                results.append(
                    BenchmarkResult(
                        model=source.label,
                        source_kind=source.kind,
                        config_field=source.field,
                        config_error=str(exc),
                        vibe=vibe,
                        id_in_dataset=row.get("id_in_dataset"),
                        dataset=row.get("dataset"),
                        split=row.get("split"),
                        config=config_payload,
                    )
                )

    _write_results(output_path, results)
    summary = _summarize(results)
    summary_dict = {model: stats.model_dump() for model, stats in summary.items()}
    summary_path.write_text(json.dumps(summary_dict, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote results to %s", output_path)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
