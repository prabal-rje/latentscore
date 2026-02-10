"""Benchmark config generators with LAION-CLAP scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

if __package__ is None and __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import shared ClapScore + ClapScorer
from data_work.lib.clap_scorer import ClapScore, ClapScorer
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import (
    LocalHFClient,
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
    wrap_vibe_for_chat,
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
DEFAULT_MAX_NEW_TOKENS = 3000


class BenchmarkSource(BaseModel):
    """Configuration for a benchmark source (model or dataset field)."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    label: str
    model: str | None = None
    field: str | None = None
    api_base: str | None = None


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
    elapsed_s: float | None = None
    config_gen_s: float | None = None
    audio_synth_s: float | None = None
    success: bool = False


class ModelSummary(BaseModel):
    """Summary statistics for a model's benchmark results."""

    model_config = ConfigDict(extra="forbid")

    total: int
    succeeded: int
    failed: int
    success_rate: float
    mean_clap_reward: float
    mean_elapsed_s: float
    mean_config_gen_s: float
    mean_audio_synth_s: float


class HumanEvalModelEntry(BaseModel):
    """One model's output for a single vibe in the human-eval dataset."""

    model_config = ConfigDict(extra="forbid")

    source_kind: str
    config: dict[str, Any] | None = None
    config_error: str | None = None
    audio_path: str | None = None
    clap_reward: float | None = None
    success: bool = False


class HumanEvalRow(BaseModel):
    """One row of the human-eval dataset — one vibe, all models side-by-side."""

    model_config = ConfigDict(extra="forbid")

    vibe: str
    id_in_dataset: str | int | None = None
    dataset: str | None = None
    split: str | None = None
    models: dict[str, HumanEvalModelEntry]


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

    if isinstance(parsed, dict):
        parsed_dict = cast(dict[str, Any], parsed)
        inner = parsed_dict.get("config")
        if isinstance(inner, dict):
            return cast(dict[str, Any], inner)
        return parsed_dict

    raise ValueError("Config payload was not a JSON object.")


def _config_to_audio(config_payload: Mapping[str, Any], duration: float) -> NDArray[np.float64]:
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


_SCHEMA_SECTION_RE = re.compile(r"\n*<schema>\n.*?\n</schema>", re.DOTALL)


def _strip_schema_section(prompt: str) -> str:
    """Remove <schema>...</schema> from the prompt.

    LiteLLM models get the schema via ``response_format`` (structured output),
    so embedding it in the prompt is redundant for large models.
    """
    return _SCHEMA_SECTION_RE.sub("", prompt).strip()


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
        {"role": "system", "content": _strip_schema_section(system_prompt)},
        {"role": "user", "content": wrap_vibe_for_chat(vibe)},
    ]
    return await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=MusicConfigPromptPayload,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )


T = TypeVar("T")


def _split_rows(rows: list[T], n: int) -> list[list[T]]:
    """Split rows into n roughly-equal chunks."""
    k, remainder = divmod(len(rows), n)
    chunks: list[list[T]] = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < remainder else 0)
        chunks.append(rows[start:end])
        start = end
    return chunks


@dataclass
class WorkerConfig:
    """Picklable config for a worker process — no model objects, just primitives."""

    worker_id: int
    rows: list[dict[str, Any]]
    sources: list[dict[str, Any]]  # BenchmarkSource.model_dump() list
    system_prompt: str
    duration: float
    vibe_field: str
    keep_audio: bool
    audio_dir: str | None = None
    clap_vibe_prefix: str = ""
    # LiteLLM
    api_key: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # Local model
    local_device: str | None = None
    local_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    local_temperature: float = 0.0
    local_force_cpu: bool = False
    local_4bit: bool = False
    local_no_int8: bool = False


def _worker_fn(config: WorkerConfig) -> list[dict[str, Any]]:
    """Run in a child process: init models, process chunk, return serializable results."""
    worker_tag = f"worker-{config.worker_id}"
    logger = logging.getLogger(f"data_work.clap_benchmark.{worker_tag}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sources = [BenchmarkSource(**s) for s in config.sources]

    # --- init models per-worker ---
    local_clients: dict[str, LocalHFClient] = {}
    for source in sources:
        if source.kind == "local":
            assert source.model is not None
            local_clients[source.label] = LocalHFClient(
                source.model,
                device=config.local_device,
                max_new_tokens=config.local_max_new_tokens,
                temperature=config.local_temperature,
                force_cpu=config.local_force_cpu,
                quantize_4bit=config.local_4bit,
                no_int8=config.local_no_int8,
            )

    baseline_clients: dict[str, Any] = {}
    baseline_sources = [s for s in sources if s.kind == "baseline"]
    if baseline_sources:
        from data_work.lib.baselines import get_baseline

        for source in baseline_sources:
            assert source.model is not None
            baseline_clients[source.label] = get_baseline(source.model)

    scorer = ClapScorer()

    audio_dir = Path(config.audio_dir) if config.audio_dir else None
    if config.keep_audio and audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)

    # --- process rows ---
    results: list[dict[str, Any]] = []
    for row_idx, row in enumerate(config.rows):
        vibe = row.get(config.vibe_field)
        if not isinstance(vibe, str) or not vibe.strip():
            logger.warning("[%s] Skipping row without vibe text.", worker_tag)
            continue
        clap_vibe = f"{config.clap_vibe_prefix}{vibe}" if config.clap_vibe_prefix else vibe

        for source in sources:
            t0 = time.monotonic()
            logger.info(
                "[%s] row %d/%d [%s] vibe: %.60s...",
                worker_tag,
                row_idx + 1,
                len(config.rows),
                source.label,
                vibe,
            )
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
                                api_key=config.api_key,
                                api_base=source.api_base,
                                model_kwargs=config.model_kwargs,
                                system_prompt=config.system_prompt,
                            )
                        )
                        config_payload = payload.config.model_dump()
                    case "local":
                        client = local_clients[source.label]
                        prompt = client.format_chat_prompt(
                            system_prompt=config.system_prompt,
                            user_prompt=wrap_vibe_for_chat(vibe),
                        )
                        payload = client.generate_structured(prompt, MusicConfigPromptPayload)
                        config_payload = payload.config.model_dump()
                    case "baseline":
                        assert source.model is not None
                        baseline = baseline_clients[source.label]
                        baseline_payload = baseline.generate(vibe)
                        config_payload = baseline_payload.config.model_dump()
                    case _:
                        error = f"Unsupported source kind: {source.kind}"
            except Exception as exc:
                error = str(exc)

            t_config = time.monotonic() - t0

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
                        elapsed_s=t_config,
                        config_gen_s=t_config,
                    ).model_dump()
                )
                continue

            try:
                t_synth_start = time.monotonic()
                audio = _config_to_audio(config_payload, config.duration)
                t_synth = time.monotonic() - t_synth_start

                audio_record = None
                if config.keep_audio and audio_dir:
                    safe_label = _safe_filename(source.label)
                    safe_id = _safe_filename(row.get("id_in_dataset") or "")
                    audio_path = audio_dir / f"{row_idx:04d}_{safe_label}_{safe_id}.wav"
                    write_wav(audio_path, audio)
                    audio_file = str(audio_path)
                    audio_record = audio_file
                    clap_metrics = scorer.score(clap_vibe, audio_file)
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        audio_path = Path(tmpdir) / "sample.wav"
                        write_wav(audio_path, audio)
                        audio_file = str(audio_path)
                        clap_metrics = scorer.score(clap_vibe, audio_file)
                elapsed = time.monotonic() - t0
                logger.info(
                    "[%s] [%s] done in %.1fs (config=%.1fs, synth=%.1fs)",
                    worker_tag, source.label, elapsed, t_config, t_synth,
                )
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
                        elapsed_s=elapsed,
                        config_gen_s=t_config,
                        audio_synth_s=t_synth,
                        success=True,
                    ).model_dump()
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
                        elapsed_s=time.monotonic() - t0,
                        config_gen_s=t_config,
                    ).model_dump()
                )

    logger.info("[%s] Finished processing %d rows.", worker_tag, len(config.rows))
    return results


def _write_results(path: Path, rows: Iterable[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_human_eval(results: Sequence[BenchmarkResult]) -> list[HumanEvalRow]:
    """Pivot results into human-eval rows: one per vibe, all models side-by-side."""
    rows: dict[str | int | None, HumanEvalRow] = {}
    order: list[str | int | None] = []
    for r in results:
        key = r.id_in_dataset
        if key not in rows:
            order.append(key)
            rows[key] = HumanEvalRow(
                vibe=r.vibe,
                id_in_dataset=r.id_in_dataset,
                dataset=r.dataset,
                split=r.split,
                models={},
            )
        rows[key].models[r.model] = HumanEvalModelEntry(
            source_kind=r.source_kind,
            config=r.config,
            config_error=r.config_error,
            audio_path=r.audio_path,
            clap_reward=r.clap_reward,
            success=r.success,
        )
    return [rows[k] for k in order]


def _summarize(results: Sequence[BenchmarkResult]) -> dict[str, ModelSummary]:
    """Summarize benchmark results by model (only successful samples count for averages)."""
    totals: dict[str, int] = {}
    successes: dict[str, list[BenchmarkResult]] = {}

    for result in results:
        totals[result.model] = totals.get(result.model, 0) + 1
        if result.success:
            successes.setdefault(result.model, []).append(result)

    summaries: dict[str, ModelSummary] = {}
    for model, total in totals.items():
        ok = successes.get(model, [])
        succeeded = len(ok)
        failed = total - succeeded
        summaries[model] = ModelSummary(
            total=total,
            succeeded=succeeded,
            failed=failed,
            success_rate=succeeded / total if total else 0.0,
            mean_clap_reward=_mean_or_zero([r.clap_reward for r in ok if r.clap_reward is not None]),
            mean_elapsed_s=_mean_or_zero([r.elapsed_s for r in ok if r.elapsed_s is not None]),
            mean_config_gen_s=_mean_or_zero([r.config_gen_s for r in ok if r.config_gen_s is not None]),
            mean_audio_synth_s=_mean_or_zero(
                [r.audio_synth_s for r in ok if r.audio_synth_s is not None]
            ),
        )
    return summaries


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
        help=(
            "Baseline type to benchmark. Format: type[:label]. "
            "Available: random, rule_based, mode, embedding_lookup. "
            "embedding_lookup retrieves configs from a fixed synthetic dataset "
            "(HF: guprab/latentscore-data), excluding TEST-split rows to prevent data leakage."
        ),
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
        "--clap-vibe-prefix",
        type=str,
        default="",
        help="Prefix prepended to vibe text for CLAP scoring (not config generation). "
        "E.g. 'electronic music representing: '",
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
        default=1.0,
        help="Sampling temperature for local models (default: 1.0 per Unsloth Gemma 3 recommendations).",
    )
    parser.add_argument(
        "--local-force-cpu",
        action="store_true",
        help="Force CPU inference for local models (ignore CUDA even if available).",
    )
    parser.add_argument(
        "--local-4bit",
        action="store_true",
        help="Load local models with 4-bit NF4 quantization (requires CUDA + bitsandbytes).",
    )
    parser.add_argument(
        "--local-no-int8",
        action="store_true",
        help="Disable automatic int8 quantization on Apple Silicon (use full float32).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep generated audio files under output-dir/audio.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Each loads its own models. (default: 1 = sequential)",
    )
    return parser


def _build_worker_config(
    *,
    worker_id: int,
    rows: list[dict[str, Any]],
    sources: list[BenchmarkSource],
    args: argparse.Namespace,
    api_key: str | None,
    model_kwargs: dict[str, Any],
    audio_dir: Path | None,
) -> WorkerConfig:
    return WorkerConfig(
        worker_id=worker_id,
        rows=rows,
        sources=[s.model_dump() for s in sources],
        system_prompt=args.system_prompt,
        duration=args.duration,
        vibe_field=args.vibe_field,
        keep_audio=args.keep_audio,
        audio_dir=str(audio_dir) if audio_dir else None,
        clap_vibe_prefix=args.clap_vibe_prefix,
        api_key=api_key,
        model_kwargs=model_kwargs,
        local_device=args.local_device,
        local_max_new_tokens=args.local_max_new_tokens,
        local_temperature=args.local_temperature,
        local_force_cpu=args.local_force_cpu,
        local_4bit=args.local_4bit,
        local_no_int8=args.local_no_int8,
    )


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
    model_kwargs = cast(dict[str, Any], model_kwargs)

    # Normalize litellm model names + resolve API base
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

    # Emit embedding_lookup warnings from main process
    if baseline_sources:
        from data_work.lib.baselines import EMBEDDING_LOOKUP_WARNING, is_embedding_lookup

        for source in baseline_sources:
            assert source.model is not None
            if is_embedding_lookup(source.model):
                LOGGER.warning(EMBEDDING_LOOKUP_WARNING)

    audio_dir = output_dir / "audio" if args.keep_audio else None

    # Collect all rows upfront (needed for splitting across workers)
    all_rows = list(_iter_rows(input_path, args.limit, args.split))
    LOGGER.info("Loaded %d rows to benchmark across %d source(s).", len(all_rows), len(sources))

    if not all_rows:
        raise SystemExit("No rows matched the input/split/limit criteria.")

    n_workers = min(args.workers, len(all_rows))

    if n_workers <= 1:
        # --- Single-process path (no multiprocessing overhead) ---
        wc = _build_worker_config(
            worker_id=0,
            rows=all_rows,
            sources=sources,
            args=args,
            api_key=api_key,
            model_kwargs=model_kwargs,
            audio_dir=audio_dir,
        )
        raw_results = _worker_fn(wc)
    else:
        # --- Multi-process path ---
        LOGGER.info("Spawning %d worker processes...", n_workers)
        chunks = _split_rows(all_rows, n_workers)
        worker_configs = [
            _build_worker_config(
                worker_id=i,
                rows=chunk,
                sources=sources,
                args=args,
                api_key=api_key,
                model_kwargs=model_kwargs,
                audio_dir=audio_dir,
            )
            for i, chunk in enumerate(chunks)
        ]
        with multiprocessing.Pool(processes=n_workers) as pool:
            chunk_results = pool.map(_worker_fn, worker_configs)
        raw_results = [r for chunk in chunk_results for r in chunk]

    # Strip computed fields that don't survive Pydantic round-trip (ClapScore.final_score)
    for r in raw_results:
        if isinstance(r.get("clap_details"), dict):
            r["clap_details"].pop("final_score", None)
    results = [BenchmarkResult(**r) for r in raw_results]
    _write_results(output_path, results)
    summary = _summarize(results)
    summary_dict = {model_name: stats.model_dump() for model_name, stats in summary.items()}
    summary_path.write_text(json.dumps(summary_dict, indent=2) + "\n", encoding="utf-8")
    # Write human-eval dataset (pivoted: one row per vibe, all models side-by-side)
    human_eval_path = output_dir / "human_eval.jsonl"
    human_eval_rows = _build_human_eval(results)
    with human_eval_path.open("w", encoding="utf-8") as f:
        for row in human_eval_rows:
            f.write(row.model_dump_json() + "\n")

    LOGGER.info("Wrote %d results to %s", len(results), output_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote %d human-eval rows to %s", len(human_eval_rows), human_eval_path)


if __name__ == "__main__":
    main()
