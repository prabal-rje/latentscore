"""Evaluation suite for benchmarking vibe-to-config models."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError

if __package__ is None and __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_work.lib.eval_schema import (
    ErrorTag,
    EvalPrompt,
    EvalResult,
    EvalSetMetrics,
    compute_field_accuracy,
    compute_field_distributions,
)
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import (
    LocalHFClient,
    format_prompt_json,
    litellm_structured_completion,
    load_env_file,
)
from data_work.lib.music_prompt import build_music_prompt
from data_work.lib.music_schema import MusicConfigPromptPayload

LOGGER = logging.getLogger("data_work.eval_suite")

DEFAULT_SYSTEM_PROMPT = build_music_prompt()
DEFAULT_OUTPUT_DIR = Path("data_work/.eval_results")
DEFAULT_EVAL_SETS_DIR = Path("data_work/eval_sets")
DEFAULT_MAX_NEW_TOKENS = 512


class EvalSource(BaseModel):
    """Configuration for an evaluation source (model or baseline)."""

    model_config = ConfigDict(extra="forbid")

    kind: str  # "local", "litellm", "baseline"
    label: str
    model: str | None = None
    api_base: str | None = None


def load_eval_set(path: Path) -> list[EvalPrompt]:
    """Load an evaluation set from a JSONL file."""
    prompts = []
    for record in iter_jsonl(path):
        prompts.append(EvalPrompt(**record))
    return prompts


def validate_config_schema(config: dict[str, Any]) -> tuple[bool, str | None]:
    """Check if a config dict is valid against MusicConfigPromptPayload schema."""
    try:
        MusicConfigPromptPayload.model_validate(config)
        return True, None
    except ValidationError as exc:
        return False, str(exc)


def check_field_matches(
    config: dict[str, Any],
    expected_fields: dict[str, str],
) -> dict[str, bool]:
    """Check if config fields match expected values."""
    matches = {}
    for field, expected in expected_fields.items():
        actual = config.get(field)
        if actual is None:
            # Check nested in config.config if present
            inner = config.get("config", {})
            actual = inner.get(field)
        matches[field] = str(actual).lower() == str(expected).lower()
    return matches


async def run_litellm_inference(
    *,
    prompt: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    system_prompt: str,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Run inference with a LiteLLM model. Returns (config, error, time_ms)."""
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        result = await litellm_structured_completion(
            model=model,
            messages=messages,
            response_model=MusicConfigPromptPayload,
            api_key=api_key,
            api_base=api_base,
            model_kwargs=model_kwargs,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return result.model_dump(), None, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return None, str(exc), elapsed


def run_local_inference(
    *,
    prompt: str,
    client: LocalHFClient,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Run inference with a local HF model. Returns (config, error, time_ms)."""
    start = time.perf_counter()
    try:
        formatted = format_prompt_json(system_prompt, prompt)
        output = client.generate(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        # Parse JSON from output
        config = json.loads(output)
        elapsed = (time.perf_counter() - start) * 1000
        return config, None, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return None, str(exc), elapsed


def run_baseline_inference(
    *,
    prompt: str,
    baseline_type: str,
) -> tuple[dict[str, Any] | None, str | None, float]:
    """Run inference with a baseline. Returns (config dict, error, time_ms)."""
    start = time.perf_counter()
    try:
        # Import baselines lazily to avoid circular imports
        from data_work.lib.baselines import get_baseline

        baseline = get_baseline(baseline_type)
        payload = baseline.generate(prompt)
        # Convert Pydantic model to dict for consistency with other sources
        config = payload.model_dump()
        elapsed = (time.perf_counter() - start) * 1000
        return config, None, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return None, str(exc), elapsed


async def evaluate_source(
    *,
    source: EvalSource,
    prompts: list[EvalPrompt],
    api_key: str | None,
    model_kwargs: Mapping[str, Any],
    system_prompt: str,
    local_client: LocalHFClient | None,
    max_new_tokens: int,
    temperature: float,
    include_clap: bool,
    clap_scorer: Any | None,
    llm_scorer: Any | None,
    duration: float,
) -> list[EvalResult]:
    """Evaluate a single source on all prompts."""
    results = []

    for prompt in prompts:
        config: dict[str, Any] | None = None
        error: str | None = None
        time_ms: float = 0.0

        if source.kind == "litellm":
            config, error, time_ms = await run_litellm_inference(
                prompt=prompt.prompt,
                model=source.model or "",
                api_key=api_key,
                api_base=source.api_base,
                model_kwargs=model_kwargs,
                system_prompt=system_prompt,
            )
        elif source.kind == "local":
            if local_client is None:
                error = "No local client available"
            else:
                config, error, time_ms = run_local_inference(
                    prompt=prompt.prompt,
                    client=local_client,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
        elif source.kind == "baseline":
            config, error, time_ms = run_baseline_inference(
                prompt=prompt.prompt,
                baseline_type=source.model or "random",
            )

        # Compute validity metrics
        json_valid = config is not None and error is None
        schema_valid = False
        schema_error: str | None = None
        if json_valid and config is not None:
            schema_valid, schema_error = validate_config_schema(config)
            if not schema_valid:
                error = schema_error

        # Check field matches
        field_matches: dict[str, bool] = {}
        if config is not None and prompt.expected_fields:
            field_matches = check_field_matches(config, prompt.expected_fields)

        # Track errors
        error_tags: list[ErrorTag] = []
        clap_error: str | None = None
        llm_error: str | None = None

        # Tag config generation errors (vibe-to-config model output)
        if not json_valid:
            error_tags.append(ErrorTag.CONFIG_JSON_INVALID)
        elif not schema_valid:
            error_tags.append(ErrorTag.CONFIG_SCHEMA_INVALID)

        # CLAP scoring (optional, expensive)
        clap_score: float | None = None
        clap_audio_text_sim: float | None = None
        clap_badness_penalty: float | None = None

        if include_clap and clap_scorer is not None and config is not None and schema_valid:
            try:
                from data_work.lib.clap_scorer import score_config

                clap_result = score_config(
                    vibe=prompt.prompt,
                    config=config,
                    scorer=clap_scorer,
                    duration=duration,
                )
                clap_score = clap_result.final_reward
                clap_audio_text_sim = clap_result.audio_text_similarity
                clap_badness_penalty = clap_result.penalty
            except Exception as exc:
                clap_error = f"{type(exc).__name__}: {exc}"
                error_tags.append(ErrorTag.CLAP_FAILED)
                LOGGER.warning("CLAP scoring failed for %s: %s", prompt.id, exc)

        # LLM-based audio scoring (optional)
        llm_vibe_match: float | None = None
        llm_audio_quality: float | None = None
        llm_coherence: float | None = None
        llm_creativity: float | None = None
        llm_justification: str | None = None
        llm_score_value: float | None = None

        if llm_scorer is not None and config is not None and schema_valid:
            try:
                from data_work.lib.llm_scorer import score_config_with_llm_detailed_async

                llm_detailed = await score_config_with_llm_detailed_async(
                    vibe=prompt.prompt,
                    config=config,
                    scorer=llm_scorer,
                    duration=duration,
                )
                llm_vibe_match = llm_detailed.vibe_match
                llm_audio_quality = llm_detailed.audio_quality
                llm_coherence = llm_detailed.coherence
                llm_creativity = llm_detailed.creativity
                llm_justification = llm_detailed.justification

                # Compute weighted overall score
                llm_score_value = (
                    llm_vibe_match * 0.5
                    + llm_audio_quality * 0.2
                    + llm_coherence * 0.2
                    + llm_creativity * 0.1
                )
            except ValidationError as exc:
                llm_error = f"SchemaError: {exc}"
                error_tags.append(ErrorTag.LLM_SCHEMA_ERROR)
                LOGGER.warning("LLM returned invalid schema for %s: %s", prompt.id, exc)
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                error_tags.append(ErrorTag.LLM_FAILED)
                LOGGER.warning("LLM scoring failed for %s: %s", prompt.id, exc)

        result = EvalResult(
            prompt_id=prompt.id,
            source_label=source.label,
            config=config,
            config_error=error,
            json_valid=json_valid,
            schema_valid=schema_valid,
            field_matches=field_matches,
            clap_score=clap_score,
            clap_audio_text_sim=clap_audio_text_sim,
            clap_badness_penalty=clap_badness_penalty,
            clap_error=clap_error,
            llm_vibe_match=llm_vibe_match,
            llm_audio_quality=llm_audio_quality,
            llm_coherence=llm_coherence,
            llm_creativity=llm_creativity,
            llm_justification=llm_justification,
            llm_score=llm_score_value,
            llm_error=llm_error,
            error_tags=error_tags,
            inference_time_ms=time_ms,
        )
        results.append(result)

        LOGGER.info(
            "Evaluated %s with %s: json=%s schema=%s time=%.1fms",
            prompt.id,
            source.label,
            json_valid,
            schema_valid,
            time_ms,
        )

    return results


def compute_metrics(
    results: list[EvalResult],
    prompts: list[EvalPrompt],
    eval_set_name: str,
    source_label: str,
) -> EvalSetMetrics:
    """Compute aggregate metrics from evaluation results."""
    n_samples = len(results)
    if n_samples == 0:
        return EvalSetMetrics(
            eval_set_name=eval_set_name,
            source_label=source_label,
            n_samples=0,
        )

    # Validity rates
    json_valid_rate = sum(1 for r in results if r.json_valid) / n_samples
    schema_valid_rate = sum(1 for r in results if r.schema_valid) / n_samples

    # Field accuracy
    field_accuracy = compute_field_accuracy(results, prompts)

    # CLAP scores
    clap_scores = [r.clap_score for r in results if r.clap_score is not None]
    clap_score_mean = statistics.mean(clap_scores) if clap_scores else None
    clap_score_std = statistics.stdev(clap_scores) if len(clap_scores) > 1 else None
    clap_score_median = statistics.median(clap_scores) if clap_scores else None

    # LLM scores
    llm_scores = [r.llm_score for r in results if r.llm_score is not None]
    llm_score_mean = statistics.mean(llm_scores) if llm_scores else None
    llm_score_std = statistics.stdev(llm_scores) if len(llm_scores) > 1 else None

    # LLM sub-scores
    llm_vibe_matches = [r.llm_vibe_match for r in results if r.llm_vibe_match is not None]
    llm_vibe_match_mean = statistics.mean(llm_vibe_matches) if llm_vibe_matches else None

    llm_audio_qualities = [r.llm_audio_quality for r in results if r.llm_audio_quality is not None]
    llm_audio_quality_mean = statistics.mean(llm_audio_qualities) if llm_audio_qualities else None

    llm_coherences = [r.llm_coherence for r in results if r.llm_coherence is not None]
    llm_coherence_mean = statistics.mean(llm_coherences) if llm_coherences else None

    llm_creativities = [r.llm_creativity for r in results if r.llm_creativity is not None]
    llm_creativity_mean = statistics.mean(llm_creativities) if llm_creativities else None

    # Field distributions
    field_distributions = compute_field_distributions(results)

    # Latency
    inference_times = [r.inference_time_ms for r in results if r.inference_time_ms is not None]
    inference_time_mean = statistics.mean(inference_times) if inference_times else None
    inference_time_p95 = (
        sorted(inference_times)[int(len(inference_times) * 0.95)]
        if len(inference_times) > 1
        else inference_times[0]
        if inference_times
        else None
    )

    # Attribute accuracy (for controllability)
    attribute_accuracy: dict[str, float] = {}
    for field in field_accuracy:
        attribute_accuracy[field] = field_accuracy[field]

    # Error counts
    n_config_errors = sum(1 for r in results if not r.schema_valid)
    n_clap_errors = sum(1 for r in results if r.clap_error is not None)
    n_llm_errors = sum(1 for r in results if r.llm_error is not None)

    # Error tag counts
    error_tag_counts: dict[str, int] = {}
    for r in results:
        for tag in r.error_tags:
            error_tag_counts[tag] = error_tag_counts.get(tag, 0) + 1

    return EvalSetMetrics(
        eval_set_name=eval_set_name,
        source_label=source_label,
        n_samples=n_samples,
        json_valid_rate=json_valid_rate,
        schema_valid_rate=schema_valid_rate,
        field_accuracy=field_accuracy,
        clap_score_mean=clap_score_mean,
        clap_score_std=clap_score_std,
        clap_score_median=clap_score_median,
        llm_score_mean=llm_score_mean,
        llm_score_std=llm_score_std,
        llm_vibe_match_mean=llm_vibe_match_mean,
        llm_audio_quality_mean=llm_audio_quality_mean,
        llm_coherence_mean=llm_coherence_mean,
        llm_creativity_mean=llm_creativity_mean,
        n_config_errors=n_config_errors,
        n_clap_errors=n_clap_errors,
        n_llm_errors=n_llm_errors,
        error_tag_counts=error_tag_counts,
        field_distributions=field_distributions,
        inference_time_mean_ms=inference_time_mean,
        inference_time_p95_ms=inference_time_p95,
        attribute_accuracy=attribute_accuracy,
    )


def write_results(
    output_dir: Path,
    results: list[EvalResult],
    metrics: EvalSetMetrics,
    prompts: list[EvalPrompt],
) -> None:
    """Write evaluation results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-sample results
    results_path = output_dir / "per_sample.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    # Write metrics
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(metrics.model_dump_json(indent=2))

    # Write summary
    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Evaluation Summary: {metrics.eval_set_name}\n\n")
        f.write(f"**Source:** {metrics.source_label}\n")
        f.write(f"**Samples:** {metrics.n_samples}\n\n")

        f.write("## Validity\n\n")
        f.write(f"- JSON Valid: {metrics.json_valid_rate:.1%}\n")
        f.write(f"- Schema Valid: {metrics.schema_valid_rate:.1%}\n\n")

        if metrics.field_accuracy:
            f.write("## Field Accuracy\n\n")
            for field, accuracy in metrics.field_accuracy.items():
                f.write(f"- {field}: {accuracy:.1%}\n")
            f.write("\n")

        if metrics.clap_score_mean is not None:
            f.write("## CLAP Scores\n\n")
            f.write(f"- Mean: {metrics.clap_score_mean:.3f}\n")
            if metrics.clap_score_std is not None:
                f.write(f"- Std: {metrics.clap_score_std:.3f}\n")
            if metrics.clap_score_median is not None:
                f.write(f"- Median: {metrics.clap_score_median:.3f}\n")
            f.write("\n")

        if metrics.llm_score_mean is not None:
            f.write("## LLM Scores\n\n")
            f.write(f"- Overall Mean: {metrics.llm_score_mean:.3f}\n")
            if metrics.llm_score_std is not None:
                f.write(f"- Overall Std: {metrics.llm_score_std:.3f}\n")
            if metrics.llm_vibe_match_mean is not None:
                f.write(f"- Vibe Match Mean: {metrics.llm_vibe_match_mean:.3f}\n")
            if metrics.llm_audio_quality_mean is not None:
                f.write(f"- Audio Quality Mean: {metrics.llm_audio_quality_mean:.3f}\n")
            if metrics.llm_coherence_mean is not None:
                f.write(f"- Coherence Mean: {metrics.llm_coherence_mean:.3f}\n")
            if metrics.llm_creativity_mean is not None:
                f.write(f"- Creativity Mean: {metrics.llm_creativity_mean:.3f}\n")
            f.write("\n")

        if metrics.inference_time_mean_ms is not None:
            f.write("## Latency\n\n")
            f.write(f"- Mean: {metrics.inference_time_mean_ms:.1f}ms\n")
            if metrics.inference_time_p95_ms is not None:
                f.write(f"- P95: {metrics.inference_time_p95_ms:.1f}ms\n")
            f.write("\n")

        # Error summary
        has_errors = (
            metrics.n_config_errors > 0 or metrics.n_clap_errors > 0 or metrics.n_llm_errors > 0
        )
        if has_errors:
            f.write("## Errors\n\n")
            if metrics.n_config_errors > 0:
                f.write(f"- Config Errors: {metrics.n_config_errors}\n")
            if metrics.n_clap_errors > 0:
                f.write(f"- CLAP Errors: {metrics.n_clap_errors}\n")
            if metrics.n_llm_errors > 0:
                f.write(f"- LLM Errors: {metrics.n_llm_errors}\n")
            if metrics.error_tag_counts:
                f.write("\n**Error Tags:**\n")
                for tag, count in sorted(metrics.error_tag_counts.items()):
                    f.write(f"- `{tag}`: {count}\n")
            f.write("\n")

    LOGGER.info("Results written to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for eval suite."""
    parser = argparse.ArgumentParser(
        description="Evaluation suite for vibe-to-config models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--eval-set",
        type=str,
        required=True,
        help="Eval set name (e.g., short_prompts, controllability/tempo_prompts)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write evaluation results.",
    )

    # Model sources
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Local HF model path. Format: path[:label]",
    )
    parser.add_argument(
        "--litellm-model",
        type=str,
        default=None,
        help="LiteLLM model. Format: model[:label]",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline type (random, rule_based). Format: type[:label]. Can be repeated.",
    )

    # Audio scoring
    parser.add_argument(
        "--include-clap",
        action="store_true",
        help="Include CLAP scoring (requires laion_clap).",
    )
    parser.add_argument(
        "--llm-scorer",
        type=str,
        default=None,
        help=(
            "LLM model for audio scoring. Supports Gemini 3 (gemini/gemini-3-flash-preview, "
            "gemini/gemini-3-pro-preview) or Voxtral (mistral/voxtral-small-latest)."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Audio duration for scoring.",
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LiteLLM providers.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Env var name for API key.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON dict of extra LiteLLM kwargs.",
    )

    # Local model options
    parser.add_argument(
        "--local-device",
        type=str,
        default=None,
        help="Device for local models (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--local-max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max tokens for local model generation.",
    )
    parser.add_argument(
        "--local-temperature",
        type=float,
        default=0.0,
        help="Temperature for local models (0 = greedy).",
    )

    # Other
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of prompts to evaluate (0 = all).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for config generation.",
    )
    parser.add_argument(
        "--eval-sets-dir",
        type=Path,
        default=DEFAULT_EVAL_SETS_DIR,
        help="Directory containing eval sets.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def parse_source_spec(spec: str, kind: str) -> EvalSource:
    """Parse a source specification like 'path:label' into an EvalSource."""
    name, _, label = spec.partition(":")
    clean_label = label or name
    return EvalSource(kind=kind, label=clean_label, model=name)


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    # Load environment
    if args.env_file:
        load_env_file(args.env_file)

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get(args.api_key_env)

    model_kwargs = json.loads(args.model_kwargs)

    # Load eval set
    eval_set_path = args.eval_sets_dir / f"{args.eval_set}.jsonl"
    if not eval_set_path.exists():
        raise SystemExit(f"Eval set not found: {eval_set_path}")

    prompts = load_eval_set(eval_set_path)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    LOGGER.info("Loaded %d prompts from %s", len(prompts), eval_set_path)

    # Build source list
    sources: list[EvalSource] = []
    if args.local_model:
        sources.append(parse_source_spec(args.local_model, "local"))
    if args.litellm_model:
        sources.append(parse_source_spec(args.litellm_model, "litellm"))
    for baseline in args.baseline:
        sources.append(parse_source_spec(baseline, "baseline"))

    if not sources:
        raise SystemExit(
            "No evaluation source specified. Use --local-model, --litellm-model, or --baseline."
        )

    # Initialize local client if needed
    local_client: LocalHFClient | None = None
    for source in sources:
        if source.kind == "local":
            local_client = LocalHFClient(
                model_path=source.model or "",
                device=args.local_device,
                max_new_tokens=args.local_max_new_tokens,
                temperature=args.local_temperature,
            )
            break

    # Initialize CLAP scorer if needed
    clap_scorer = None
    if args.include_clap:
        try:
            import importlib

            clap_module = importlib.import_module("data_work.04_clap_benchmark")
            ClapScorer = clap_module.ClapScorer  # noqa: N806
            clap_scorer = ClapScorer()
            LOGGER.info("CLAP scorer initialized")
        except ImportError:
            LOGGER.warning("CLAP scorer not available. Install laion_clap.")

    # Initialize LLM scorer if specified
    llm_scorer = None
    if args.llm_scorer:
        from data_work.lib.llm_scorer import LLMScorer

        llm_scorer = LLMScorer(
            model=args.llm_scorer,
            api_key=api_key,
            api_base=None,
        )
        LOGGER.info("LLM scorer initialized with model: %s", args.llm_scorer)

    # Run evaluation for each source
    for source in sources:
        LOGGER.info("Evaluating source: %s", source.label)

        results = await evaluate_source(
            source=source,
            prompts=prompts,
            api_key=api_key,
            model_kwargs=model_kwargs,
            system_prompt=args.system_prompt,
            local_client=local_client if source.kind == "local" else None,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
            include_clap=args.include_clap,
            clap_scorer=clap_scorer,
            llm_scorer=llm_scorer,
            duration=args.duration,
        )

        # Compute metrics
        eval_set_name = args.eval_set.replace("/", "_")
        metrics = compute_metrics(results, prompts, eval_set_name, source.label)

        # Write results
        source_output_dir = args.output_dir / eval_set_name / source.label
        write_results(source_output_dir, results, metrics, prompts)

        # Print summary
        print(f"\n=== {source.label} on {eval_set_name} ===")
        print(f"Samples: {metrics.n_samples}")
        print(f"JSON Valid: {metrics.json_valid_rate:.1%}")
        print(f"Schema Valid: {metrics.schema_valid_rate:.1%}")
        if metrics.field_accuracy:
            print("Field Accuracy:")
            for field, acc in metrics.field_accuracy.items():
                print(f"  {field}: {acc:.1%}")
        if metrics.clap_score_mean is not None:
            print(f"CLAP Mean: {metrics.clap_score_mean:.3f}")
        if metrics.llm_score_mean is not None:
            print(f"LLM Score Mean: {metrics.llm_score_mean:.3f}")
            if metrics.llm_vibe_match_mean is not None:
                print(f"  Vibe Match: {metrics.llm_vibe_match_mean:.3f}")
            if metrics.llm_audio_quality_mean is not None:
                print(f"  Audio Quality: {metrics.llm_audio_quality_mean:.3f}")
            if metrics.llm_coherence_mean is not None:
                print(f"  Coherence: {metrics.llm_coherence_mean:.3f}")
            if metrics.llm_creativity_mean is not None:
                print(f"  Creativity: {metrics.llm_creativity_mean:.3f}")
        if metrics.inference_time_mean_ms is not None:
            print(f"Inference Time Mean: {metrics.inference_time_mean_ms:.1f}ms")

        # Print errors
        has_errors = (
            metrics.n_config_errors > 0 or metrics.n_clap_errors > 0 or metrics.n_llm_errors > 0
        )
        if has_errors:
            print("Errors:")
            if metrics.n_config_errors > 0:
                print(f"  Config: {metrics.n_config_errors}")
            if metrics.n_clap_errors > 0:
                print(f"  CLAP: {metrics.n_clap_errors}")
            if metrics.n_llm_errors > 0:
                print(f"  LLM: {metrics.n_llm_errors}")
            if metrics.error_tag_counts:
                for tag, count in sorted(metrics.error_tag_counts.items()):
                    print(f"    [{tag}]: {count}")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
