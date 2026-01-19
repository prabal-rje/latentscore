"""Generate music configs from vibes using Best-of-N sampling.

This script:
1. Reads vibe JSONL from 02a_extract_vibes output
2. Generates N config candidates per vibe using a SOTA model (Claude Opus)
3. Scores each candidate (format_valid, schema_valid, palette_valid)
4. Stores all N candidates + scores
5. Selects the best valid config
6. Writes incrementally to prevent data loss on crash

Output format:
{
    "dataset": "...",
    "id_in_dataset": "...",
    "split": "SFT-Train",
    ...all vibe fields from input...,
    "config_model": "anthropic/claude-opus-4-5-20251101",
    "config_candidates": [
        {"thinking": "...", "config": {...}, "palettes": [...]},
        ...
    ],
    "scores": {
        "format_valid": [1, 1, 1, 0, 1],
        "schema_valid": [1, 1, 0, 0, 1],
        "palette_valid": [1, 1, 0, 0, 1]
    },
    "best_index": 0,
    "config_payload": {"thinking": "...", "config": {...}, "palettes": [...]},
    "config_error": null
}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import JsonValue, ValidationError

from common.prompts import build_config_generation_prompt
from data_work.lib.config_batcher import (
    ConfigBatcher,
    build_batch_prompt,
    build_batch_response_keys,
    build_batch_response_model,
    parse_batch_response,
)
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_cache import SqliteCache, cached_async
from data_work.lib.llm_client import (
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
    wrap_vibe_for_chat,
)
from data_work.lib.music_schema import MusicConfigPromptPayload
from data_work.lib.periodic_writer import AsyncPeriodicWriter
from data_work.lib.pipeline_models import (
    ConfigCandidateScores,
    ConfigGenerationRow,
    JsonDict,
    VibeRow,
)
from data_work.lib.resilience import (
    AsyncRateLimiter,
    EWMAErrorTracker,
    rate_limited,
    retry_async,
    with_semaphore,
)

_LOGGER = logging.getLogger("data_work.generate_configs")

DEFAULT_MODEL = "anthropic/claude-opus-4-5-20251101"
DEFAULT_CACHE_PATH = Path("data_work/.cache/config_cache.sqlite")
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_QPS = 2.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_TEMPERATURE = 0.8
DEFAULT_BATCH_SIZE = 1
DEFAULT_BATCH_WAIT_MS = 200
DEFAULT_EWMA_ALPHA = 0.2
DEFAULT_EWMA_THRESHOLD = 0.25
DEFAULT_EWMA_MIN_SAMPLES = 10
DEFAULT_PERIODIC_WRITE_INTERVAL = 60.0
RUN_CONFIG_NAME = "run_config.json"

ModelKwargs = Mapping[str, JsonValue]
_DEBUG_PROMPTS = os.getenv("LATENTSCORE_DEBUG_PROMPTS") == "1"
_DEBUG_PROMPTS_EMITTED = False


def _maybe_debug_prompts(
    messages: Sequence[Mapping[str, str]],
    cache_control: list[dict[str, str]] | None,
    label: str,
) -> None:
    global _DEBUG_PROMPTS_EMITTED
    if not _DEBUG_PROMPTS or _DEBUG_PROMPTS_EMITTED:
        return
    _DEBUG_PROMPTS_EMITTED = True
    print(f"[DEBUG] {label} prompt preview")
    print(f"[DEBUG] cache_control: {cache_control}")
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        print(f"[DEBUG] role={role}\n{content}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate music configs from vibes using Best-of-N sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing vibe JSONL files from 02a_extract_vibes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where config JSONL files will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LiteLLM model string for config generation.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="ANTHROPIC_API_KEY",
        help="Environment variable for the API key.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help="Number of config candidates to generate per vibe (Best-of-N).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for diversity in generations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for vibe-to-config generation (1 disables batching).",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=DEFAULT_BATCH_WAIT_MS,
        help="Max wait (ms) to fill a batch before sending.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit total rows to process (0 = no limit).",
    )
    parser.add_argument(
        "--only-splits",
        nargs="*",
        default=None,
        help="Only process specific splits (e.g., SFT-Train TEST).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum concurrent LLM requests.",
    )
    parser.add_argument(
        "--max-qps",
        type=float,
        default=DEFAULT_MAX_QPS,
        help="Rate limit in requests per second.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry attempts after first failure.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help="Base delay for exponential backoff.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="SQLite cache path for LLM responses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip already processed rows).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=DEFAULT_EWMA_ALPHA,
        help="EWMA smoothing factor for error tracking.",
    )
    parser.add_argument(
        "--ewma-threshold",
        type=float,
        default=DEFAULT_EWMA_THRESHOLD,
        help="Abort if error rate exceeds this.",
    )
    parser.add_argument(
        "--ewma-min-samples",
        type=int,
        default=DEFAULT_EWMA_MIN_SAMPLES,
        help="Minimum samples before enforcing abort threshold.",
    )
    parser.add_argument(
        "--no-prompt-caching",
        action="store_true",
        help="Disable prompt caching (not recommended - increases input token costs).",
    )
    parser.add_argument(
        "--write-interval",
        type=float,
        default=DEFAULT_PERIODIC_WRITE_INTERVAL,
        help="Seconds between periodic output writes (default 60s).",
    )
    return parser


def hash_text(text: str) -> str:
    """Hash text for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    vibe_text: str,
    model: str,
    api_base: str | None,
    temperature: float,
    candidate_index: int,
    seed: int,
    prompt_hash: str,
) -> tuple[str, JsonDict]:
    """Build cache key for a single config generation call."""
    payload: JsonDict = {
        "vibe_text_hash": hash_text(vibe_text),
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "candidate_index": candidate_index,
        "seed": seed,
        "prompt_hash": prompt_hash,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(payload_json).hexdigest()
    return cache_key, payload


def row_key(row: VibeRow | ConfigGenerationRow) -> str:
    """Generate unique key for a row."""
    return f"{row.dataset}:{row.id_in_dataset}:{row.vibe_index}:{row.vibe_scope}:{row.vibe_level}"


def load_processed_keys(path: Path) -> set[str]:
    """Load keys of already-processed rows for resume."""
    if not path.exists():
        return set()
    processed: set[str] = set()
    for row in iter_jsonl(path):
        processed.add(row_key(ConfigGenerationRow.model_validate(row)))
    return processed


async def call_llm_for_config(
    *,
    vibe_text: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
    temperature: float,
    use_prompt_caching: bool = True,
) -> JsonDict:
    """Call LLM for config generation.

    Args:
        vibe_text: The vibe description to convert to a config
        model: LiteLLM model string
        api_key: API key for the model provider
        api_base: API base URL override
        system_prompt: System prompt with instructions and schema
        temperature: Sampling temperature
        use_prompt_caching: If True, enable prompt caching on system prompt.
            Supported providers: OpenAI, Anthropic, Bedrock, Deepseek.
            This reduces input token costs by ~90% for calls 2-5 in Best-of-N sampling
            since the system prompt is identical across all candidates.
            Other providers will ignore this parameter.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": wrap_vibe_for_chat(vibe_text)},
    ]

    # Enable prompt caching on the system message.
    # This caches the system prompt (instructions + schema) across all N candidates,
    # reducing input token costs by ~90% for calls 2-5.
    # Supported providers: OpenAI, Anthropic, Bedrock, Deepseek
    # Other providers will ignore this parameter.
    # See: https://docs.litellm.ai/docs/completion/prompt_caching
    cache_control: list[dict[str, str]] | None = None
    if use_prompt_caching:
        cache_control = [{"location": "message", "role": "system"}]

    _maybe_debug_prompts(messages, cache_control, "single-config")

    # Use structured completion but catch validation errors
    result = await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=MusicConfigPromptPayload,
        api_key=api_key,
        api_base=api_base,
        model_kwargs={"temperature": temperature},
        cache_control_injection_points=cache_control,
    )

    return result.model_dump(mode="json")


async def call_music_config_batch(
    *,
    vibe_texts: Sequence[str],
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
    model_kwargs: ModelKwargs,
    use_prompt_caching: bool = True,
) -> list[JsonDict]:
    keys = build_batch_response_keys(len(vibe_texts))
    prompt = build_batch_prompt(vibe_texts, keys)
    response_model = build_batch_response_model(len(vibe_texts))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    cache_control: list[dict[str, str]] | None = None
    if use_prompt_caching:
        cache_control = [{"location": "message", "role": "system"}]
    _maybe_debug_prompts(messages, cache_control, "batch-config")
    response = await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=response_model,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
        cache_control_injection_points=cache_control,
    )
    payloads = parse_batch_response(response, len(vibe_texts))
    return [payload.model_dump(mode="json") for payload in payloads]


def score_candidate(candidate: JsonDict | None) -> dict[str, int]:
    """Score a single config candidate.

    Returns dict with binary scores:
    - format_valid: 1 if it's a dict, 0 otherwise
    - schema_valid: 1 if validates against MusicConfigPromptPayload, 0 otherwise
    - palette_valid: 1 if palettes have correct structure (3 palettes, 5 colors each)
    """
    scores = {
        "format_valid": 0,
        "schema_valid": 0,
        "palette_valid": 0,
    }

    # Format check
    if not isinstance(candidate, dict):
        return scores
    scores["format_valid"] = 1

    # Schema validation
    try:
        MusicConfigPromptPayload.model_validate(candidate)
        scores["schema_valid"] = 1
    except (ValidationError, Exception):
        pass

    # Palette structure check
    palettes = candidate.get("palettes", [])
    if isinstance(palettes, list) and len(palettes) == 3:
        palette_ok = True
        for palette in palettes:
            colors = palette.get("colors", []) if isinstance(palette, dict) else []
            if not isinstance(colors, list) or len(colors) != 5:
                palette_ok = False
                break
        if palette_ok:
            scores["palette_valid"] = 1

    return scores


def select_best_candidate(
    candidates: list[JsonDict | None],
    scores: ConfigCandidateScores,
) -> tuple[int, JsonDict | None]:
    """Select the best candidate based on scores.

    Priority: schema_valid > palette_valid > format_valid.
    Among tied candidates, take first (deterministic).
    """
    if not candidates:
        return -1, None

    best_index = -1
    best_score = (-1, -1, -1)

    for i, candidate in enumerate(candidates):
        if candidate is None:
            continue

        score = (
            scores.schema_valid[i],
            scores.palette_valid[i],
            scores.format_valid[i],
        )

        if score > best_score:
            best_score = score
            best_index = i

    if best_index >= 0:
        return best_index, candidates[best_index]
    return -1, None


async def generate_configs_for_row(
    row: VibeRow,
    *,
    model: str,
    api_base: str | None,
    prompt_hash: str,
    num_candidates: int,
    temperature: float,
    seed: int,
    cache: SqliteCache,
    config_call: Callable[[str], Awaitable[JsonDict]],
    max_retries: int,
    retry_base_delay: float,
) -> ConfigGenerationRow:
    """Generate N config candidates for a single vibe row."""
    vibe_text = row.vibe_original

    candidates: list[JsonDict | None] = []
    errors: list[str | None] = []

    async def generate_one(
        candidate_index: int,
    ) -> tuple[JsonDict | None, str | None]:
        """Generate a single config candidate."""
        cache_key, payload = build_cache_key(
            vibe_text=vibe_text,
            model=model,
            api_base=api_base,
            temperature=temperature,
            candidate_index=candidate_index,
            seed=seed,
            prompt_hash=prompt_hash,
        )

        @cached_async(
            cache=cache,
            key_fn=lambda: (cache_key, payload),
            serializer=lambda response: response,
            deserializer=lambda data: data,
        )
        @retry_async(max_retries=max_retries, base_delay=retry_base_delay, logger=_LOGGER)
        async def _call() -> JsonDict:
            return await config_call(vibe_text)

        try:
            result = await _call()
            return result, None
        except Exception as exc:
            _LOGGER.warning(
                "Config generation failed for candidate %d: %s",
                candidate_index,
                exc,
            )
            return None, str(exc)

    # Generate all candidates concurrently
    tasks = [generate_one(i) for i in range(num_candidates)]
    results = await asyncio.gather(*tasks)

    for result, error in results:
        candidates.append(result)
        errors.append(error)

    candidate_payloads = candidates

    all_scores = ConfigCandidateScores(
        format_valid=[],
        schema_valid=[],
        palette_valid=[],
    )

    for candidate in candidate_payloads:
        candidate_scores = score_candidate(candidate)
        all_scores.format_valid.append(candidate_scores["format_valid"])
        all_scores.schema_valid.append(candidate_scores["schema_valid"])
        all_scores.palette_valid.append(candidate_scores["palette_valid"])

    # Select best
    best_index, best_candidate = select_best_candidate(candidate_payloads, all_scores)

    # Build output row
    return ConfigGenerationRow(
        **row.model_dump(),
        config_model=model,
        config_candidates=candidate_payloads,
        scores=all_scores,
        best_index=best_index,
        config_payload=best_candidate,
        config_error="; ".join(e for e in errors if e) if best_candidate is None else None,
    )


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Write run configuration to output directory."""
    config = {
        "seed": args.seed,
        "model": args.model,
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "batch_wait_ms": args.batch_wait_ms,
    }
    config_path = output_dir / RUN_CONFIG_NAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    from tqdm import tqdm  # type: ignore[import]

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Find input files
    split_files: list[tuple[str, Path]] = []
    for split_file in sorted(input_dir.glob("*.jsonl")):
        split_name = split_file.stem
        if args.only_splits and split_name not in args.only_splits:
            continue
        split_files.append((split_name, split_file))

    if not split_files:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    # Check output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, _ in split_files:
        output_path = output_dir / f"{split_name}.jsonl"
        if output_path.exists() and not args.overwrite and not args.resume:
            raise SystemExit(f"Output exists: {output_path}. Use --overwrite or --resume.")

    # Load environment and setup API
    load_env_file(args.env_file)
    model, api_base = normalize_model_and_base(args.model, None)

    # Use the API key env provided by the user (no auto-detection - too fragile)
    api_key = resolve_api_key_for_models(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        models=[(model, api_base)],
    )

    # Write run config
    write_run_config(output_dir, args)

    # Setup LLM infrastructure
    base_prompt = build_config_generation_prompt(batch=False)
    batch_prompt = (
        build_config_generation_prompt(batch=True) if args.batch_size > 1 else base_prompt
    )
    prompt_hash = hash_text(batch_prompt if args.batch_size > 1 else base_prompt)
    cache = SqliteCache(args.cache_path)
    await cache.initialize()

    config_semaphore = asyncio.Semaphore(args.max_concurrency)
    config_limiter = AsyncRateLimiter(args.max_qps)
    tracker = EWMAErrorTracker(alpha=args.ewma_alpha, min_samples=args.ewma_min_samples)

    use_prompt_caching = not args.no_prompt_caching

    _LOGGER.info("Model: %s", model)
    _LOGGER.info("Candidates per vibe: %d", args.num_candidates)
    _LOGGER.info("Temperature: %.2f", args.temperature)
    if use_prompt_caching:
        _LOGGER.info(
            "Prompt caching: ENABLED (system prompt cached across %d candidates; "
            "supported by OpenAI, Anthropic, Bedrock, Deepseek)",
            args.num_candidates,
        )
    else:
        _LOGGER.info("Prompt caching: DISABLED (full input token cost per candidate)")

    model_kwargs: ModelKwargs = {"temperature": args.temperature}

    @rate_limited(config_limiter)
    @with_semaphore(config_semaphore)
    async def single_config_call(vibe_text: str) -> JsonDict:
        return await call_llm_for_config(
            vibe_text=vibe_text,
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=base_prompt,
            temperature=args.temperature,
            use_prompt_caching=use_prompt_caching,
        )

    @rate_limited(config_limiter)
    @with_semaphore(config_semaphore)
    async def batched_config_call(vibe_texts: Sequence[str]) -> list[JsonDict]:
        return await call_music_config_batch(
            vibe_texts=vibe_texts,
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=batch_prompt,
            model_kwargs=model_kwargs,
            use_prompt_caching=use_prompt_caching,
        )

    batcher: ConfigBatcher | None = None
    if args.batch_size > 1:
        batcher = ConfigBatcher(
            batch_size=args.batch_size,
            max_wait=args.batch_wait_ms / 1000.0,
            call_batch=batched_config_call,
        )
        _LOGGER.info(
            "Batching ENABLED (batch_size=%d, max_wait_ms=%d)",
            args.batch_size,
            args.batch_wait_ms,
        )
    else:
        _LOGGER.info("Batching DISABLED (batch_size=1)")

    async def config_call(vibe_text: str) -> JsonDict:
        if batcher is None:
            return await single_config_call(vibe_text)
        try:
            return await batcher.submit(vibe_text)
        except Exception as exc:
            _LOGGER.warning(
                "Batch config call failed; falling back to single-item batch call. (%s)",
                exc,
            )
            return (await batched_config_call([vibe_text]))[0]

    # Process each split
    total_processed = 0
    try:
        for split_name, split_file in split_files:
            output_path = output_dir / f"{split_name}.jsonl"

            # Load processed keys for resume
            processed_keys: set[str] = set()
            if args.resume and output_path.exists():
                processed_keys = load_processed_keys(output_path)
                _LOGGER.info(
                    "Resuming %s: %d already processed",
                    split_name,
                    len(processed_keys),
                )

            # Overwrite mode: clear existing file
            if args.overwrite and output_path.exists():
                output_path.unlink()

            # Load input rows
            input_rows: list[VibeRow] = []
            for row in iter_jsonl(split_file):
                try:
                    input_rows.append(VibeRow.model_validate(row))
                except ValidationError as exc:
                    raise SystemExit(f"Invalid vibe row in {split_file}: {exc}") from exc
            if args.limit > 0:
                remaining = args.limit - total_processed
                if remaining <= 0:
                    break
                input_rows = input_rows[:remaining]

            # Filter out already processed
            pending_rows = [row for row in input_rows if row_key(row) not in processed_keys]

            if not pending_rows:
                _LOGGER.info("Split %s: all rows already processed", split_name)
                continue

            _LOGGER.info("Processing %s: %d pending rows", split_name, len(pending_rows))

            # Setup periodic writer for this split
            writer = AsyncPeriodicWriter(
                output_path,
                interval_seconds=args.write_interval,
                overwrite=False,  # Never overwrite in append/resume mode
            )
            await writer.start()
            _LOGGER.info(
                "Output file: %s (updates every %.0fs)",
                output_path,
                args.write_interval,
            )

            try:
                with tqdm(total=len(pending_rows), desc=split_name) as progress:
                    for row in pending_rows:
                        try:
                            result = await generate_configs_for_row(
                                row,
                                model=model,
                                api_base=api_base,
                                prompt_hash=prompt_hash,
                                num_candidates=args.num_candidates,
                                temperature=args.temperature,
                                seed=args.seed,
                                cache=cache,
                                config_call=config_call,
                                max_retries=args.max_retries,
                                retry_base_delay=args.retry_base_delay,
                            )

                            # Track success based on whether we got a valid config
                            success = result.config_payload is not None
                            tracker.update(success=success)

                            # Add to periodic writer buffer
                            await writer.add_row(result)

                            progress.update(1)
                            total_processed += 1

                            # Check error threshold
                            if tracker.threshold_reached(args.ewma_threshold):
                                _LOGGER.error(
                                    "Excessive error rate (EWMA=%.2f). Aborting.",
                                    tracker.error_rate,
                                )
                                raise SystemExit("Excessive error rate - try again later.")

                        except SystemExit:
                            raise
                        except Exception as exc:
                            _LOGGER.error("Fatal error on row: %s", exc, exc_info=True)
                            raise
            finally:
                await writer.stop()
                _LOGGER.info(
                    "Split %s: wrote %d rows total",
                    split_name,
                    writer.total_written,
                )

            _LOGGER.info("Completed %s: wrote to %s", split_name, output_path)
    finally:
        if batcher is not None:
            await batcher.aclose()

    _LOGGER.info("Done! Total processed: %d rows", total_processed)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    _LOGGER.info(
        "Config generation with Best-of-%d using %s",
        args.num_candidates,
        args.model,
    )

    try:
        import uvloop  # type: ignore[import]

        uvloop.install()
    except ImportError:
        _LOGGER.info("uvloop not installed; using default asyncio event loop.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
