"""Extract vibes from base data into train/test/eval splits.

This script:
1. Reads raw texts from input-dir (output of 01_download_base_data)
2. Extracts vibes via LLM (cheap model like gpt-oss-20b)
3. Builds vibe rows (one row per vibe level/scope/character)
4. Dedupes on vibe_original content (not raw source text)
5. Splits into SFT-Train/Val, GRPO (diversity sampled), TEST
6. Writes incrementally to prevent data loss on crash

Output format (one row per vibe):
{
    "dataset": "...",
    "id_in_dataset": "...",
    "split": "SFT-Train",
    "vibe_index": 0,
    "text_page": 0,
    "vibe_scope": "scene",
    "character_name": null,
    "vibe_level": "xl",
    "vibe_original": "The city glows...",
    "vibe_noisy": "The city glows...",
    "tags_original": ["neon"],
    "tags_noisy": ["neon"],
    "vibe_model": "openai/gpt-oss-20b"
}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data_work.lib.dedupe import dedupe_vibe_rows, diversity_sample_rows
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_cache import SqliteCache, cached_async
from data_work.lib.llm_client import (
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
)
from data_work.lib.periodic_writer import AsyncPeriodicWriter
from data_work.lib.pipeline_models import BaseRecord, JsonDict, VibeRow
from data_work.lib.record_builder import build_vibe_rows
from data_work.lib.resilience import (
    AsyncRateLimiter,
    EWMAErrorTracker,
    rate_limited,
    retry_async,
    with_semaphore,
)
from data_work.lib.vibe_schema import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_PAGE_TOKENS,
    SPLITS,
    VibeResponse,
    build_vibe_prompt,
    normalize_vibe_indices,
    paginate_text,
    schema_hash,
)

_LOGGER = logging.getLogger("data_work.extract_vibes")

REQUIRED_FIELDS = ("created", "metadata", "dataset", "id_in_dataset", "text")

DEFAULT_MODEL = "openrouter/openai/gpt-oss-20b"
DEFAULT_OPENROUTER_CONTEXT_LIMIT = 131_072
DEFAULT_CACHE_PATH = Path("data_work/.cache/vibe_cache.sqlite")
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_QPS = 2.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 0.5
DEFAULT_ERROR_RATE = 0.15
DEFAULT_EWMA_ALPHA = 0.2
DEFAULT_EWMA_THRESHOLD = 0.25
DEFAULT_EWMA_MIN_SAMPLES = 10
DEFAULT_CHAR_AUG_P = 0.15
DEFAULT_FORCE_CHAR_AUG_P = 1.0
DEFAULT_DEDUPE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEDUPE_THRESHOLD = 0.95
DEFAULT_PERIODIC_WRITE_INTERVAL = 60.0
RUN_CONFIG_NAME = "run_config.json"
PROGRESS_FILE_NAME = "_progress.jsonl"


class RecordResult(BaseModel):
    """Result of processing a single input record."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    id_in_dataset: str | int
    rows: list[VibeRow] = Field(default_factory=list)
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract vibes from base JSONL files into splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing base JSONL files from 01_download_base_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where vibe JSONL splits will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LiteLLM model string for vibe extraction.",
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
        default="OPENROUTER_API_KEY",
        help="Environment variable for the API key.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum tokens to feed into the LLM before truncation.",
    )
    parser.add_argument(
        "--page-tokens",
        type=int,
        default=DEFAULT_PAGE_TOKENS,
        help="Approximate tokens per page marker.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Limit records per split (0 = no limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit total input records to process (0 = no limit). Use for quick tests.",
    )
    parser.add_argument(
        "--max-vibes",
        type=int,
        default=0,
        help="Stop when this many output vibes are extracted (0 = no limit). "
        "Useful for targeting a specific dataset size regardless of input text count.",
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
        "--error-rate",
        type=float,
        default=DEFAULT_ERROR_RATE,
        help="Probability of noise corruption per vibe.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="SQLite cache path for LLM responses.",
    )
    parser.add_argument(
        "--dedupe-model",
        type=str,
        default=DEFAULT_DEDUPE_MODEL,
        help="Sentence-transformers model for dedupe.",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=DEFAULT_DEDUPE_THRESHOLD,
        help="Cosine similarity threshold for dedupe (on vibe_original).",
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
        help="Seconds between periodic progress writes (default 60s).",
    )
    return parser


def hash_text(text: str) -> str:
    """Hash text for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_paths(paths: Sequence[Path]) -> str:
    """Hash paths for dataset fingerprint."""
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    return hasher.hexdigest()


def derive_seed(base_seed: int, record_id: str, dataset_hash: str, salt: str) -> int:
    """Derive deterministic seed from inputs."""
    combined = f"{base_seed}:{dataset_hash}:{record_id}:{salt}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_cache_payload(
    *,
    dataset_hash: str,
    record_id: str,
    text_hash: str,
    seed: int,
    model: str,
    api_base: str | None,
    max_input_tokens: int,
    page_tokens: int,
    prompt_hash: str,
    error_rate: float,
) -> tuple[str, JsonDict]:
    """Build cache key and payload for LLM call."""
    payload: JsonDict = {
        "schema_hash": schema_hash(),
        "dataset_hash": dataset_hash,
        "record_id": record_id,
        "text_hash": text_hash,
        "seed": seed,
        "model": model,
        "api_base": api_base,
        "max_input_tokens": max_input_tokens,
        "page_tokens": page_tokens,
        "prompt_hash": prompt_hash,
        "error_rate": error_rate,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(payload_json).hexdigest()
    return cache_key, payload


def validate_schema(paths: Sequence[Path]) -> None:
    """Validate input JSONL files have required fields."""
    for path in paths:
        for row in iter_jsonl(path):
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing required fields {missing} in {path}.")
            try:
                BaseRecord.model_validate(row)
            except ValidationError as exc:
                raise ValueError(f"Invalid record in {path}: {exc}") from exc


def load_records(paths: Sequence[Path]) -> list[BaseRecord]:
    """Load records from JSONL files."""
    records: list[BaseRecord] = []
    for path in paths:
        for row in iter_jsonl(path):
            records.append(BaseRecord.model_validate(row))
    return records


async def call_llm(
    *,
    paged_text: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
    use_prompt_caching: bool = True,
) -> VibeResponse:
    """Call LLM for vibe extraction.

    Args:
        paged_text: The paginated text to extract vibes from
        model: LiteLLM model string
        api_key: API key for the model provider
        api_base: API base URL override
        system_prompt: System prompt with instructions and schema
        use_prompt_caching: If True, enable prompt caching on system prompt.
            Supported providers: OpenAI, Anthropic, Bedrock, Deepseek.
            This caches the system prompt (instructions + schema) across calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": paged_text},
    ]

    # Enable prompt caching on the system message.
    # This caches the system prompt (instructions + schema) across all texts,
    # reducing input token costs significantly for supported providers.
    # See: https://docs.litellm.ai/docs/completion/prompt_caching
    cache_control: list[dict[str, str]] | None = None
    if use_prompt_caching:
        cache_control = [{"location": "message", "role": "system"}]

    return await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=VibeResponse,
        api_key=api_key,
        api_base=api_base,
        model_kwargs={},
        cache_control_injection_points=cache_control,
    )


def _coerce_augmented(value: Any, fallback: str) -> str:
    """Coerce augmented text result."""
    match value:
        case list() if value:
            return str(value[0])
        case str():
            return value
        case _:
            return fallback


def _augment_text(augmenter: Any, text: str) -> str:
    """Apply character augmentation."""
    augmented = augmenter.augment(text)
    return _coerce_augmented(augmented, text)


async def apply_noise(
    response: VibeResponse,
    *,
    seed: int,
    error_rate: float,
    lock: asyncio.Lock,
) -> VibeResponse:
    """Apply noise corruption to vibes for robustness training."""
    if error_rate <= 0:
        return response

    rng = random.Random(seed)
    async with lock:
        try:
            import nlpaug.util as nlpaug_util  # type: ignore[import]
            from nlpaug.augmenter.char import RandomCharAug  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("nlpaug not installed: %s", exc)
            raise SystemExit(
                "nlpaug is required for noise injection. "
                "Install data_work/requirements.txt or set --error-rate 0."
            ) from exc

        nlpaug_util.Randomness.seed(seed)
        augmenter = RandomCharAug(
            action="substitute", aug_char_p=min(DEFAULT_CHAR_AUG_P, error_rate)
        )

        data = response.model_dump()
        for vibe in data["vibes"]:
            if rng.random() >= error_rate:
                continue
            for character in vibe.get("characters", []):
                for descriptor in character.get("character_perceived_vibes", []):
                    for key, value in descriptor.items():
                        descriptor[key] = _augment_text(augmenter, value)
            for descriptor in vibe.get("scene_vibes", []):
                for key, value in descriptor.items():
                    descriptor[key] = _augment_text(augmenter, value)
            tags = [_augment_text(augmenter, tag) for tag in vibe.get("tags", [])]
            vibe["tags"] = tags

        return VibeResponse.model_validate(data)


async def process_record(
    *,
    record: BaseRecord,
    dataset_hash: str,
    system_prompt: str,
    llm_call: Callable[[str, str, JsonDict], Awaitable[VibeResponse]],
    error_rate: float,
    noise_lock: asyncio.Lock,
    model: str,
    api_base: str | None,
    max_input_tokens: int,
    page_tokens: int,
    seed: int,
) -> RecordResult:
    """Process a single input record to extract vibes."""
    paged_text = paginate_text(record.text, max_input_tokens, page_tokens)
    text_hash = hash_text(paged_text)
    record_id = str(record.id_in_dataset)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    cache_key, payload = build_cache_payload(
        dataset_hash=dataset_hash,
        record_id=record_id,
        text_hash=text_hash,
        seed=seed,
        model=model,
        api_base=api_base,
        max_input_tokens=max_input_tokens,
        page_tokens=page_tokens,
        prompt_hash=prompt_hash,
        error_rate=error_rate,
    )

    async def _llm_call() -> VibeResponse:
        return await llm_call(paged_text, cache_key, payload)

    try:
        response = await _llm_call()
        normalized = VibeResponse(vibes=normalize_vibe_indices(response.vibes))

        noise_seed = derive_seed(seed, record_id, dataset_hash, "noise")
        noisy = await apply_noise(
            normalized,
            seed=noise_seed,
            error_rate=error_rate,
            lock=noise_lock,
        )

        # Build rows WITHOUT split assignment (done later after dedupe)
        rows = build_vibe_rows(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split_name="",  # Will be assigned after dedupe
            original=normalized,
            noisy=noisy,
        )

        rows = [row.model_copy(update={"vibe_model": model}) for row in rows]

        return RecordResult(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            rows=rows,
            error=None,
        )

    except Exception as exc:
        _LOGGER.warning("Failed processing record %s: %s", record_id, exc, exc_info=True)
        return RecordResult(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            rows=[],
            error=str(exc),
        )


def _row_is_noisy(row: VibeRow) -> bool:
    """Check if row has been noise-corrupted."""
    return row.vibe_noisy != row.vibe_original or row.tags_noisy != row.tags_original


def _row_noise_key(row: VibeRow, error_seed: int, dataset_hash: str) -> int:
    """Compute deterministic noise key for row."""
    combined = (
        f"{error_seed}:{dataset_hash}:"
        f"{row.id_in_dataset}:{row.vibe_index}:"
        f"{row.vibe_scope}:{row.vibe_level}"
    )
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _force_noise_row(
    rows: list[VibeRow],
    *,
    error_seed: int,
    dataset_hash: str,
) -> None:
    """Force noise on at least one row if none were corrupted."""
    if not rows:
        return

    try:
        import nlpaug.util as nlpaug_util  # type: ignore[import]
        from nlpaug.augmenter.char import RandomCharAug  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("nlpaug not installed for fallback noise: %s", exc)
        return

    target_index = min(
        range(len(rows)),
        key=lambda idx: _row_noise_key(rows[idx], error_seed, dataset_hash),
    )
    target = rows[target_index]
    nlpaug_util.Randomness.seed(error_seed)
    augmenter = RandomCharAug(action="substitute", aug_char_p=DEFAULT_FORCE_CHAR_AUG_P)
    original = str(target.vibe_original or "")
    tags = [str(tag) for tag in target.tags_original]
    rows[target_index] = target.model_copy(
        update={
            "vibe_noisy": _augment_text(augmenter, original),
            "tags_noisy": [_augment_text(augmenter, tag) for tag in tags],
        }
    )


def split_vibe_rows(
    rows: list[VibeRow],
    seed: int,
    dedupe_model: str,
) -> dict[str, list[VibeRow]]:
    """Split vibe rows into train/val/GRPO/test with diversity sampling for GRPO.

    Splitting order (for scientific validity):
    1. TEST (15%): Random sample - held out for final evaluation
    2. SFT-Val (5%): Random sample - must predict TEST performance
    3. GRPO (25%): Diversity-sampled from remaining - maximizes coverage
    4. SFT-Train (55%): Leftovers
    """
    if not rows:
        return {name: [] for name, _ in SPLITS}

    rng = random.Random(seed)
    n_total = len(rows)

    # Calculate counts for each split
    split_counts = {name: int(n_total * ratio) for name, ratio in SPLITS}

    # Ensure we use all rows (assign remainder to SFT-Train)
    assigned = sum(split_counts.values())
    if assigned < n_total:
        split_counts["SFT-Train"] += n_total - assigned

    # Step 1: Random sample for TEST (evaluation - must be representative)
    indices = list(range(n_total))
    rng.shuffle(indices)

    test_count = split_counts["TEST"]
    test_indices = set(indices[:test_count])
    test_rows = [rows[i].model_copy(update={"split": "TEST"}) for i in sorted(test_indices)]

    # Step 2: Random sample for SFT-Val (evaluation - must predict TEST)
    remaining_after_test = [i for i in indices if i not in test_indices]
    val_count = split_counts["SFT-Val"]
    val_indices = set(remaining_after_test[:val_count])
    val_rows = [rows[i].model_copy(update={"split": "SFT-Val"}) for i in sorted(val_indices)]

    # Pool for training (after removing evaluation sets)
    train_pool_indices = [i for i in remaining_after_test if i not in val_indices]
    train_pool = [rows[i] for i in train_pool_indices]

    # Step 3: Diversity sample for GRPO (from training pool)
    grpo_count = split_counts["GRPO"]

    if grpo_count > 0 and train_pool:
        grpo_rows, grpo_local_indices = diversity_sample_rows(
            train_pool,
            grpo_count,
            model_name=dedupe_model,
            seed=seed,
        )
        grpo_indices_set = set(grpo_local_indices)
        grpo_rows = [row.model_copy(update={"split": "GRPO"}) for row in grpo_rows]
    else:
        grpo_rows = []
        grpo_indices_set = set()

    # Step 4: SFT-Train gets the leftovers
    train_rows = [
        row.model_copy(update={"split": "SFT-Train"})
        for i, row in enumerate(train_pool)
        if i not in grpo_indices_set
    ]

    return {
        "SFT-Train": train_rows,
        "SFT-Val": val_rows,
        "GRPO": grpo_rows,
        "TEST": test_rows,
    }


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Write run configuration to output directory."""
    config = {
        "seed": args.seed,
        "model": args.model,
        "max_input_tokens": args.max_input_tokens,
        "page_tokens": args.page_tokens,
        "error_rate": args.error_rate,
        "dedupe_model": args.dedupe_model,
        "dedupe_threshold": args.dedupe_threshold,
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

    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    # Check output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, _ in SPLITS:
        output_path = output_dir / f"{split_name}.jsonl"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {output_path}. Use --overwrite to replace.")

    # Load environment and setup API
    load_env_file(args.env_file)
    model, api_base = normalize_model_and_base(args.model, None)
    api_key = resolve_api_key_for_models(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        models=[(model, api_base)],
    )
    use_prompt_caching = not args.no_prompt_caching

    _LOGGER.info("Model: %s", model)
    if use_prompt_caching:
        _LOGGER.info(
            "Prompt caching: ENABLED (system prompt cached across texts; "
            "supported by OpenAI, Anthropic, Bedrock, Deepseek)"
        )
    else:
        _LOGGER.info("Prompt caching: DISABLED (full input token cost per text)")

    # Validate input schema
    _LOGGER.info("Validating input schema...")
    validate_schema(jsonl_paths)

    # Load records
    _LOGGER.info("Loading records...")
    records = load_records(jsonl_paths)
    dataset_hash = hash_paths(jsonl_paths)
    _LOGGER.info("Loaded %d records from %d files.", len(records), len(jsonl_paths))

    # Apply input limit if specified
    if args.limit > 0 and len(records) > args.limit:
        _LOGGER.info("Limiting to %d input records (from %d).", args.limit, len(records))
        rng = random.Random(args.seed)
        records = rng.sample(records, args.limit)

    # Write run config
    write_run_config(output_dir, args)

    # Setup LLM infrastructure
    system_prompt = build_vibe_prompt()
    cache = SqliteCache(args.cache_path)
    await cache.initialize()

    semaphore = asyncio.Semaphore(args.max_concurrency)
    limiter = AsyncRateLimiter(args.max_qps)
    tracker = EWMAErrorTracker(alpha=args.ewma_alpha, min_samples=args.ewma_min_samples)
    noise_lock = asyncio.Lock()

    @cached_async(
        cache=cache,
        key_fn=lambda paged_text, cache_key, payload: (cache_key, payload),
        serializer=lambda response: response.model_dump(),
        deserializer=lambda data: VibeResponse.model_validate(data),
    )
    @retry_async(max_retries=args.max_retries, base_delay=args.retry_base_delay, logger=_LOGGER)
    @rate_limited(limiter)
    @with_semaphore(semaphore)
    async def cached_call(paged_text: str, cache_key: str, payload: JsonDict) -> VibeResponse:
        return await call_llm(
            paged_text=paged_text,
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=system_prompt,
            use_prompt_caching=use_prompt_caching,
        )

    # Process all records with periodic progress writes
    _LOGGER.info("Extracting vibes from %d records...", len(records))

    # Setup periodic progress writer
    progress_path = output_dir / PROGRESS_FILE_NAME
    progress_writer = AsyncPeriodicWriter(
        progress_path,
        interval_seconds=args.write_interval,
        overwrite=args.overwrite,
    )
    await progress_writer.start()
    _LOGGER.info(
        "Progress file: %s (updates every %.0fs)",
        progress_path,
        args.write_interval,
    )

    async def bound_process(index: int, record: BaseRecord) -> tuple[int, RecordResult]:
        result = await process_record(
            record=record,
            dataset_hash=dataset_hash,
            system_prompt=system_prompt,
            llm_call=cached_call,
            error_rate=args.error_rate,
            noise_lock=noise_lock,
            model=model,
            api_base=api_base,
            max_input_tokens=args.max_input_tokens,
            page_tokens=args.page_tokens,
            seed=args.seed,
        )
        return index, result

    tasks = [
        asyncio.create_task(bound_process(index, record)) for index, record in enumerate(records)
    ]

    all_rows: list[VibeRow] = []
    error_count = 0
    abort_event = asyncio.Event()
    max_vibes_reached = False

    try:
        with tqdm(total=len(records), desc="Extracting vibes") as progress:
            for task in asyncio.as_completed(tasks):
                index, result = await task
                progress.update(1)

                tracker.update(success=result.error is None)
                if result.error:
                    error_count += 1
                    _LOGGER.warning("Record %s failed: %s", result.id_in_dataset, result.error)
                else:
                    all_rows.extend(result.rows)
                    # Write to progress file periodically
                    await progress_writer.add_rows(result.rows)

                # Check if max-vibes limit reached
                if args.max_vibes > 0 and len(all_rows) >= args.max_vibes:
                    max_vibes_reached = True
                    _LOGGER.info(
                        "Reached --max-vibes limit (%d vibes). Stopping early.",
                        args.max_vibes,
                    )
                    break

                if tracker.threshold_reached(args.ewma_threshold):
                    abort_event.set()
                    _LOGGER.error(
                        "Excessive error rate (EWMA=%.2f). Aborting.",
                        tracker.error_rate,
                    )
                    break
    finally:
        # Always stop the writer to flush remaining
        await progress_writer.stop()
        _LOGGER.info(
            "Progress file: wrote %d rows total",
            progress_writer.total_written,
        )

    # Cancel remaining tasks if we stopped early
    if abort_event.is_set() or max_vibes_reached:
        cancelled = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        if cancelled:
            _LOGGER.info("Cancelled %d remaining tasks.", cancelled)
        if abort_event.is_set():
            raise SystemExit("Excessive error rate - try again later.")

    _LOGGER.info(
        "Vibe extraction complete: %d rows from %d records (%d errors).",
        len(all_rows),
        len(records),
        error_count,
    )

    # Dedupe on vibe_original content
    _LOGGER.info("Deduplicating vibes (threshold=%.2f)...", args.dedupe_threshold)
    original_count = len(all_rows)
    deduped_rows, removed_count = dedupe_vibe_rows(
        all_rows,
        threshold=args.dedupe_threshold,
        model_name=args.dedupe_model,
    )
    _LOGGER.info(
        "Dedupe: kept=%d removed=%d (%.1f%% reduction)",
        len(deduped_rows),
        removed_count,
        100.0 * removed_count / original_count if original_count else 0,
    )

    # Split into train/val/GRPO/test
    _LOGGER.info("Splitting into train/val/GRPO/test...")
    split_map = split_vibe_rows(deduped_rows, args.seed, args.dedupe_model)

    for split_name, split_rows in split_map.items():
        _LOGGER.info("  %s: %d rows", split_name, len(split_rows))

    # Apply limit per split if specified
    if args.limit_per_split > 0:
        for split_name in split_map:
            split_map[split_name] = split_map[split_name][: args.limit_per_split]

    # Ensure at least one noisy row per split
    for split_name, split_rows in split_map.items():
        if args.error_rate > 0 and split_rows:
            noisy_count = sum(1 for row in split_rows if _row_is_noisy(row))
            if noisy_count == 0:
                _LOGGER.warning("Split %s has no noisy rows; forcing noise on one.", split_name)
                _force_noise_row(split_rows, error_seed=args.seed, dataset_hash=dataset_hash)

    # Write outputs (incrementally to each split file)
    _LOGGER.info("Writing output files...")
    for split_name, split_rows in split_map.items():
        output_path = output_dir / f"{split_name}.jsonl"
        if args.overwrite and output_path.exists():
            output_path.unlink()

        with output_path.open("a", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
                f.flush()  # Ensure written to disk

        _LOGGER.info("  Wrote %d rows to %s", len(split_rows), output_path)

    _LOGGER.info("Done! Output in %s", output_dir)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    _LOGGER.warning(
        "Seed=%d provides partial reproducibility. External LLM calls are "
        "non-deterministic due to API variability. Seed controls: split assignment, "
        "noise injection, diversity sampling. SQLite cache ensures re-runs produce "
        "identical outputs for cached calls.",
        args.seed,
    )

    try:
        import uvloop  # type: ignore[import]

        uvloop.install()
    except ImportError:
        _LOGGER.info("uvloop not installed; using default asyncio event loop.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
