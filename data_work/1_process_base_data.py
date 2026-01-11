"""Process base data into train/test/eval splits with vibe extraction."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence
from urllib.parse import urlparse

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError

from data_work.llm_cache import SqliteCache, cached_async
from data_work.resilience import (
    AsyncRateLimiter,
    EWMAErrorTracker,
    rate_limited,
    retry_async,
    with_semaphore,
)
from data_work.vibe_schema import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_PAGE_TOKENS,
    SPLITS,
    VibeResponse,
    build_vibe_prompt,
    normalize_vibe_indices,
    paginate_text,
    schema_hash,
    split_records,
)

_LOGGER = logging.getLogger("data_work.process_base_data")

REQUIRED_FIELDS = ("created", "metadata", "dataset", "id_in_dataset", "text")

DEFAULT_MODEL = "openrouter/openai/gpt-oss-20b"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
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


class BaseRecord(BaseModel):
    created: str
    metadata: dict[str, Any]
    dataset: str
    id_in_dataset: Any = Field(alias="id_in_dataset")
    text: str


class ProcessedRecord(BaseModel):
    created: str
    metadata: dict[str, Any]
    dataset: str
    id_in_dataset: Any
    split: str
    text_hash: str
    text_with_pages: str
    model: str
    seed: int
    vibes: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split base JSONL files into SFT/GRPO/TEST splits and extract vibe objects via LiteLLM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing base JSONL files created by 0_download_base_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed JSONL splits will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic splitting and error injection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "LiteLLM model string. For OpenRouter use openrouter/openai/gpt-oss-20b "
            "or a full OpenRouter URL."
        ),
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Optional API base URL override (e.g., https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override. Otherwise read from --api-key-env.",
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
        help="Optional .env file to load (defaults to .env or examples/.env if present).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum in-flight LLM requests per split.",
    )
    parser.add_argument(
        "--max-qps",
        type=float,
        default=DEFAULT_MAX_QPS,
        help="Rate limit in requests per second (0 disables rate limiting).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Number of retry attempts after the first failure.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help="Base delay (seconds) for exponential backoff.",
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
        "--error-rate",
        type=float,
        default=DEFAULT_ERROR_RATE,
        help="Probability that a vibe object is noise-corrupted.",
    )
    parser.add_argument(
        "--error-seed",
        type=int,
        default=None,
        help="Seed for deterministic error injection (defaults to --seed).",
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
        help="Abort if EWMA error rate exceeds this value.",
    )
    parser.add_argument(
        "--ewma-min-samples",
        type=int,
        default=DEFAULT_EWMA_MIN_SAMPLES,
        help="Minimum samples before enforcing EWMA abort threshold.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Process only the first N records per split (0 = no limit).",
    )
    parser.add_argument(
        "--only-splits",
        nargs="*",
        choices=[name for name, _ in SPLITS],
        default=None,
        help="Optional subset of splits to process.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="SQLite cache path for LLM responses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files instead of resuming.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume behavior when output files already exist.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def validate_schema(paths: Iterable[Path]) -> None:
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
    records: list[BaseRecord] = []
    for path in paths:
        for row in iter_jsonl(path):
            records.append(BaseRecord.model_validate(row))
    return records


def hash_paths(paths: Sequence[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_env_file(env_path: Path | None) -> None:
    candidates = [Path(".env"), Path("examples/.env")]
    resolved = env_path
    if resolved is None:
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate
                break
    if resolved is None or not resolved.is_file():
        return
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def normalize_model_and_base(model: str, api_base: str | None) -> tuple[str, str | None]:
    if model.startswith("http"):
        parsed = urlparse(model)
        if "openrouter.ai" in parsed.netloc and parsed.path:
            model = f"openrouter/{parsed.path.lstrip('/')}"
            api_base = api_base or DEFAULT_OPENROUTER_API_BASE
    if model.startswith("openrouter/") and api_base is None:
        api_base = DEFAULT_OPENROUTER_API_BASE
    return model, api_base


def requires_api_key(model: str, api_base: str | None) -> bool:
    match model:
        case _ if model.startswith("openrouter/"):
            return True
        case _ if model.startswith("openai/"):
            return True
        case _ if model.startswith("anthropic/"):
            return True
        case _:
            return api_base is not None and "openrouter.ai" in api_base


def resolve_api_key(
    *,
    api_key: str | None,
    api_key_env: str,
    model: str,
    api_base: str | None,
) -> str | None:
    if api_key:
        return api_key
    env_value = os.environ.get(api_key_env)
    if env_value:
        return env_value
    if requires_api_key(model, api_base):
        raise SystemExit(
            f"API key required for model '{model}'. Set {api_key_env} or pass --api-key."
        )
    return None


def warn_if_context_limit(model: str, max_input_tokens: int) -> None:
    if "gpt-oss-20b" in model and max_input_tokens > DEFAULT_OPENROUTER_CONTEXT_LIMIT:
        _LOGGER.warning(
            "Requested %s tokens exceeds OpenRouter GPT-OSS context limit (%s). "
            "Use --max-input-tokens to avoid 400 errors.",
            max_input_tokens,
            DEFAULT_OPENROUTER_CONTEXT_LIMIT,
        )


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
    error_seed: int,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
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
        "error_seed": error_seed,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(payload_json).hexdigest()
    return cache_key, payload


def derive_seed(base_seed: int, record_id: str, dataset_hash: str, salt: str) -> int:
    combined = f"{base_seed}:{dataset_hash}:{record_id}:{salt}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


async def call_llm(
    *,
    paged_text: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
) -> VibeResponse:
    try:
        import litellm  # type: ignore[import]
        from litellm import acompletion  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("LiteLLM not installed: %s", exc)
        raise SystemExit("litellm is required. Install via data_work/requirements.txt.") from exc

    litellm.turn_off_message_logging = True
    litellm.disable_streaming_logging = True
    litellm.logging = False

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": paged_text},
    ]
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": VibeResponse,
    }
    if api_key:
        request["api_key"] = api_key
    if api_base:
        request["api_base"] = api_base

    response = await acompletion(**request)
    content = None
    match response:
        case {"choices": choices} if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
        case _ if hasattr(response, "choices"):
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "parsed", None) or getattr(message, "content", None)
        case _:
            content = None

    if isinstance(content, dict):
        return VibeResponse.model_validate(content)
    if isinstance(content, VibeResponse):
        return content
    if isinstance(content, BaseModel):
        return VibeResponse.model_validate(content.model_dump())
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            _LOGGER.warning("LiteLLM returned non-JSON content: %s", content)
            raise ValueError("LiteLLM returned invalid JSON") from exc
        return VibeResponse.model_validate(data)

    raise ValueError("LiteLLM response missing content")


async def apply_noise(
    response: VibeResponse,
    *,
    seed: int,
    error_rate: float,
    lock: asyncio.Lock,
) -> VibeResponse:
    if error_rate <= 0:
        return response
    rng = random.Random(seed)
    async with lock:
        try:
            import nlpaug.util as nlpaug_util  # type: ignore[import]
            from nlpaug.augmenter.char import RandomCharAug  # type: ignore[import]
        except ImportError as exc:
            _LOGGER.warning("nlpaug not installed: %s", exc)
            return response
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
                        augmented = augmenter.augment(value)
                        match augmented:
                            case list() if augmented:
                                descriptor[key] = str(augmented[0])
                            case str():
                                descriptor[key] = augmented
                            case _:
                                descriptor[key] = value
            for descriptor in vibe.get("scene_vibes", []):
                for key, value in descriptor.items():
                    augmented = augmenter.augment(value)
                    match augmented:
                        case list() if augmented:
                            descriptor[key] = str(augmented[0])
                        case str():
                            descriptor[key] = augmented
                        case _:
                            descriptor[key] = value
            tags = []
            for tag in vibe.get("tags", []):
                augmented = augmenter.augment(tag)
                match augmented:
                    case list() if augmented:
                        tags.append(str(augmented[0]))
                    case str():
                        tags.append(augmented)
                    case _:
                        tags.append(tag)
            vibe["tags"] = tags
        return VibeResponse.model_validate(data)


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    for row in iter_jsonl(path):
        record_id = row.get("id_in_dataset")
        if record_id is not None:
            processed.add(str(record_id))
    return processed


async def process_record(
    *,
    record: BaseRecord,
    split_name: str,
    dataset_hash: str,
    system_prompt: str,
    llm_call: Callable[[str, str, dict[str, Any]], Awaitable[VibeResponse]],
    error_rate: float,
    error_seed: int,
    noise_lock: asyncio.Lock,
    model: str,
    api_base: str | None,
    max_input_tokens: int,
    page_tokens: int,
    seed: int,
) -> ProcessedRecord:
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
        error_seed=error_seed,
    )

    async def _llm_call() -> VibeResponse:
        return await llm_call(paged_text, cache_key, payload)

    try:
        response = await _llm_call()
        normalized = VibeResponse(vibes=normalize_vibe_indices(response.vibes))
        noise_seed = derive_seed(error_seed, record_id, dataset_hash, "noise")
        noisy = await apply_noise(
            normalized,
            seed=noise_seed,
            error_rate=error_rate,
            lock=noise_lock,
        )
        return ProcessedRecord(
            created=record.created,
            metadata=record.metadata,
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split=split_name,
            text_hash=text_hash,
            text_with_pages=paged_text,
            model=model,
            seed=seed,
            vibes=[item.model_dump() for item in noisy.vibes],
            error=None,
        )
    except Exception as exc:
        _LOGGER.warning("Failed processing record %s: %s", record_id, exc, exc_info=True)
        return ProcessedRecord(
            created=record.created,
            metadata=record.metadata,
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split=split_name,
            text_hash=text_hash,
            text_with_pages=paged_text,
            model=model,
            seed=seed,
            vibes=[],
            error=str(exc),
        )


async def process_split(
    *,
    split_name: str,
    records: Sequence[BaseRecord],
    dataset_hash: str,
    system_prompt: str,
    cache: SqliteCache,
    model: str,
    api_key: str | None,
    api_base: str | None,
    max_input_tokens: int,
    page_tokens: int,
    max_concurrency: int,
    max_qps: float,
    max_retries: int,
    retry_base_delay: float,
    error_rate: float,
    error_seed: int,
    ewma_alpha: float,
    ewma_threshold: float,
    ewma_min_samples: int,
    output_path: Path,
    overwrite: bool,
    resume: bool,
    seed: int,
) -> None:
    from tqdm import tqdm  # type: ignore[import]

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        elif not resume:
            raise SystemExit(f"Output exists (use --overwrite or --no-resume): {output_path}")

    processed_ids = load_processed_ids(output_path) if resume else set()
    pending_records = [
        record for record in records if str(record.id_in_dataset) not in processed_ids
    ]
    if not pending_records:
        _LOGGER.info("Split %s already processed (nothing new).", split_name)
        return

    semaphore = asyncio.Semaphore(max_concurrency)
    limiter = AsyncRateLimiter(max_qps)
    tracker = EWMAErrorTracker(alpha=ewma_alpha, min_samples=ewma_min_samples)
    noise_lock = asyncio.Lock()

    @cached_async(
        cache=cache,
        key_fn=lambda paged_text, cache_key, payload: (cache_key, payload),
        serializer=lambda response: response.model_dump(),
        deserializer=lambda data: VibeResponse.model_validate(data),
    )
    @retry_async(max_retries=max_retries, base_delay=retry_base_delay, logger=_LOGGER)
    @rate_limited(limiter)
    @with_semaphore(semaphore)
    async def cached_call(paged_text: str, cache_key: str, payload: dict[str, Any]) -> VibeResponse:
        return await call_llm(
            paged_text=paged_text,
            model=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=system_prompt,
        )

    async def bound_process(index: int, record: BaseRecord) -> tuple[int, ProcessedRecord]:
        result = await process_record(
            record=record,
            split_name=split_name,
            dataset_hash=dataset_hash,
            system_prompt=system_prompt,
            llm_call=cached_call,
            error_rate=error_rate,
            error_seed=error_seed,
            noise_lock=noise_lock,
            model=model,
            api_base=api_base,
            max_input_tokens=max_input_tokens,
            page_tokens=page_tokens,
            seed=seed,
        )
        return index, result

    tasks = [
        asyncio.create_task(bound_process(index, record))
        for index, record in enumerate(pending_records)
    ]

    results: dict[int, ProcessedRecord] = {}
    abort_event = asyncio.Event()

    with tqdm(total=len(pending_records), desc=split_name) as progress:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
            progress.update(1)
            tracker.update(success=result.error is None)
            if tracker.threshold_reached(ewma_threshold):
                abort_event.set()
                _LOGGER.error(
                    "Excessive error rate detected (EWMA=%.2f) for split %s. Aborting.",
                    tracker.error_rate,
                    split_name,
                )
                break

    if abort_event.is_set():
        for task in tasks:
            if not task.done():
                task.cancel()
        raise SystemExit("Excessive error rate detected - try again later.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index in sorted(results):
            row = results[index].model_dump()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    load_env_file(args.env_file)
    model, api_base = normalize_model_and_base(args.model, args.api_base)
    warn_if_context_limit(model, args.max_input_tokens)
    api_key = resolve_api_key(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        model=model,
        api_base=api_base,
    )

    validate_schema(jsonl_paths)
    records = load_records(jsonl_paths)
    dataset_hash = hash_paths(jsonl_paths)

    split_map = split_records(records, args.seed)
    target_splits = args.only_splits or [name for name, _ in SPLITS]
    system_prompt = build_vibe_prompt()
    cache = SqliteCache(args.cache_path)
    await cache.initialize()

    error_seed = args.error_seed if args.error_seed is not None else args.seed

    for split_name in target_splits:
        split_records_list = split_map.get(split_name, [])
        if args.limit_per_split > 0:
            split_records_list = split_records_list[: args.limit_per_split]
        await process_split(
            split_name=split_name,
            records=split_records_list,
            dataset_hash=dataset_hash,
            system_prompt=system_prompt,
            cache=cache,
            model=model,
            api_key=api_key,
            api_base=api_base,
            max_input_tokens=args.max_input_tokens,
            page_tokens=args.page_tokens,
            max_concurrency=args.max_concurrency,
            max_qps=args.max_qps,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            error_rate=args.error_rate,
            error_seed=error_seed,
            ewma_alpha=args.ewma_alpha,
            ewma_threshold=args.ewma_threshold,
            ewma_min_samples=args.ewma_min_samples,
            output_path=output_dir / f"{split_name}.jsonl",
            overwrite=args.overwrite,
            resume=args.resume,
            seed=args.seed,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        import uvloop  # type: ignore[import]

        uvloop.install()
    except ImportError:
        _LOGGER.info("uvloop not installed; using default asyncio event loop.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
