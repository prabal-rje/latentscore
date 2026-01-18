"""Process base data into train/test/eval splits with vibe extraction."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError

from common.prompts import build_config_generation_prompt
from data_work.lib.config_batcher import (
    ConfigBatcher,
    build_batch_prompt,
    build_batch_response_keys,
    build_batch_response_model,
    parse_batch_response,
)
from data_work.lib.config_io import build_config_hash, load_config_file, write_config_file
from data_work.lib.dedupe import dedupe_records, diversity_sample
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_cache import SqliteCache, cached_async
from data_work.lib.llm_client import (
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
)
from data_work.lib.music_schema import MusicConfigPromptPayload
from data_work.lib.music_schema import schema_hash as music_schema_hash
from data_work.lib.record_builder import VibeRow, build_vibe_rows
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
    split_records_with_diversity,
)

_LOGGER = logging.getLogger("data_work.process_base_data")

REQUIRED_FIELDS = ("created", "metadata", "dataset", "id_in_dataset", "text")

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_CONFIG_MODEL = "openai/gpt-oss-20b"
DEFAULT_CONTEXT_LIMIT = 131_072
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
DEFAULT_CONFIG_MAX_CONCURRENCY = 4
DEFAULT_CONFIG_MAX_QPS = 2.0
DEFAULT_CONFIG_MAX_RETRIES = 3
DEFAULT_CONFIG_RETRY_BASE_DELAY = 0.5
DEFAULT_CONFIG_BATCH_SIZE = 1
DEFAULT_CONFIG_BATCH_WAIT_MS = 200
DEFAULT_DEDUPE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEDUPE_THRESHOLD = 0.97
RUN_CONFIG_NAME = "run_config.json"
VIBE_SCOPE_FILTER_CHOICES = ("scene", "character")
VIBE_LEVEL_FILTER_CHOICES = ("xl", "lg", "m", "sm", "xs")


class BaseRecord(BaseModel):
    created: str
    metadata: dict[str, Any]
    dataset: str
    id_in_dataset: Any = Field(alias="id_in_dataset")
    text: str


class RecordResult(BaseModel):
    dataset: str
    id_in_dataset: Any
    split: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def _add_arg(
    parser: argparse.ArgumentParser,
    *args: str,
    default: Any,
    use_defaults: bool,
    **kwargs: Any,
) -> None:
    if use_defaults:
        parser.add_argument(*args, default=default, **kwargs)
    else:
        parser.add_argument(*args, default=argparse.SUPPRESS, **kwargs)


def _build_parser(*, include_extra: bool, use_defaults: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split base JSONL files into SFT/GRPO/TEST splits and extract vibe objects via LiteLLM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit."
    )
    parser.add_argument(
        "--extra-help",
        action="store_true",
        help="Show full help with advanced options and exit.",
    )
    _add_arg(
        parser,
        "--config-file",
        type=Path,
        default=None,
        use_defaults=use_defaults,
        help="Optional JSON config file to preload arguments.",
    )
    _add_arg(
        parser,
        "--input-dir",
        type=Path,
        default=None,
        use_defaults=use_defaults,
        help="Directory containing base JSONL files created by 01_download_base_data.py.",
    )
    _add_arg(
        parser,
        "--output-dir",
        type=Path,
        default=None,
        use_defaults=use_defaults,
        help="Directory where processed JSONL splits will be written.",
    )
    _add_arg(
        parser,
        "--source-dir",
        type=Path,
        default=None,
        use_defaults=use_defaults,
        help="Optional directory of processed split JSONLs to transform (skips LLM extraction).",
    )
    _add_arg(
        parser,
        "--seed",
        type=int,
        default=0,
        use_defaults=use_defaults,
        help="Seed for deterministic splitting and error injection.",
    )
    _add_arg(
        parser,
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        use_defaults=use_defaults,
        help="LiteLLM model string (e.g., openai/gpt-oss-20b, anthropic/claude-sonnet-4-20250514).",
    )
    _add_arg(
        parser,
        "--config-model",
        type=str,
        default=DEFAULT_CONFIG_MODEL,
        use_defaults=use_defaults,
        help="LiteLLM model string for music config generation.",
    )
    _add_arg(
        parser,
        "--api-key",
        type=str,
        default=None,
        use_defaults=use_defaults,
        help="Optional API key override. Otherwise read from --api-key-env.",
    )
    _add_arg(
        parser,
        "--env-file",
        type=Path,
        default=None,
        use_defaults=use_defaults,
        help="Optional .env file to load (defaults to .env if present).",
    )
    _add_arg(
        parser,
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        use_defaults=use_defaults,
        help="Maximum tokens to feed into the LLM before truncation.",
    )
    _add_arg(
        parser,
        "--limit-per-split",
        type=int,
        default=0,
        use_defaults=use_defaults,
        help="Process only the first N records per split (0 = no limit).",
    )
    _add_arg(
        parser,
        "--only-splits",
        nargs="*",
        choices=[name for name, _ in SPLITS],
        default=None,
        use_defaults=use_defaults,
        help="Optional subset of splits to process.",
    )
    _add_arg(
        parser,
        "--filter-vibe-scope",
        nargs="*",
        choices=list(VIBE_SCOPE_FILTER_CHOICES),
        default=None,
        use_defaults=use_defaults,
        help="Optional vibe_scope filter when transforming existing outputs.",
    )
    _add_arg(
        parser,
        "--filter-vibe-level",
        nargs="*",
        choices=list(VIBE_LEVEL_FILTER_CHOICES),
        default=None,
        use_defaults=use_defaults,
        help="Optional vibe_level filter when transforming existing outputs.",
    )
    _add_arg(
        parser,
        "--overwrite",
        action="store_true",
        default=False,
        use_defaults=use_defaults,
        help="Overwrite existing split files instead of resuming.",
    )
    _add_arg(
        parser,
        "--raw-text-direct",
        action="store_true",
        default=False,
        use_defaults=use_defaults,
        help="Replace vibe text with raw input text when transforming existing outputs.",
    )
    _add_arg(
        parser,
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        use_defaults=use_defaults,
        help="Disable resume behavior when output files already exist.",
    )
    if include_extra:
        _add_arg(
            parser,
            "--api-base",
            type=str,
            default=None,
            use_defaults=use_defaults,
            help="Optional API base URL override (e.g., https://openrouter.ai/api/v1).",
        )
        _add_arg(
            parser,
            "--config-api-base",
            type=str,
            default=None,
            use_defaults=use_defaults,
            help="Optional API base URL override for config generation.",
        )
        _add_arg(
            parser,
            "--api-key-env",
            type=str,
            default="OPENAI_API_KEY",
            use_defaults=use_defaults,
            help="Environment variable for the API key.",
        )
        _add_arg(
            parser,
            "--max-concurrency",
            type=int,
            default=DEFAULT_MAX_CONCURRENCY,
            use_defaults=use_defaults,
            help="Maximum in-flight vibe LLM requests per split.",
        )
        _add_arg(
            parser,
            "--max-qps",
            type=float,
            default=DEFAULT_MAX_QPS,
            use_defaults=use_defaults,
            help="Rate limit in requests per second (0 disables rate limiting).",
        )
        _add_arg(
            parser,
            "--max-retries",
            type=int,
            default=DEFAULT_MAX_RETRIES,
            use_defaults=use_defaults,
            help="Number of retry attempts after the first failure.",
        )
        _add_arg(
            parser,
            "--retry-base-delay",
            type=float,
            default=DEFAULT_RETRY_BASE_DELAY,
            use_defaults=use_defaults,
            help="Base delay (seconds) for exponential backoff.",
        )
        _add_arg(
            parser,
            "--page-tokens",
            type=int,
            default=DEFAULT_PAGE_TOKENS,
            use_defaults=use_defaults,
            help="Approximate tokens per page marker.",
        )
        _add_arg(
            parser,
            "--error-rate",
            type=float,
            default=DEFAULT_ERROR_RATE,
            use_defaults=use_defaults,
            help="Probability that a vibe object is noise-corrupted.",
        )
        _add_arg(
            parser,
            "--error-seed",
            type=int,
            default=None,
            use_defaults=use_defaults,
            help="Seed for deterministic error injection (defaults to --seed).",
        )
        _add_arg(
            parser,
            "--ewma-alpha",
            type=float,
            default=DEFAULT_EWMA_ALPHA,
            use_defaults=use_defaults,
            help="EWMA smoothing factor for error tracking.",
        )
        _add_arg(
            parser,
            "--ewma-threshold",
            type=float,
            default=DEFAULT_EWMA_THRESHOLD,
            use_defaults=use_defaults,
            help="Abort if EWMA error rate exceeds this value.",
        )
        _add_arg(
            parser,
            "--ewma-min-samples",
            type=int,
            default=DEFAULT_EWMA_MIN_SAMPLES,
            use_defaults=use_defaults,
            help="Minimum samples before enforcing EWMA abort threshold.",
        )
        _add_arg(
            parser,
            "--cache-path",
            type=Path,
            default=DEFAULT_CACHE_PATH,
            use_defaults=use_defaults,
            help="SQLite cache path for LLM responses.",
        )
        _add_arg(
            parser,
            "--model-kwargs",
            type=str,
            default=None,
            use_defaults=use_defaults,
            help="JSON dict of extra kwargs for the vibe LLM request.",
        )
        _add_arg(
            parser,
            "--config-model-kwargs",
            type=str,
            default=None,
            use_defaults=use_defaults,
            help="JSON dict of extra kwargs for the config LLM request.",
        )
        _add_arg(
            parser,
            "--config-max-concurrency",
            type=int,
            default=DEFAULT_CONFIG_MAX_CONCURRENCY,
            use_defaults=use_defaults,
            help="Maximum in-flight config LLM requests per split.",
        )
        _add_arg(
            parser,
            "--config-max-qps",
            type=float,
            default=DEFAULT_CONFIG_MAX_QPS,
            use_defaults=use_defaults,
            help="Rate limit for config requests per second (0 disables rate limiting).",
        )
        _add_arg(
            parser,
            "--config-max-retries",
            type=int,
            default=DEFAULT_CONFIG_MAX_RETRIES,
            use_defaults=use_defaults,
            help="Number of retry attempts for config generation.",
        )
        _add_arg(
            parser,
            "--config-retry-base-delay",
            type=float,
            default=DEFAULT_CONFIG_RETRY_BASE_DELAY,
            use_defaults=use_defaults,
            help="Base delay (seconds) for config exponential backoff.",
        )
        _add_arg(
            parser,
            "--config-batch-size",
            type=int,
            default=DEFAULT_CONFIG_BATCH_SIZE,
            use_defaults=use_defaults,
            help="Batch size for vibe-to-config generation (1 disables batching).",
        )
        _add_arg(
            parser,
            "--config-batch-wait-ms",
            type=int,
            default=DEFAULT_CONFIG_BATCH_WAIT_MS,
            use_defaults=use_defaults,
            help="Max wait (ms) to fill a config batch before sending.",
        )
        _add_arg(
            parser,
            "--dedupe-model",
            type=str,
            default=DEFAULT_DEDUPE_MODEL,
            use_defaults=use_defaults,
            help="Sentence-transformers model name for dedupe.",
        )
        _add_arg(
            parser,
            "--dedupe-threshold",
            type=float,
            default=DEFAULT_DEDUPE_THRESHOLD,
            use_defaults=use_defaults,
            help="Cosine similarity threshold for dedupe.",
        )
    return parser


def _default_config() -> dict[str, Any]:
    return {
        "input_dir": None,
        "output_dir": None,
        "source_dir": None,
        "seed": 0,
        "model": DEFAULT_MODEL,
        "config_model": DEFAULT_CONFIG_MODEL,
        "api_base": None,
        "config_api_base": None,
        "api_key": None,
        "api_key_env": "OPENAI_API_KEY",
        "env_file": None,
        "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
        "page_tokens": DEFAULT_PAGE_TOKENS,
        "error_rate": DEFAULT_ERROR_RATE,
        "error_seed": None,
        "ewma_alpha": DEFAULT_EWMA_ALPHA,
        "ewma_threshold": DEFAULT_EWMA_THRESHOLD,
        "ewma_min_samples": DEFAULT_EWMA_MIN_SAMPLES,
        "limit_per_split": 0,
        "only_splits": None,
        "filter_vibe_scopes": None,
        "filter_vibe_levels": None,
        "cache_path": DEFAULT_CACHE_PATH,
        "overwrite": False,
        "resume": True,
        "raw_text_direct": False,
        "max_concurrency": DEFAULT_MAX_CONCURRENCY,
        "max_qps": DEFAULT_MAX_QPS,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_base_delay": DEFAULT_RETRY_BASE_DELAY,
        "config_max_concurrency": DEFAULT_CONFIG_MAX_CONCURRENCY,
        "config_max_qps": DEFAULT_CONFIG_MAX_QPS,
        "config_max_retries": DEFAULT_CONFIG_MAX_RETRIES,
        "config_retry_base_delay": DEFAULT_CONFIG_RETRY_BASE_DELAY,
        "config_batch_size": DEFAULT_CONFIG_BATCH_SIZE,
        "config_batch_wait_ms": DEFAULT_CONFIG_BATCH_WAIT_MS,
        "model_kwargs": {},
        "config_model_kwargs": {},
        "dedupe_model": DEFAULT_DEDUPE_MODEL,
        "dedupe_threshold": DEFAULT_DEDUPE_THRESHOLD,
        "config_file": None,
    }


def _coerce_path(value: Any) -> Path | None:
    """Coerce various input types to Path."""
    match value:
        case None:
            return None
        case Path():
            return value
        case _:
            return Path(str(value))


def _coerce_only_splits(value: Any) -> list[str] | None:
    """Coerce various input types to list of strings."""
    match value:
        case None:
            return None
        case str():
            return [value]
        case _:
            return [str(item) for item in value]


def _coerce_filter_list(value: Any) -> list[str] | None:
    """Coerce various input types to list of strings, flattening nested lists."""
    match value:
        case None:
            return None
        case str():
            return [value]
        case list():
            flattened: list[str] = []
            for item in value:
                match item:
                    case None:
                        continue
                    case list():
                        flattened.extend(str(inner) for inner in item)
                    case _:
                        flattened.append(str(item))
            return flattened or None
        case _:
            return [str(value)]


def _parse_model_kwargs(value: Any, *, label: str) -> dict[str, Any]:
    """Parse model kwargs from various input types."""
    match value:
        case None | "":
            return {}
        case str():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{label} must be valid JSON.") from exc
        case Mapping():
            parsed = dict(value)
        case _:
            raise SystemExit(f"{label} must be a JSON object.")

    match parsed:
        case dict():
            return parsed
        case _:
            raise SystemExit(f"{label} must be a JSON object.")


def _sanitize_model_kwargs(kwargs: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    reserved = {"model", "messages", "temperature", "response_format", "api_key", "api_base"}
    sanitized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in reserved:
            _LOGGER.warning("%s ignores reserved key '%s'.", label, key)
            continue
        sanitized[key] = value
    return sanitized


def _merge_config(
    cli_values: Mapping[str, Any],
    config_values: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = _default_config()
    allowed_keys = set(defaults)
    unknown = [key for key in config_values if key not in allowed_keys]
    if unknown:
        raise SystemExit(f"Unknown config keys: {', '.join(sorted(unknown))}")

    merged = {**defaults, **config_values, **cli_values}
    merged["input_dir"] = _coerce_path(merged["input_dir"])
    merged["output_dir"] = _coerce_path(merged["output_dir"])
    merged["source_dir"] = _coerce_path(merged["source_dir"])
    merged["env_file"] = _coerce_path(merged["env_file"])
    merged["cache_path"] = _coerce_path(merged["cache_path"])
    merged["config_file"] = _coerce_path(merged["config_file"])
    merged["only_splits"] = _coerce_only_splits(merged["only_splits"])
    merged["filter_vibe_scopes"] = _coerce_filter_list(merged["filter_vibe_scopes"])
    merged["filter_vibe_levels"] = _coerce_filter_list(merged["filter_vibe_levels"])
    merged["model_kwargs"] = _sanitize_model_kwargs(
        _parse_model_kwargs(merged.get("model_kwargs"), label="--model-kwargs"),
        label="--model-kwargs",
    )
    merged["config_model_kwargs"] = _sanitize_model_kwargs(
        _parse_model_kwargs(merged.get("config_model_kwargs"), label="--config-model-kwargs"),
        label="--config-model-kwargs",
    )

    if merged["source_dir"] is None:
        if merged["input_dir"] is None or merged["output_dir"] is None:
            raise SystemExit("--input-dir and --output-dir are required (or set in --config-file).")
    else:
        if merged["output_dir"] is None:
            raise SystemExit("--output-dir is required (or set in --config-file).")
        if merged["raw_text_direct"] and merged["input_dir"] is None:
            raise SystemExit("--input-dir is required for --raw-text-direct.")

    allowed_splits = {name for name, _ in SPLITS}
    match merged["only_splits"]:
        case None:
            pass
        case splits:
            invalid = [item for item in splits if item not in allowed_splits]
            if invalid:
                raise SystemExit(f"Invalid splits: {', '.join(invalid)}")

    if merged["filter_vibe_scopes"] is not None:
        invalid = [
            scope
            for scope in merged["filter_vibe_scopes"]
            if scope not in VIBE_SCOPE_FILTER_CHOICES
        ]
        if invalid:
            raise SystemExit(f"Invalid vibe scopes: {', '.join(invalid)}")

    if merged["filter_vibe_levels"] is not None:
        invalid = [
            level
            for level in merged["filter_vibe_levels"]
            if level not in VIBE_LEVEL_FILTER_CHOICES
        ]
        if invalid:
            raise SystemExit(f"Invalid vibe levels: {', '.join(invalid)}")

    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_parser = _build_parser(include_extra=False, use_defaults=True)
    full_parser = _build_parser(include_extra=True, use_defaults=True)
    cli_parser = _build_parser(include_extra=True, use_defaults=False)

    probe, _ = cli_parser.parse_known_args(argv)
    if getattr(probe, "help", False):
        base_parser.print_help()
        raise SystemExit(0)
    if getattr(probe, "extra_help", False):
        full_parser.print_help()
        raise SystemExit(0)

    cli_values = vars(cli_parser.parse_args(argv))
    cli_values.pop("help", None)
    cli_values.pop("extra_help", None)
    config_path = cli_values.get("config_file")
    config_values = load_config_file(config_path) if config_path else {}
    merged = _merge_config(cli_values, config_values)
    return argparse.Namespace(**merged)


_CONFIG_EXCLUDE_KEYS = {
    "input_dir",
    "output_dir",
    "api_key",
    "api_key_env",
    "env_file",
    "cache_path",
    "overwrite",
    "resume",
    "config_file",
    "help",
    "extra_help",
}


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    raw = vars(args)
    config: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _CONFIG_EXCLUDE_KEYS:
            continue
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


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


def load_raw_text_map(input_dir: Path) -> dict[tuple[str, str], str]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")
    text_map: dict[tuple[str, str], str] = {}
    for path in jsonl_paths:
        for row in iter_jsonl(path):
            try:
                record = BaseRecord.model_validate(row)
            except ValidationError as exc:
                raise SystemExit(f"Invalid record in {path}: {exc}") from exc
            key = (record.dataset, str(record.id_in_dataset))
            text_map[key] = record.text
    return text_map


def transform_processed_splits(
    *,
    source_dir: Path,
    output_dir: Path,
    input_dir: Path | None,
    only_splits: Sequence[str] | None,
    filter_vibe_scopes: Sequence[str] | None,
    filter_vibe_levels: Sequence[str] | None,
    raw_text_direct: bool,
    overwrite: bool,
) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    scope_filter = set(filter_vibe_scopes) if filter_vibe_scopes else None
    level_filter = set(filter_vibe_levels) if filter_vibe_levels else None
    raw_text_map = None
    if raw_text_direct:
        if input_dir is None:
            raise SystemExit("--input-dir is required for --raw-text-direct.")
        raw_text_map = load_raw_text_map(input_dir)

    target_splits = only_splits or [name for name, _ in SPLITS]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in target_splits:
        source_path = source_dir / f"{split_name}.jsonl"
        if not source_path.exists():
            raise SystemExit(f"Source split not found: {source_path}")
        output_path = output_dir / f"{split_name}.jsonl"
        if output_path.exists() and not overwrite:
            raise SystemExit(
                f"Output path already exists: {output_path}. Use --overwrite to replace."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in iter_jsonl(source_path):
                if scope_filter and row.get("vibe_scope") not in scope_filter:
                    continue
                if level_filter and row.get("vibe_level") not in level_filter:
                    continue
                if raw_text_map is not None:
                    key = (str(row.get("dataset", "")), str(row.get("id_in_dataset", "")))
                    raw_text = raw_text_map.get(key, "")
                    row["vibe_original"] = raw_text
                    row["vibe_noisy"] = raw_text
                    row["vibe_scope"] = "raw_text"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def warn_if_context_limit(model: str, max_input_tokens: int) -> None:
    if "gpt-oss-20b" in model and max_input_tokens > DEFAULT_CONTEXT_LIMIT:
        _LOGGER.warning(
            "Requested %s tokens exceeds OpenRouter GPT-OSS context limit (%s). "
            "Use --max-input-tokens to avoid 400 errors.",
            max_input_tokens,
            DEFAULT_CONTEXT_LIMIT,
        )


def warn_if_model_overridden(
    *, model: str, default_model: str, max_input_tokens: int, label: str
) -> None:
    if model != default_model:
        _LOGGER.warning(
            "%s model overridden from %s to %s. Current max_input_tokens=%s; "
            "adjust if the new model's context differs.",
            label,
            default_model,
            model,
            max_input_tokens,
        )


def build_cache_payload(
    *,
    dataset_hash: str,
    record_id: str,
    text_hash: str,
    seed: int,
    model: str,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
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
        "model_kwargs": dict(model_kwargs),
        "max_input_tokens": max_input_tokens,
        "page_tokens": page_tokens,
        "prompt_hash": prompt_hash,
        "error_rate": error_rate,
        "error_seed": error_seed,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(payload_json).hexdigest()
    return cache_key, payload


def build_config_cache_payload(
    *,
    dataset_hash: str,
    record_id: str,
    vibe_index: int,
    vibe_scope: str,
    vibe_level: str,
    vibe_text: str,
    seed: int,
    model: str,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    prompt_hash: str,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "schema_hash": music_schema_hash(),
        "dataset_hash": dataset_hash,
        "record_id": record_id,
        "vibe_index": vibe_index,
        "vibe_scope": vibe_scope,
        "vibe_level": vibe_level,
        "vibe_text_hash": hash_text(vibe_text),
        "seed": seed,
        "model": model,
        "api_base": api_base,
        "model_kwargs": dict(model_kwargs),
        "prompt_hash": prompt_hash,
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
    model_kwargs: Mapping[str, Any],
) -> VibeResponse:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": paged_text},
    ]
    return await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=VibeResponse,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )


async def call_music_config(
    *,
    vibe_text: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
    model_kwargs: Mapping[str, Any],
) -> MusicConfigPromptPayload:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": vibe_text},
    ]
    return await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=MusicConfigPromptPayload,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )


async def call_music_config_batch(
    *,
    vibe_texts: Sequence[str],
    model: str,
    api_key: str | None,
    api_base: str | None,
    system_prompt: str,
    model_kwargs: Mapping[str, Any],
) -> list[MusicConfigPromptPayload]:
    keys = build_batch_response_keys(len(vibe_texts))
    prompt = build_batch_prompt(vibe_texts, keys)
    response_model = build_batch_response_model(len(vibe_texts))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = await litellm_structured_completion(
        model=model,
        messages=messages,
        response_model=response_model,
        api_key=api_key,
        api_base=api_base,
        model_kwargs=model_kwargs,
    )
    return parse_batch_response(response, len(vibe_texts))


def _coerce_augmented(value: Any, fallback: str) -> str:
    match value:
        case list() if value:
            return str(value[0])
        case str():
            return value
        case _:
            return fallback


def _augment_text(augmenter: Any, text: str) -> str:
    augmented = augmenter.augment(text)
    return _coerce_augmented(augmented, text)


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
            tags = []
            for tag in vibe.get("tags", []):
                tags.append(_augment_text(augmenter, tag))
            vibe["tags"] = tags
        return VibeResponse.model_validate(data)


def _row_is_noisy(row: Mapping[str, Any]) -> bool:
    return row.get("vibe_noisy") != row.get("vibe_original") or row.get("tags_noisy") != row.get(
        "tags_original"
    )


def _row_noise_key(row: Mapping[str, Any], error_seed: int, dataset_hash: str) -> int:
    combined = (
        f"{error_seed}:{dataset_hash}:"
        f"{row.get('id_in_dataset')}:{row.get('vibe_index')}:"
        f"{row.get('vibe_scope')}:{row.get('vibe_level')}"
    )
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _force_noise_row(
    rows: list[dict[str, Any]],
    *,
    error_seed: int,
    dataset_hash: str,
) -> None:
    if not rows:
        return
    try:
        import nlpaug.util as nlpaug_util  # type: ignore[import]
        from nlpaug.augmenter.char import RandomCharAug  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("nlpaug not installed for fallback noise: %s", exc)
        return
    target = min(rows, key=lambda row: _row_noise_key(row, error_seed, dataset_hash))
    nlpaug_util.Randomness.seed(error_seed)
    augmenter = RandomCharAug(action="substitute", aug_char_p=DEFAULT_FORCE_CHAR_AUG_P)
    original = str(target.get("vibe_original") or "")
    target["vibe_noisy"] = _augment_text(augmenter, original)
    tags_value = target.get("tags_original") or []
    tags = [str(tag) for tag in tags_value]
    target["tags_noisy"] = [_augment_text(augmenter, tag) for tag in tags]


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
    config_prompt: str,
    llm_call: Callable[[str, str, dict[str, Any]], Awaitable[VibeResponse]],
    config_call: Callable[[str, str, dict[str, Any]], Awaitable[MusicConfigPromptPayload]],
    error_rate: float,
    error_seed: int,
    noise_lock: asyncio.Lock,
    model: str,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    config_model: str,
    config_api_base: str | None,
    config_model_kwargs: Mapping[str, Any],
    max_input_tokens: int,
    page_tokens: int,
    seed: int,
) -> RecordResult:
    paged_text = paginate_text(record.text, max_input_tokens, page_tokens)
    text_hash = hash_text(paged_text)
    record_id = str(record.id_in_dataset)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    config_prompt_hash = hashlib.sha256(config_prompt.encode("utf-8")).hexdigest()
    cache_key, payload = build_cache_payload(
        dataset_hash=dataset_hash,
        record_id=record_id,
        text_hash=text_hash,
        seed=seed,
        model=model,
        api_base=api_base,
        model_kwargs=model_kwargs,
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
        rows = build_vibe_rows(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split_name=split_name,
            original=normalized,
            noisy=noisy,
        )

        config_seed = derive_seed(seed, record_id, dataset_hash, "config")

        async def _attach_config(row: VibeRow) -> dict[str, Any]:
            config_key, config_payload = build_config_cache_payload(
                dataset_hash=dataset_hash,
                record_id=record_id,
                vibe_index=row["vibe_index"],
                vibe_scope=row["vibe_scope"],
                vibe_level=row["vibe_level"],
                vibe_text=row["vibe_original"],
                seed=config_seed,
                model=config_model,
                api_base=config_api_base,
                model_kwargs=config_model_kwargs,
                prompt_hash=config_prompt_hash,
            )
            try:
                config = await config_call(row["vibe_original"], config_key, config_payload)
                return {
                    **row,
                    "vibe_model": model,
                    "config_model": config_model,
                    "config_payload": config.model_dump(),
                    "config_error": None,
                }
            except Exception as exc:
                _LOGGER.warning(
                    "Failed config generation for record %s vibe %s/%s: %s",
                    record_id,
                    row["vibe_scope"],
                    row["vibe_level"],
                    exc,
                    exc_info=True,
                )
                return {
                    **row,
                    "vibe_model": model,
                    "config_model": config_model,
                    "config_payload": None,
                    "config_error": str(exc),
                }

        tasks = [asyncio.create_task(_attach_config(row)) for row in rows]
        updated_rows = await asyncio.gather(*tasks) if tasks else []
        record_error = None
        if any(row.get("config_error") for row in updated_rows):
            record_error = "config_error"
        return RecordResult(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split=split_name,
            rows=updated_rows,
            error=record_error,
        )
    except Exception as exc:
        _LOGGER.warning("Failed processing record %s: %s", record_id, exc, exc_info=True)
        error_row = {
            "dataset": record.dataset,
            "id_in_dataset": record.id_in_dataset,
            "split": split_name,
            "error": str(exc),
        }
        return RecordResult(
            dataset=record.dataset,
            id_in_dataset=record.id_in_dataset,
            split=split_name,
            rows=[error_row],
            error=str(exc),
        )


async def process_split(
    *,
    split_name: str,
    records: Sequence[BaseRecord],
    dataset_hash: str,
    system_prompt: str,
    config_prompt: str,
    cache: SqliteCache,
    model: str,
    api_key: str | None,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    config_model: str,
    config_api_base: str | None,
    config_model_kwargs: Mapping[str, Any],
    max_input_tokens: int,
    page_tokens: int,
    max_concurrency: int,
    max_qps: float,
    max_retries: int,
    retry_base_delay: float,
    config_max_concurrency: int,
    config_max_qps: float,
    config_max_retries: int,
    config_retry_base_delay: float,
    config_batch_size: int,
    config_batch_wait_ms: int,
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
    config_semaphore = asyncio.Semaphore(config_max_concurrency)
    config_limiter = AsyncRateLimiter(config_max_qps)
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
            model_kwargs=model_kwargs,
        )

    @retry_async(
        max_retries=config_max_retries,
        base_delay=config_retry_base_delay,
        logger=_LOGGER,
    )
    @rate_limited(config_limiter)
    @with_semaphore(config_semaphore)
    async def single_config_call(vibe_text: str) -> MusicConfigPromptPayload:
        return await call_music_config(
            vibe_text=vibe_text,
            model=config_model,
            api_key=api_key,
            api_base=config_api_base,
            system_prompt=config_prompt,
            model_kwargs=config_model_kwargs,
        )

    @retry_async(
        max_retries=config_max_retries,
        base_delay=config_retry_base_delay,
        logger=_LOGGER,
    )
    @rate_limited(config_limiter)
    @with_semaphore(config_semaphore)
    async def batched_config_call(vibe_texts: Sequence[str]) -> list[MusicConfigPromptPayload]:
        return await call_music_config_batch(
            vibe_texts=vibe_texts,
            model=config_model,
            api_key=api_key,
            api_base=config_api_base,
            system_prompt=config_prompt,
            model_kwargs=config_model_kwargs,
        )

    batcher: ConfigBatcher | None = None
    if config_batch_size > 1:
        batcher = ConfigBatcher(
            batch_size=config_batch_size,
            max_wait=config_batch_wait_ms / 1000.0,
            call_batch=batched_config_call,
        )

    @cached_async(
        cache=cache,
        key_fn=lambda vibe_text, cache_key, payload: (cache_key, payload),
        serializer=lambda response: response.model_dump(),
        deserializer=lambda data: MusicConfigPromptPayload.model_validate(data),
    )
    async def cached_config_call(
        vibe_text: str,
        cache_key: str,
        payload: dict[str, Any],
    ) -> MusicConfigPromptPayload:
        if batcher is None:
            return await single_config_call(vibe_text)
        try:
            return await batcher.submit(vibe_text)
        except Exception as exc:
            _LOGGER.warning(
                "Batch config call failed; falling back to single-item call. (%s)",
                exc,
                exc_info=True,
            )
            return await single_config_call(vibe_text)

    async def bound_process(index: int, record: BaseRecord) -> tuple[int, RecordResult]:
        result = await process_record(
            record=record,
            split_name=split_name,
            dataset_hash=dataset_hash,
            system_prompt=system_prompt,
            config_prompt=config_prompt,
            llm_call=cached_call,
            config_call=cached_config_call,
            error_rate=error_rate,
            error_seed=error_seed,
            noise_lock=noise_lock,
            model=model,
            api_base=api_base,
            model_kwargs=model_kwargs,
            config_model=config_model,
            config_api_base=config_api_base,
            config_model_kwargs=config_model_kwargs,
            max_input_tokens=max_input_tokens,
            page_tokens=page_tokens,
            seed=seed,
        )
        return index, result

    try:
        tasks = [
            asyncio.create_task(bound_process(index, record))
            for index, record in enumerate(pending_records)
        ]

        results: dict[int, RecordResult] = {}
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

        good_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        for index in sorted(results):
            result = results[index]
            for row in result.rows:
                if row.get("error") or row.get("config_error"):
                    error_rows.append(row)
                else:
                    good_rows.append(row)
    finally:
        if batcher is not None:
            await batcher.aclose()

    # NOTE: Dedupe now happens globally BEFORE split (in main_async) to prevent
    # train/test leakage. Per-split dedupe removed as of 2026-01-18.
    final_rows = good_rows

    if error_rate > 0 and final_rows:
        noisy_rows = sum(1 for row in final_rows if _row_is_noisy(row))
        if noisy_rows == 0:
            _LOGGER.warning(
                "Split %s produced no noisy rows; forcing noise on one row.",
                split_name,
            )
            _force_noise_row(
                final_rows,
                error_seed=error_seed,
                dataset_hash=dataset_hash,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in final_rows + error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if args.source_dir is not None:
        run_config = build_run_config(args)
        config_hash = build_config_hash(run_config)
        _LOGGER.info("Using configuration hash: %s", config_hash)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config_file(output_dir / RUN_CONFIG_NAME, run_config)
        transform_processed_splits(
            source_dir=args.source_dir,
            output_dir=output_dir,
            input_dir=args.input_dir,
            only_splits=args.only_splits,
            filter_vibe_scopes=args.filter_vibe_scopes,
            filter_vibe_levels=args.filter_vibe_levels,
            raw_text_direct=args.raw_text_direct,
            overwrite=args.overwrite,
        )
        return

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    load_env_file(args.env_file)
    model, api_base = normalize_model_and_base(args.model, args.api_base)
    config_model, config_api_base = normalize_model_and_base(
        args.config_model, args.config_api_base
    )
    args.model = model
    args.api_base = api_base
    args.config_model = config_model
    args.config_api_base = config_api_base

    warn_if_model_overridden(
        model=model,
        default_model=DEFAULT_MODEL,
        max_input_tokens=args.max_input_tokens,
        label="Vibe extraction",
    )
    if config_model != DEFAULT_CONFIG_MODEL:
        _LOGGER.warning(
            "Config generation model overridden from %s to %s. "
            "If the new model's context differs, review --max-input-tokens.",
            DEFAULT_CONFIG_MODEL,
            config_model,
        )
    warn_if_context_limit(model, args.max_input_tokens)
    api_key = resolve_api_key_for_models(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        models=[(model, api_base), (config_model, config_api_base)],
    )

    run_config = build_run_config(args)
    config_hash = build_config_hash(run_config)
    _LOGGER.info("Using configuration hash: %s", config_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config_file(output_dir / RUN_CONFIG_NAME, run_config)

    validate_schema(jsonl_paths)
    records = load_records(jsonl_paths)
    dataset_hash = hash_paths(jsonl_paths)

    # Global dedupe BEFORE split to prevent train/test leakage
    original_count = len(records)
    records, removed_count = dedupe_records(
        records,
        threshold=args.dedupe_threshold,
        model_name=args.dedupe_model,
    )
    _LOGGER.info(
        "Global dedupe: kept=%d removed=%d (threshold=%.2f)",
        len(records),
        removed_count,
        args.dedupe_threshold,
    )
    if removed_count > 0:
        _LOGGER.info(
            "Dedupe reduction: %.1f%% of records removed",
            100.0 * removed_count / original_count,
        )

    # Use diversity sampling for GRPO split to maximize coverage
    def _diversity_fn(recs: list[BaseRecord], n: int) -> tuple[list[BaseRecord], list[int]]:
        return diversity_sample(
            recs,
            n,
            model_name=args.dedupe_model,
            seed=args.seed,
        )

    split_map = split_records_with_diversity(
        records,
        args.seed,
        diversity_sample_fn=_diversity_fn,
    )
    target_splits = args.only_splits or [name for name, _ in SPLITS]
    system_prompt = build_vibe_prompt()
    config_prompt = build_config_generation_prompt(batch=args.config_batch_size > 1)
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
            config_prompt=config_prompt,
            cache=cache,
            model=model,
            api_key=api_key,
            api_base=api_base,
            model_kwargs=args.model_kwargs,
            config_model=config_model,
            config_api_base=config_api_base,
            config_model_kwargs=args.config_model_kwargs,
            max_input_tokens=args.max_input_tokens,
            page_tokens=args.page_tokens,
            max_concurrency=args.max_concurrency,
            max_qps=args.max_qps,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            config_max_concurrency=args.config_max_concurrency,
            config_max_qps=args.config_max_qps,
            config_max_retries=args.config_max_retries,
            config_retry_base_delay=args.config_retry_base_delay,
            config_batch_size=args.config_batch_size,
            config_batch_wait_ms=args.config_batch_wait_ms,
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

    _LOGGER.warning(
        "Seed=%d provides partial reproducibility. External LLM calls (vibe extraction, "
        "config generation) are non-deterministic due to API load balancing and model "
        "variability. Seed controls: split assignment, noise injection, diversity sampling. "
        "SQLite cache ensures re-runs with same inputs produce identical outputs.",
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
