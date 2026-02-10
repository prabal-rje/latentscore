"""Performance benchmark for vibe-to-music tiers (latency + real-time factor).

This script measures:
- config generation time (prompt -> config)
- synth render time for first chunk (time to first audio)
- synth render time for full clip

It is intentionally CLAP-free: CLAP inference is not part of the live interface.
Use `04_clap_benchmark` for alignment scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Coroutine, Iterable, Mapping, Sequence, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

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
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble

LOGGER = logging.getLogger("data_work.perf_benchmark")

DEFAULT_SYSTEM_PROMPT = build_music_prompt()
DEFAULT_VIBE_FIELD = "vibe_original"
DEFAULT_OUTPUT_DIR = Path("data_work/.experiments/perf_benchmark")

AudioArray = NDArray[np.float64]
T = TypeVar("T")


class PerfSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str  # "litellm" | "local" | "baseline"
    label: str
    model: str | None = None
    api_base: str | None = None


class PerfResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    vibe: str
    source_label: str
    source_kind: str

    config_error: str | None = None
    synth_error: str | None = None

    config_time_ms: float | None = None
    synth_first_chunk_ms: float | None = None
    synth_full_ms: float | None = None

    total_first_chunk_ms: float | None = None
    total_full_ms: float | None = None

    rtf_synth_only: float | None = None
    rtf_total: float | None = None


def parse_source_entries(values: Sequence[str], kind: str) -> list[PerfSource]:
    sources: list[PerfSource] = []
    for value in values:
        name, _, label = value.partition(":")
        clean_label = label or name
        if not name:
            raise SystemExit(f"Invalid {kind} entry: {value}")
        sources.append(PerfSource(kind=kind, label=clean_label, model=name))
    return sources


def _stable_prompt_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return (base_seed + value) % (2**32)


def _parse_config_to_internal(config_payload: Mapping[str, Any]) -> Any:
    # Mirror clap_scorer behavior: prompt schema first, then numeric.
    try:
        prompt = MusicConfigPrompt.model_validate(config_payload)
        return prompt.to_config().to_internal()
    except Exception:
        return MusicConfig.model_validate(config_payload).to_internal()


def _render_audio(
    internal_config: Any,
    *,
    duration: float,
    seed: int,
) -> AudioArray:
    rng = np.random.default_rng(seed)
    return assemble(internal_config, duration=duration, rng=rng)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    return {
        "count": float(len(sorted_vals)),
        "mean": float(statistics.mean(sorted_vals)),
        "median": float(statistics.median(sorted_vals)),
        "p95": float(sorted_vals[int(0.95 * (len(sorted_vals) - 1))]),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark tier latency/RTF on a prompt set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with vibes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vibe-field", type=str, default=DEFAULT_VIBE_FIELD)
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all).")

    parser.add_argument("--duration", type=float, default=8.0, help="Full clip duration.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.0,
        help="Chunk duration for time-to-first-audio measurement.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base seed for deterministic audio.")

    parser.add_argument("--baseline", action="append", default=[], help="Baseline name[:label].")
    parser.add_argument("--local-model", action="append", default=[], help="Model path[:label].")
    parser.add_argument("--litellm-model", action="append", default=[], help="LiteLLM model[:label].")

    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--env-file", type=Path, default=None, help="Optional .env file to load.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override.")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--api-base", type=str, default=None, help="Optional API base override.")
    parser.add_argument("--model-kwargs", type=str, default="{}", help="JSON dict of LiteLLM kwargs.")

    parser.add_argument("--local-device", type=str, default=None, help="Device for local HF models.")
    parser.add_argument("--local-max-new-tokens", type=int, default=3000)
    parser.add_argument("--local-temperature", type=float, default=0.0)
    parser.add_argument("--local-force-cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--local-4bit", action="store_true", help="4-bit NF4 quantization (CUDA).")

    return parser


def _iter_prompt_rows(path: Path, *, limit: int) -> Iterable[tuple[str, dict[str, Any]]]:
    count = 0
    for row in iter_jsonl(path):
        count += 1
        if limit and count > limit:
            break
        record_id = str(row.get("id") or row.get("id_in_dataset") or count)
        yield record_id, row


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "perf_results.jsonl"
    summary_path = output_dir / "perf_summary.json"

    load_env_file(args.env_file)

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit("--model-kwargs must be valid JSON.") from exc
    if not isinstance(model_kwargs, dict):
        raise SystemExit("--model-kwargs must be a JSON object.")
    model_kwargs = cast(dict[str, Any], model_kwargs)

    baseline_sources = parse_source_entries(args.baseline, "baseline")
    litellm_sources = parse_source_entries(args.litellm_model, "litellm")
    local_sources = parse_source_entries(args.local_model, "local")
    sources = [*baseline_sources, *litellm_sources, *local_sources]
    if not sources:
        raise SystemExit("At least one --baseline, --local-model, or --litellm-model is required.")

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
            force_cpu=args.local_force_cpu,
            quantize_4bit=args.local_4bit,
        )

    baseline_clients: dict[str, Any] = {}
    if baseline_sources:
        from data_work.lib.baselines import get_baseline

        for source in baseline_sources:
            assert source.model is not None
            baseline_clients[source.label] = get_baseline(source.model)

    results: list[PerfResult] = []

    for record_id, row in _iter_prompt_rows(input_path, limit=args.limit):
        vibe = row.get(args.vibe_field)
        if not isinstance(vibe, str) or not vibe.strip():
            continue
        vibe = vibe.strip()

        for source in sources:
            config_payload: dict[str, Any] | None = None
            config_error: str | None = None
            synth_error: str | None = None

            config_time_ms: float | None = None
            synth_first_ms: float | None = None
            synth_full_ms: float | None = None

            t0 = time.perf_counter()
            try:
                match source.kind:
                    case "baseline":
                        baseline = baseline_clients[source.label]
                        payload = baseline.generate(vibe)
                        # Convert to raw dict for consistent downstream parsing.
                        config_payload = payload.config.model_dump()
                    case "local":
                        client = local_clients[source.label]
                        prompt = client.format_chat_prompt(
                            system_prompt=args.system_prompt,
                            user_prompt=wrap_vibe_for_chat(vibe),
                        )
                        payload = client.generate_structured(prompt, MusicConfigPromptPayload)
                        config_payload = payload.config.model_dump()
                    case "litellm":
                        assert source.model is not None
                        messages = [
                            {"role": "system", "content": args.system_prompt},
                            {"role": "user", "content": wrap_vibe_for_chat(vibe)},
                        ]
                        payload = asyncio_run(
                            litellm_structured_completion(
                                model=source.model,
                                messages=messages,
                                response_model=MusicConfigPromptPayload,
                                api_key=api_key,
                                api_base=source.api_base,
                                model_kwargs=model_kwargs,
                            )
                        )
                        config_payload = payload.config.model_dump()
                    case _:
                        raise SystemExit(f"Unsupported source kind: {source.kind}")
            except Exception as exc:
                config_error = f"{type(exc).__name__}: {exc}"
            finally:
                config_time_ms = (time.perf_counter() - t0) * 1000

            if config_payload is None or config_error is not None:
                results.append(
                    PerfResult(
                        id=record_id,
                        vibe=vibe,
                        source_label=source.label,
                        source_kind=source.kind,
                        config_error=config_error or "missing_config",
                        config_time_ms=config_time_ms,
                    )
                )
                continue

            seed = _stable_prompt_seed(args.seed, f"{record_id}:{vibe}")

            try:
                internal = _parse_config_to_internal(config_payload)

                t_first = time.perf_counter()
                _ = _render_audio(internal, duration=args.chunk_seconds, seed=seed)
                synth_first_ms = (time.perf_counter() - t_first) * 1000

                t_full = time.perf_counter()
                _ = _render_audio(internal, duration=args.duration, seed=seed)
                synth_full_ms = (time.perf_counter() - t_full) * 1000
            except Exception as exc:
                synth_error = f"{type(exc).__name__}: {exc}"

            total_first = None
            total_full = None
            rtf_synth = None
            rtf_total = None
            if synth_error is None:
                assert synth_first_ms is not None
                assert synth_full_ms is not None
                total_first = config_time_ms + synth_first_ms
                total_full = config_time_ms + synth_full_ms
                rtf_synth = (synth_full_ms / 1000.0) / float(args.duration)
                rtf_total = (total_full / 1000.0) / float(args.duration)

            results.append(
                PerfResult(
                    id=record_id,
                    vibe=vibe,
                    source_label=source.label,
                    source_kind=source.kind,
                    config_error=config_error,
                    synth_error=synth_error,
                    config_time_ms=config_time_ms,
                    synth_first_chunk_ms=synth_first_ms,
                    synth_full_ms=synth_full_ms,
                    total_first_chunk_ms=total_first,
                    total_full_ms=total_full,
                    rtf_synth_only=rtf_synth,
                    rtf_total=rtf_total,
                )
            )

    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(row.model_dump_json() + "\n")

    # Aggregate summary per source label
    by_source: dict[str, list[PerfResult]] = {}
    for row in results:
        by_source.setdefault(row.source_label, []).append(row)

    summary: dict[str, Any] = {
        "input": str(input_path),
        "vibe_field": args.vibe_field,
        "duration": args.duration,
        "chunk_seconds": args.chunk_seconds,
        "seed": args.seed,
        "sources": [s.model_dump() for s in sources],
        "per_source": {},
    }

    for label, rows in by_source.items():
        config_ms = [r.config_time_ms for r in rows if r.config_time_ms is not None and r.config_error is None]
        first_ms = [r.total_first_chunk_ms for r in rows if r.total_first_chunk_ms is not None and r.synth_error is None and r.config_error is None]
        full_ms = [r.total_full_ms for r in rows if r.total_full_ms is not None and r.synth_error is None and r.config_error is None]
        rtf_synth = [r.rtf_synth_only for r in rows if r.rtf_synth_only is not None]
        rtf_total = [r.rtf_total for r in rows if r.rtf_total is not None]
        summary["per_source"][label] = {
            "n_total": len(rows),
            "n_ok": sum(1 for r in rows if r.config_error is None and r.synth_error is None),
            "config_time_ms": _summarize([float(x) for x in config_ms]),
            "total_first_chunk_ms": _summarize([float(x) for x in first_ms]),
            "total_full_ms": _summarize([float(x) for x in full_ms]),
            "rtf_synth_only": _summarize([float(x) for x in rtf_synth]),
            "rtf_total": _summarize([float(x) for x in rtf_total]),
        }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote results: %s", results_path)
    LOGGER.info("Wrote summary: %s", summary_path)


def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    main()
