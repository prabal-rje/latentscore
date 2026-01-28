"""Modal-driven SFT inference + optional CLAP scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from common.music_schema import MusicConfigPromptPayload, repair_palette_duplicates
from common.prompt_registry import render_config_prompt
from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import (
    normalize_tokenizer_for_model,
    render_chat_prompt,
    wrap_vibe_for_chat,
)

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover
    modal = None

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger(__name__)

APP_NAME = "latentscore-modal-infer"
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "H100")
GPU_COUNT = int(os.environ.get("MODAL_GPU_COUNT", "1"))
if GPU_COUNT > 1 and ":" not in GPU_TYPE:
    GPU_TYPE = f"{GPU_TYPE}:{GPU_COUNT}"
TIMEOUT_HOURS = 4
MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
REMOTE_REPO_PATH = "/repo"
REMOTE_OUTPUT_PATH = "/outputs"
VOLUME_NAME = "latentscore-training-outputs"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_VERSION = "config_v1"
DEFAULT_PROMPT_FIELD = "vibe_noisy"
DEFAULT_SCORE_VIBE_FIELD = "vibe_original"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_RETRY_TEMPERATURE = 0.7
DEFAULT_RETRY_TOP_P = 0.95
DEFAULT_DURATION = 12.0
DEFAULT_LOG_EVERY = 25
DEFAULT_BATCH_SIZE = 8
DEFAULT_USE_TQDM = True

PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"

if modal is not None:
    modal.enable_output()


class _ModalStubRun:
    def __enter__(self) -> None:
        raise RuntimeError("Modal is required to run inference commands.")

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> bool:
        return False


class _ModalStubApp:
    def function(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - stub only
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def run(self) -> _ModalStubRun:  # pragma: no cover - stub only
        return _ModalStubRun()


if modal is None:
    RETRY_POLICY = None
    OUTPUTS_VOLUME = None
    INFER_IMAGE = None
    app = _ModalStubApp()
else:
    INFER_IMAGE_PACKAGES = (
        "hf_transfer==0.1.9",
        "huggingface_hub==0.34.4",
        "numpy==2.2.6",
        "peft==0.17.1",
        "pydantic==2.11.7",
        "sentencepiece==0.2.0",
        "soundfile",
        "torchaudio",
        "torchvision",
        "transformers==4.54.1",
        "laion-clap",
        "outlines",
        "tqdm",
        "torch",
    )
    RETRY_POLICY = modal.Retries(
        max_retries=MAX_RETRIES,
        backoff_coefficient=DEFAULT_RETRY_BACKOFF,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY,
    )
    OUTPUTS_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    INFER_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libsndfile1")
        .uv_pip_install(
            *INFER_IMAGE_PACKAGES,
            extra_index_url=PYTORCH_CU128_INDEX,
            extra_options="--index-strategy unsafe-best-match",
        )
        .add_local_python_source("data_work", copy=True)
        .env({"HF_HOME": "/model_cache", "PYTHONPATH": REMOTE_REPO_PATH})
        .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO_PATH, copy=True)
    )
    app = modal.App(APP_NAME)


class InferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str
    adapter_dir: str
    input_file: str
    output_dir: str
    system_prompt: str
    prompt_field: str
    score_vibe_field: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    retry_temperature: float
    retry_top_p: float
    max_retries: int
    limit: int
    split: str | None
    score: bool
    duration: float
    seed: int
    log_first_n: int
    log_every: int
    batch_size: int
    use_tqdm: bool


class InferResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vibe: str
    score_vibe: str
    id_in_dataset: str | int | None = None
    dataset: str | None = None
    split: str | None = None
    config: dict[str, Any] | None = None
    config_error: str | None = None
    clap_reward: float | None = None
    clap_details: dict[str, Any] | None = None


def _resolve_remote_data_file(data_file: Path) -> str:
    try:
        relative = data_file.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit("Input data must live under the repo root.") from exc
    return str(Path(REMOTE_REPO_PATH) / relative)


def _resolve_model_path(model_path: str) -> str:
    if "/" not in model_path:
        return str(Path(REMOTE_OUTPUT_PATH) / model_path)
    path = Path(model_path)
    if path.is_absolute():
        return model_path
    if path.parts and path.parts[0] == "outputs":
        return str(Path(REMOTE_OUTPUT_PATH) / Path(*path.parts[1:]))
    return model_path


def _resolve_volume_path(remote_output: str) -> str:
    output_path = Path(remote_output)
    try:
        relative = output_path.relative_to(REMOTE_OUTPUT_PATH)
    except ValueError as exc:
        raise SystemExit(
            f"Expected output path under {REMOTE_OUTPUT_PATH}: {remote_output}"
        ) from exc
    return str(relative)


def _download_from_volume(remote_output: str, local_destination: Path) -> Path:
    local_destination.mkdir(parents=True, exist_ok=True)
    remote_path = _resolve_volume_path(remote_output)
    download_path = local_destination / remote_path

    # Try to detect directory outputs and download contents explicitly.
    try:
        ls_result = subprocess.run(
            ["modal", "volume", "ls", VOLUME_NAME, remote_path],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = []
        for line in ls_result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("total"):
                continue
            entries.append(line.split()[0])
        if entries:
            if download_path.exists() and download_path.is_file():
                download_path.unlink()
            download_path.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                subprocess.run(
                    [
                        "modal",
                        "volume",
                        "get",
                        "--force",
                        VOLUME_NAME,
                        f"{remote_path}/{entry}",
                        str(download_path / entry),
                    ],
                    check=True,
                )
            return download_path
    except subprocess.CalledProcessError:
        pass

    # Fallback: treat as a single file path.
    download_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            VOLUME_NAME,
            remote_path,
            str(download_path),
        ],
        check=True,
    )
    return download_path


def _iter_rows(path: Path, limit: int, split: str | None) -> Iterable[dict[str, Any]]:
    count = 0
    for row in iter_jsonl(path):
        if split and row.get("split") != split:
            continue
        yield row
        count += 1
        if limit and count >= limit:
            break


def _repair_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_top = {"thinking", "title", "config", "palettes"}
    repaired = {k: v for k, v in payload.items() if k in allowed_top}
    config = repaired.get("config")
    if isinstance(config, dict):
        density = config.get("density")
        if isinstance(density, (int, float)):
            config["density"] = int(max(2, min(6, density)))
        rmin = config.get("register_min_oct")
        rmax = config.get("register_max_oct")
        if isinstance(rmin, (int, float)):
            config["register_min_oct"] = int(max(1, min(8, rmin)))
        if isinstance(rmax, (int, float)):
            config["register_max_oct"] = int(max(1, min(8, rmax)))
        if isinstance(rmin, (int, float)) and isinstance(rmax, (int, float)):
            if config["register_min_oct"] > config["register_max_oct"]:
                config["register_min_oct"], config["register_max_oct"] = (
                    config["register_max_oct"],
                    config["register_min_oct"],
                )
    palettes = repaired.get("palettes")
    if isinstance(palettes, list):
        repaired["palettes"] = repair_palette_duplicates(palettes)
    return repaired


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


@app.function(
    image=INFER_IMAGE,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=RETRY_POLICY,
    volumes={REMOTE_OUTPUT_PATH: OUTPUTS_VOLUME} if OUTPUTS_VOLUME else None,
)
def run_sft_infer(config: dict[str, Any]) -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.setLevel(logging.INFO)
    import random
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import outlines
    from tqdm import tqdm

    from data_work.lib.clap_scorer import ClapScorer, score_config

    cfg = InferConfig.model_validate(config)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading base model on %s...", device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    normalize_tokenizer_for_model(tokenizer, cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model = PeftModel.from_pretrained(model, cfg.adapter_dir)
    model.to(device)
    model.eval()

    outlines_model = outlines.from_transformers(model, tokenizer)

    scorer = ClapScorer() if cfg.score else None

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    total = 0
    valid = 0
    clap_scores: list[float] = []
    progress = tqdm(
        total=cfg.limit if cfg.limit else None,
        unit="row",
        disable=not cfg.use_tqdm,
    )

    def _coerce_result(result: Any) -> tuple[dict[str, Any] | None, str | None, str]:
        if isinstance(result, BaseModel):
            return result.model_dump(), None, result.model_dump_json()
        if isinstance(result, dict):
            return result, None, json.dumps(result)
        text = str(result)
        parsed = _parse_json_payload(text)
        if parsed is None:
            return None, "json_parse_error", text
        return parsed, None, text

    def _process_batch(
        batch_items: list[dict[str, Any]],
        handle: Any,
    ) -> None:
        nonlocal total, valid
        pending = list(batch_items)

        for attempt in range(cfg.max_retries + 1):
            if not pending:
                break

            if attempt == 0:
                do_sample = cfg.do_sample
                temperature = cfg.temperature
                top_p = cfg.top_p
            else:
                do_sample = True
                temperature = cfg.retry_temperature
                top_p = cfg.retry_top_p

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            prompts = [item["prompt"] for item in pending]
            with torch.no_grad():
                results = outlines_model.batch(
                    prompts,
                    output_type=MusicConfigPromptPayload,
                    **gen_kwargs,
                )

            next_pending: list[dict[str, Any]] = []
            for item, result in zip(pending, results):
                parsed, error, text = _coerce_result(result)
                item["last_text"] = text
                item["error"] = error
                if parsed is None:
                    next_pending.append(item)
                    continue

                parsed = _repair_payload(parsed)
                try:
                    MusicConfigPromptPayload.model_validate(parsed)
                    item["payload"] = parsed
                    item["error"] = None
                except ValidationError as exc:
                    item["payload"] = None
                    item["error"] = f"schema_validation_error: {exc}"
                    next_pending.append(item)

            pending = next_pending

        for item in batch_items:
            payload = item.get("payload")
            error = item.get("error")
            if payload:
                valid += 1

            if cfg.log_first_n > 0 and item["row_idx"] <= cfg.log_first_n:
                snippet = (item.get("last_text") or "").strip()
                LOGGER.info(
                    "INFER_DEBUG row=%s error=%s output=%s",
                    item["row_idx"],
                    error,
                    snippet[:2000],
                )

            clap_reward = None
            clap_details = None
            if payload and scorer is not None:
                try:
                    clap = score_config(
                        vibe=item["score_vibe"],
                        config=payload,
                        scorer=scorer,
                        duration=cfg.duration,
                    )
                    clap_reward = clap.final_score
                    clap_details = clap.model_dump()
                    clap_scores.append(clap_reward)
                except Exception as exc:  # pragma: no cover - runtime scoring
                    error = f"clap_error: {exc}"

            result = InferResult(
                vibe=item["vibe"],
                score_vibe=item["score_vibe"],
                id_in_dataset=item.get("id_in_dataset"),
                dataset=item.get("dataset"),
                split=item.get("split"),
                config=payload,
                config_error=error,
                clap_reward=clap_reward,
                clap_details=clap_details,
            )
            handle.write(result.model_dump_json() + "\n")
            progress.update(1)

            if cfg.log_every > 0 and item["row_idx"] % cfg.log_every == 0:
                LOGGER.info(
                    "INFER_PROGRESS rows=%s valid=%s valid_rate=%.3f",
                    item["row_idx"],
                    valid,
                    (valid / item["row_idx"]) if item["row_idx"] else 0.0,
                )

    with results_path.open("w", encoding="utf-8") as handle:
        batch: list[dict[str, Any]] = []
        row_idx = 0
        for row in _iter_rows(Path(cfg.input_file), cfg.limit, cfg.split):
            vibe = row.get(cfg.prompt_field)
            if not vibe:
                continue
            row_idx += 1
            score_vibe = row.get(cfg.score_vibe_field) or vibe
            total += 1

            user_prompt = wrap_vibe_for_chat(vibe)
            prompt = render_chat_prompt(
                system_prompt=cfg.system_prompt,
                user_prompt=user_prompt,
                tokenizer=tokenizer,
                model_name=cfg.base_model,
                add_generation_prompt=True,
            )

            batch.append(
                {
                    "row_idx": row_idx,
                    "vibe": vibe,
                    "score_vibe": score_vibe,
                    "id_in_dataset": row.get("id_in_dataset"),
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "prompt": prompt,
                    "payload": None,
                    "error": None,
                    "last_text": None,
                }
            )

            if len(batch) >= cfg.batch_size:
                _process_batch(batch, handle)
                batch = []

        if batch:
            _process_batch(batch, handle)

    summary = {
        "total": total,
        "valid": valid,
        "valid_rate": (valid / total) if total else 0.0,
        "mean_clap_reward": (sum(clap_scores) / len(clap_scores)) if clap_scores else None,
    }
    progress.close()
    summary_path.write_text(json.dumps(summary, indent=2))
    return str(output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SFT inference + optional CLAP scoring on Modal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_work/2026-01-26_scored/SFT-Val.jsonl"),
        help="JSONL dataset to evaluate.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter directory name in Modal volume (e.g., prod-sft-gemma3-270m-v5).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help="Base model HF path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder name under /outputs (default: timestamped).",
    )
    parser.add_argument(
        "--prompt-field",
        type=str,
        default=DEFAULT_PROMPT_FIELD,
        help="Field to use as prompt input.",
    )
    parser.add_argument(
        "--score-vibe-field",
        type=str,
        default=DEFAULT_SCORE_VIBE_FIELD,
        help="Field to use for CLAP scoring text.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--retry-temperature",
        type=float,
        default=DEFAULT_RETRY_TEMPERATURE,
        help="Temperature used on retries after invalid output.",
    )
    parser.add_argument(
        "--retry-top-p",
        type=float,
        default=DEFAULT_RETRY_TOP_P,
        help="Top-p used on retries after invalid output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count after invalid output.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Compute CLAP scores for valid outputs.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Audio duration for CLAP scoring.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--log-first-n",
        type=int,
        default=0,
        help="Log raw outputs for the first N rows (0 = disabled).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Log progress every N rows (0 = disabled).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for outlines generation.",
    )
    parser.add_argument(
        "--tqdm",
        dest="use_tqdm",
        action="store_true",
        default=DEFAULT_USE_TQDM,
        help="Enable tqdm progress bar logging.",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar logging.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt registry version for system prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system prompt text.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data_work/.modal_outputs"),
        help="Local directory to download outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    input_file = args.input.expanduser().resolve()
    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")

    system_prompt = render_config_prompt(args.prompt_version)
    if args.system_prompt is not None:
        system_prompt = args.system_prompt

    output_name = args.output
    if not output_name:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_name = f"sft-infer-{args.adapter}-{ts}"

    remote_output_dir = str(Path(REMOTE_OUTPUT_PATH) / output_name)
    remote_input_file = _resolve_remote_data_file(input_file)

    config = InferConfig(
        base_model=args.base_model,
        adapter_dir=_resolve_model_path(args.adapter),
        input_file=str(remote_input_file),
        output_dir=remote_output_dir,
        system_prompt=system_prompt,
        prompt_field=args.prompt_field,
        score_vibe_field=args.score_vibe_field,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        retry_temperature=args.retry_temperature,
        retry_top_p=args.retry_top_p,
        max_retries=args.max_retries,
        limit=args.limit,
        split=args.split,
        score=args.score,
        duration=args.duration,
        seed=args.seed,
        log_first_n=args.log_first_n,
        log_every=args.log_every,
        batch_size=args.batch_size,
        use_tqdm=args.use_tqdm,
    )

    with app.run():
        result = run_sft_infer.remote(config.model_dump())

    downloaded = _download_from_volume(result, args.download_dir)
    LOGGER.info("Modal job completed. Output stored at %s.", result)
    LOGGER.info("Downloaded output to %s.", downloaded)


if __name__ == "__main__":
    main()
