"""Modal-driven SFT + GRPO training for tiny LLMs."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import modal
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from data_work.lib.rewards import RewardBreakdown

REMOTE_REPO_PATH = "/repo"

if REMOTE_REPO_PATH not in sys.path:
    sys.path.append(REMOTE_REPO_PATH)

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger(__name__)

DEFAULT_MODAL_BUILD_VALIDATION = "ignore"
os.environ.setdefault("MODAL_BUILD_VALIDATION", DEFAULT_MODAL_BUILD_VALIDATION)

APP_NAME = "latentscore-modal-train"
GPU_TYPE = "L40S"
TIMEOUT_HOURS = 6
MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
RETRY_POLICY = modal.Retries(
    max_retries=MAX_RETRIES,
    backoff_coefficient=DEFAULT_RETRY_BACKOFF,
    initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
    max_delay=DEFAULT_RETRY_MAX_DELAY,
)
REMOTE_OUTPUT_PATH = "/outputs"
VOLUME_NAME = "latentscore-training-outputs"
OUTPUTS_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYSTEM_PROMPT = (
    "You are a synthesizer configuration assistant. "
    "Given a vibe description, output a JSON configuration for an "
    "ambient/electronic synthesizer. Output ONLY valid JSON with no explanation."
)
DEFAULT_DATASET_TEXT_FIELD = "text"
DEFAULT_PROMPT_FIELD = "vibe_noisy"
DEFAULT_RESPONSE_FIELD = "config_payload"
DEFAULT_MAX_SEQ_LEN = 512

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_BIAS = "none"
DEFAULT_OPTIM = "adamw_8bit"
DEFAULT_LR = 2e-4
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACCUM = 1
DEFAULT_GRPO_NUM_GENERATIONS = 4
DEFAULT_GRPO_BETA = 0.04

BASE_MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "gemma3-270m": "unsloth/gemma-3-270m-it",
    "qwen3-600m": "unsloth/Qwen3-600M",
}

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "pydantic==2.7.4",
        "wandb==0.21.0",
        "weave==0.50.0",
    )
    .add_local_python_source("data_work", copy=True)
    .env({"HF_HOME": "/model_cache", "PYTHONPATH": REMOTE_REPO_PATH})
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO_PATH, copy=True)
)

app = modal.App(APP_NAME)


class SftConfig(BaseModel):
    """SFT configuration for modal training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    data_file: str
    output_dir: str
    system_prompt: str
    prompt_field: str
    response_field: str
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    optim: str
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    epochs: int
    seed: int
    overwrite: bool
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None


class GrpoConfig(BaseModel):
    """GRPO configuration for modal training."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    data_file: str
    output_dir: str
    system_prompt: str
    prompt_field: str
    response_field: str
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    optim: str
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    epochs: int
    num_generations: int
    beta: float
    reward_type: str
    audio_reward: str | None
    seed: int
    overwrite: bool
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None


def _format_prompt(system_prompt: str, prompt: str, response: str) -> str:
    return json.dumps(
        {
            "system": system_prompt,
            "user": prompt,
            "assistant": response,
        },
        ensure_ascii=False,
    )


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output path already exists: {path}. Use --overwrite to replace.")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_model(choice: str) -> str:
    return BASE_MODELS.get(choice, choice)


def _parse_audio_scorer(value: str | None) -> Callable[[str, dict[str, Any]], float] | None:
    if value is None:
        return None
    module_name, _, attr = value.partition(":")
    if not module_name or not attr:
        raise SystemExit("--audio-reward must be in module:function format.")
    module = __import__(module_name, fromlist=[attr])
    scorer = getattr(module, attr)
    if not callable(scorer):
        raise SystemExit(f"Audio scorer {value} is not callable.")
    return scorer


def _log_wandb_run(run: Any) -> None:
    if run is None:
        return
    run_id = getattr(run, "id", None)
    run_name = getattr(run, "name", None)
    LOGGER.info("W&B run: %s (%s)", run_name, run_id)
    print(f"W&B run: {run_name} ({run_id})")


def _resolve_remote_data_file(data_file: Path) -> str:
    try:
        relative = data_file.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit("Training data must live under the repo root.") from exc
    return str(Path(REMOTE_REPO_PATH) / relative)


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
    if download_path.is_dir():
        has_files = any(download_path.rglob("*"))
    else:
        has_files = download_path.exists()
    if not has_files:
        LOGGER.warning(
            "Downloaded output appears empty at %s. "
            "Verify the Modal volume path with `modal volume ls %s %s`.",
            download_path,
            VOLUME_NAME,
            remote_path,
        )
    return download_path


def _delete_from_volume(remote_output: str) -> None:
    remote_path = _resolve_volume_path(remote_output)
    subprocess.run(
        [
            "modal",
            "volume",
            "rm",
            VOLUME_NAME,
            remote_path,
        ],
        check=True,
    )


def _supports_param(callable_obj: Any, name: str) -> bool:
    return name in inspect.signature(callable_obj).parameters


def _set_first_supported(
    callable_obj: Any,
    kwargs: dict[str, Any],
    value: Any,
    *names: str,
) -> bool:
    for name in names:
        if _supports_param(callable_obj, name):
            kwargs[name] = value
            return True
    return False


def _filter_kwargs_for_callable(
    callable_obj: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if _supports_param(callable_obj, key)}


@app.function(image=TRAIN_IMAGE, timeout=60, retries=0)
def check_imports() -> dict[str, Any]:
    repo_exists = Path(REMOTE_REPO_PATH).exists()
    repo_entries = sorted(Path(REMOTE_REPO_PATH).iterdir()) if repo_exists else []
    repo_names = [entry.name for entry in repo_entries]
    data_work_error = None
    data_work_path = None
    try:
        import data_work  # type: ignore
    except Exception as exc:  # pragma: no cover - debugging helper
        data_work_error = repr(exc)
    else:
        data_work_path = getattr(data_work, "__file__", None)
    result = {
        "cwd": str(Path.cwd()),
        "sys_path": list(sys.path),
        "repo_exists": repo_exists,
        "repo_entries": repo_names,
        "data_work_path": data_work_path,
        "data_work_error": data_work_error,
    }
    print(json.dumps(result, indent=2))
    return result


def _create_reward_fn(
    reward_type: str,
    audio_scorer: Callable[[str, dict[str, Any]], float] | None,
    wandb_run: Any,
) -> Callable[[Sequence[str], Sequence[str]], list[float]]:
    from data_work.lib.rewards import compute_partial_reward

    def reward_fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        breakdowns: list[RewardBreakdown] = []
        for prompt, completion in zip(prompts, completions):
            vibe = prompt
            parsed_prompt = _parse_prompt_json(prompt)
            if parsed_prompt is not None:
                vibe = parsed_prompt.get("user", prompt)
            breakdown = compute_partial_reward(
                vibe=vibe,
                output=completion,
                audio_scorer=audio_scorer,
            )
            breakdowns.append(breakdown)
            rewards.append(breakdown.total)
        if wandb_run is not None:
            if breakdowns:
                wandb_run.log(
                    {
                        "reward/format_mean": sum(b.format for b in breakdowns) / len(breakdowns),
                        "reward/schema_mean": sum(b.schema for b in breakdowns) / len(breakdowns),
                        "reward/audio_mean": sum(b.audio for b in breakdowns) / len(breakdowns),
                        "reward/total_mean": sum(b.total for b in breakdowns) / len(breakdowns),
                    }
                )
        return rewards

    match reward_type:
        case "clap":
            if audio_scorer is None:
                LOGGER.warning(
                    "Reward type clap requested, but no audio scorer provided. "
                    "Falling back to format+schema partial reward."
                )
            return reward_fn
        case "schema_only":
            return reward_fn
        case _:
            raise SystemExit(f"Unsupported reward type: {reward_type}")


def _parse_prompt_json(prompt: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_HOURS * 3600,
    retries=RETRY_POLICY,
    volumes={REMOTE_OUTPUT_PATH: OUTPUTS_VOLUME},
)
def run_sft(config: dict[str, Any]) -> str:
    # isort: off
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    # isort: on

    import datasets
    import torch
    import wandb
    import weave
    from transformers import TrainingArguments
    from trl import SFTTrainer

    sft = SftConfig.model_validate(config)
    _ensure_output_dir(Path(sft.output_dir), sft.overwrite)

    wandb_run = None
    if sft.wandb_project:
        wandb_run = wandb.init(
            project=sft.wandb_project,
            entity=sft.wandb_entity,
            name=sft.wandb_run_name,
            config=sft.model_dump(),
        )
        weave.init(sft.wandb_project)
        _log_wandb_run(wandb_run)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft.base_model,
        max_seq_length=sft.max_seq_length,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=sft.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=sft.lora_alpha,
        lora_dropout=sft.lora_dropout,
        bias=sft.lora_bias,
        use_gradient_checkpointing="unsloth",
    )

    dataset = datasets.load_dataset("json", data_files=sft.data_file, split="train")

    def has_prompt(example: dict[str, Any]) -> bool:
        return example.get(sft.prompt_field) not in (None, "")

    def has_response(example: dict[str, Any]) -> bool:
        return example.get(sft.response_field) not in (None, "")

    dataset = dataset.filter(has_prompt)
    dataset = dataset.filter(has_response)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        prompt = str(example[sft.prompt_field])
        response = json.dumps(example[sft.response_field], ensure_ascii=False)
        text = _format_prompt(sft.system_prompt, prompt, response)
        return {DEFAULT_DATASET_TEXT_FIELD: text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=sft.output_dir,
        per_device_train_batch_size=sft.batch_size,
        gradient_accumulation_steps=sft.gradient_accumulation_steps,
        learning_rate=sft.learning_rate,
        lr_scheduler_type=sft.lr_scheduler_type,
        warmup_ratio=sft.warmup_ratio,
        weight_decay=sft.weight_decay,
        optim=sft.optim,
        num_train_epochs=sft.epochs,
        logging_steps=10,
        save_strategy="epoch",
        seed=sft.seed,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to=["wandb"] if wandb_run else [],
    )

    trainer_kwargs: dict[str, Any] = {}
    if not _set_first_supported(SFTTrainer.__init__, trainer_kwargs, model, "model"):
        raise SystemExit("SFTTrainer does not accept a model argument.")
    if not _set_first_supported(
        SFTTrainer.__init__, trainer_kwargs, dataset, "train_dataset", "dataset"
    ):
        raise SystemExit("SFTTrainer does not accept a dataset argument.")
    if not _set_first_supported(
        SFTTrainer.__init__, trainer_kwargs, training_args, "args", "training_args"
    ):
        raise SystemExit("SFTTrainer does not accept training arguments.")
    _set_first_supported(
        SFTTrainer.__init__, trainer_kwargs, tokenizer, "processing_class", "tokenizer"
    )
    if _supports_param(SFTTrainer.__init__, "dataset_text_field"):
        trainer_kwargs["dataset_text_field"] = DEFAULT_DATASET_TEXT_FIELD
    if _supports_param(SFTTrainer.__init__, "max_seq_length"):
        trainer_kwargs["max_seq_length"] = sft.max_seq_length
    if _supports_param(SFTTrainer.__init__, "packing"):
        trainer_kwargs["packing"] = False
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(sft.output_dir)
    tokenizer.save_pretrained(sft.output_dir)
    OUTPUTS_VOLUME.commit()

    if wandb_run:
        wandb_run.finish()

    return sft.output_dir


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_HOURS * 3600,
    retries=RETRY_POLICY,
    volumes={REMOTE_OUTPUT_PATH: OUTPUTS_VOLUME},
)
def run_grpo(config: dict[str, Any]) -> str:
    # isort: off
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    # isort: on

    import datasets
    import torch
    import wandb
    import weave
    from transformers import TrainingArguments
    from trl import GRPOTrainer

    try:
        from trl import GRPOConfig
    except ImportError:
        GRPOConfig = None

    grpo = GrpoConfig.model_validate(config)
    _ensure_output_dir(Path(grpo.output_dir), grpo.overwrite)

    wandb_run = None
    if grpo.wandb_project:
        wandb_run = wandb.init(
            project=grpo.wandb_project,
            entity=grpo.wandb_entity,
            name=grpo.wandb_run_name,
            config=grpo.model_dump(),
        )
        weave.init(grpo.wandb_project)
        _log_wandb_run(wandb_run)

    audio_scorer = _parse_audio_scorer(grpo.audio_reward)
    reward_fn = _create_reward_fn(grpo.reward_type, audio_scorer, wandb_run)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=grpo.model_path,
        max_seq_length=grpo.max_seq_length,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=grpo.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=grpo.lora_alpha,
        lora_dropout=grpo.lora_dropout,
        bias=grpo.lora_bias,
        use_gradient_checkpointing="unsloth",
    )

    dataset = datasets.load_dataset("json", data_files=grpo.data_file, split="train")

    def has_prompt(example: dict[str, Any]) -> bool:
        return example.get(grpo.prompt_field) not in (None, "")

    dataset = dataset.filter(has_prompt)

    def format_prompts(example: dict[str, Any]) -> dict[str, str]:
        prompt = str(example[grpo.prompt_field])
        text = _format_prompt(grpo.system_prompt, prompt, "")
        return {"prompt": text}

    dataset = dataset.map(format_prompts, remove_columns=dataset.column_names)

    training_kwargs = {
        "output_dir": grpo.output_dir,
        "per_device_train_batch_size": grpo.batch_size,
        "gradient_accumulation_steps": grpo.gradient_accumulation_steps,
        "learning_rate": grpo.learning_rate,
        "lr_scheduler_type": grpo.lr_scheduler_type,
        "warmup_ratio": grpo.warmup_ratio,
        "weight_decay": grpo.weight_decay,
        "optim": grpo.optim,
        "num_train_epochs": grpo.epochs,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "seed": grpo.seed,
        "bf16": torch.cuda.is_available(),
        "fp16": not torch.cuda.is_available(),
        "report_to": ["wandb"] if wandb_run else [],
    }
    training_cls = GRPOConfig or TrainingArguments
    training_args = training_cls(**_filter_kwargs_for_callable(training_cls, training_kwargs))
    if not hasattr(training_args, "model_init_kwargs"):
        setattr(training_args, "model_init_kwargs", {})
    if not hasattr(training_args, "ref_model_init_kwargs"):
        setattr(training_args, "ref_model_init_kwargs", {})

    trainer_kwargs: dict[str, Any] = {}
    if not _set_first_supported(GRPOTrainer.__init__, trainer_kwargs, model, "model"):
        raise SystemExit("GRPOTrainer does not accept a model argument.")
    if not _set_first_supported(
        GRPOTrainer.__init__, trainer_kwargs, dataset, "train_dataset", "dataset"
    ):
        raise SystemExit("GRPOTrainer does not accept a dataset argument.")
    if not _set_first_supported(
        GRPOTrainer.__init__, trainer_kwargs, training_args, "args", "training_args"
    ):
        raise SystemExit("GRPOTrainer does not accept training arguments.")
    _set_first_supported(
        GRPOTrainer.__init__,
        trainer_kwargs,
        tokenizer,
        "processing_class",
        "tokenizer",
    )
    if _supports_param(GRPOTrainer.__init__, "reward_funcs"):
        trainer_kwargs["reward_funcs"] = [reward_fn]
    elif _supports_param(GRPOTrainer.__init__, "reward_fn"):
        trainer_kwargs["reward_fn"] = reward_fn
    else:
        raise SystemExit("GRPOTrainer does not accept a reward function argument.")
    if _supports_param(GRPOTrainer.__init__, "num_generations"):
        trainer_kwargs["num_generations"] = grpo.num_generations
    if _supports_param(GRPOTrainer.__init__, "beta"):
        trainer_kwargs["beta"] = grpo.beta
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(grpo.output_dir)
    tokenizer.save_pretrained(grpo.output_dir)
    OUTPUTS_VOLUME.commit()

    if wandb_run:
        wandb_run.finish()

    return grpo.output_dir


def _build_parser(show_advanced: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch SFT or GRPO LoRA training on Modal using Unsloth. "
            "Defaults mirror Modal's Unsloth example, but full-precision model loading "
            "is enforced for small models. Outputs are LoRA adapter weights; merge them "
            "with 05_export_models to produce full-precision checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Show advanced options in -h output.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check-imports",
        help="Run a fast Modal check to validate repo mount + import paths.",
    )

    def add_shared_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--data",
            type=Path,
            required=True,
            help="Path to JSONL training data (local file to mount in Modal).",
        )
        subparser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory name (LoRA adapters stored in Modal volume).",
        )
        subparser.add_argument(
            "--download-dir",
            type=Path,
            default=None,
            help=(
                "Optional local directory to download Modal outputs into. "
                "A subfolder named after the Modal output is created inside it."
            ),
        )
        subparser.add_argument(
            "--delete-remote",
            action="store_true",
            help="Delete the Modal volume output after a successful download.",
        )
        subparser.add_argument(
            "--system-prompt",
            type=str,
            default=DEFAULT_SYSTEM_PROMPT,
            help="System prompt used to format training samples.",
        )
        subparser.add_argument(
            "--prompt-field",
            type=str,
            default=DEFAULT_PROMPT_FIELD,
            help="JSON field containing the prompt/vibe.",
        )
        subparser.add_argument(
            "--response-field",
            type=str,
            default=DEFAULT_RESPONSE_FIELD,
            help="JSON field containing the structured response/config.",
        )
        subparser.add_argument(
            "--max-seq-length",
            type=int,
            default=DEFAULT_MAX_SEQ_LEN,
            help="Maximum sequence length for tokenization.",
        )
        subparser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing output folder in the Modal volume.",
        )
        subparser.add_argument(
            "--wandb-project",
            type=str,
            default=None,
            help="W&B project to log to (omit to disable).",
        )
        subparser.add_argument(
            "--wandb-entity",
            type=str,
            default=None,
            help="W&B entity/team to log to.",
        )
        subparser.add_argument(
            "--wandb-run-name",
            type=str,
            default=None,
            help="Explicit W&B run name.",
        )

    def add_lora_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--lora-r",
            type=int,
            default=DEFAULT_LORA_R,
            help="LoRA rank.",
        )
        subparser.add_argument(
            "--lora-alpha",
            type=int,
            default=DEFAULT_LORA_ALPHA,
            help="LoRA alpha.",
        )
        subparser.add_argument(
            "--lora-dropout",
            type=float,
            default=DEFAULT_LORA_DROPOUT,
            help="LoRA dropout.",
        )
        subparser.add_argument(
            "--lora-bias",
            type=str,
            default=DEFAULT_LORA_BIAS,
            help="LoRA bias setting.",
        )

    def add_training_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--optim",
            type=str,
            default=DEFAULT_OPTIM,
            help="Optimizer name.",
        )
        subparser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Per-device batch size.",
        )
        subparser.add_argument(
            "--grad-accum",
            type=int,
            default=DEFAULT_GRAD_ACCUM,
            help="Gradient accumulation steps.",
        )
        subparser.add_argument(
            "--lr",
            type=float,
            default=DEFAULT_LR,
            help="Learning rate.",
        )
        subparser.add_argument(
            "--lr-scheduler",
            type=str,
            default=DEFAULT_LR_SCHEDULER,
            help="Learning rate scheduler.",
        )
        subparser.add_argument(
            "--warmup-ratio",
            type=float,
            default=DEFAULT_WARMUP_RATIO,
            help="Warmup ratio.",
        )
        subparser.add_argument(
            "--weight-decay",
            type=float,
            default=DEFAULT_WEIGHT_DECAY,
            help="Weight decay.",
        )
        subparser.add_argument(
            "--epochs",
            type=int,
            required=True,
            help="Number of training epochs.",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=105,
            help="Random seed.",
        )

    sft = subparsers.add_parser(
        "sft",
        help="Run supervised fine-tuning (LoRA) on Modal.",
    )
    add_shared_args(sft)
    add_lora_args(sft)
    add_training_args(sft)
    sft.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODELS["smollm2-135m"],
        help="Base HF model name or alias.",
    )

    grpo = subparsers.add_parser(
        "grpo",
        help="Run GRPO alignment on Modal.",
    )
    add_shared_args(grpo)
    add_lora_args(grpo)
    add_training_args(grpo)
    grpo.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the SFT model/adapters (Modal volume path or HF repo).",
    )
    grpo.add_argument(
        "--reward-type",
        type=str,
        default="clap",
        help="Reward strategy: clap or schema_only.",
    )
    grpo.add_argument(
        "--audio-reward",
        type=str,
        default=None,
        help=(
            "Advanced: module:function path for audio scorer (CLAP). "
            "Example: data_work.audio_rewards:score_clap"
        )
        if show_advanced
        else argparse.SUPPRESS,
    )
    grpo.add_argument(
        "--num-generations",
        type=int,
        default=DEFAULT_GRPO_NUM_GENERATIONS,
        help="Number of generations per prompt for GRPO.",
    )
    grpo.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_GRPO_BETA,
        help="KL penalty beta for GRPO.",
    )

    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--advanced", action="store_true")
    known, _ = pre_parser.parse_known_args(argv)
    parser = _build_parser(show_advanced=known.advanced)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv if argv is not None else os.sys.argv[1:])

    data_file = None
    output_dir = None
    remote_data_file = None
    if args.command in {"sft", "grpo"}:
        data_file = args.data.expanduser().resolve()
        if not data_file.exists():
            raise SystemExit(f"Data file not found: {data_file}")
        output_dir = Path(REMOTE_OUTPUT_PATH) / args.output
        remote_data_file = _resolve_remote_data_file(data_file)

    with app.run():
        try:
            match args.command:
                case "check-imports":
                    result = check_imports.remote()
                case "sft":
                    config = SftConfig(
                        base_model=_resolve_model(args.base_model),
                        data_file=str(remote_data_file),
                        output_dir=str(output_dir),
                        system_prompt=args.system_prompt,
                        prompt_field=args.prompt_field,
                        response_field=args.response_field,
                        max_seq_length=args.max_seq_length,
                        lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        lora_bias=args.lora_bias,
                        optim=args.optim,
                        batch_size=args.batch_size,
                        gradient_accumulation_steps=args.grad_accum,
                        learning_rate=args.lr,
                        lr_scheduler_type=args.lr_scheduler,
                        warmup_ratio=args.warmup_ratio,
                        weight_decay=args.weight_decay,
                        epochs=args.epochs,
                        seed=args.seed,
                        overwrite=args.overwrite,
                        wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_run_name=args.wandb_run_name,
                    )
                    result = run_sft.remote(config.model_dump())
                case "grpo":
                    config = GrpoConfig(
                        model_path=args.model,
                        data_file=str(remote_data_file),
                        output_dir=str(output_dir),
                        system_prompt=args.system_prompt,
                        prompt_field=args.prompt_field,
                        response_field=args.response_field,
                        max_seq_length=args.max_seq_length,
                        lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        lora_bias=args.lora_bias,
                        optim=args.optim,
                        batch_size=args.batch_size,
                        gradient_accumulation_steps=args.grad_accum,
                        learning_rate=args.lr,
                        lr_scheduler_type=args.lr_scheduler,
                        warmup_ratio=args.warmup_ratio,
                        weight_decay=args.weight_decay,
                        epochs=args.epochs,
                        num_generations=args.num_generations,
                        beta=args.beta,
                        reward_type=args.reward_type,
                        audio_reward=getattr(args, "audio_reward", None),
                        seed=args.seed,
                        overwrite=args.overwrite,
                        wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_run_name=args.wandb_run_name,
                    )
                    result = run_grpo.remote(config.model_dump())
                case _:
                    raise SystemExit(f"Unsupported command: {args.command}")
        except Exception as exc:
            LOGGER.exception("Modal job failed.")
            raise SystemExit(
                "Modal job failed. Check the Modal logs for container errors."
            ) from exc

        if args.command != "check-imports" and args.download_dir:
            local_destination = args.download_dir.expanduser()
            downloaded_path = _download_from_volume(result, local_destination)
            print(f"Downloaded output to {downloaded_path}.")
            if args.delete_remote:
                _delete_from_volume(result)
                print(f"Deleted Modal output at {result}.")
        print(f"Modal job completed. Output stored at {result}.")


if __name__ == "__main__":
    main()
