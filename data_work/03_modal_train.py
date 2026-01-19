"""Modal-driven SFT + GRPO training for tiny LLMs."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from common.prompt_registry import list_prompts, render_config_prompt
from common.reward_config import DEFAULT_REWARD_CONFIG, RewardConfig
from common.training_config import TrainingConfig
from data_work.lib.llm_client import (
    normalize_tokenizer_for_model,
    render_chat_prompt,
    wrap_vibe_for_chat,
)

ModelFamily = Literal["gemma", "qwen"]

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - handled in tests without Modal installed
    modal = None

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
if modal is not None:
    modal.enable_output()

APP_NAME = "latentscore-modal-train"
GPU_TYPE = "L40S"
TIMEOUT_HOURS = 6
MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
REMOTE_OUTPUT_PATH = "/outputs"
VOLUME_NAME = "latentscore-training-outputs"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_VERSION = "config_v1"
DEFAULT_SYSTEM_PROMPT = render_config_prompt(DEFAULT_PROMPT_VERSION)
DEFAULT_DATASET_TEXT_FIELD = "text"
DEFAULT_PROMPT_FIELD = "vibe_noisy"
DEFAULT_RESPONSE_FIELD = "config_payload"
DEFAULT_MAX_SEQ_LEN = 4096

DEFAULT_GRPO_MAX_COMPLETION_LENGTH = 2048
DEFAULT_GRPO_TEMPERATURE = 0.7
DEFAULT_GRPO_TOP_P = 0.8
DEFAULT_GRPO_TOP_K = 20

DEBUG_PROMPT_TRUNCATE = 400
DEBUG_PROMPT_MAX_DEFAULT = 1

PROMPT_REGISTRY_NAMES = list_prompts()
PROMPT_REGISTRY_HELP = ", ".join(sorted(PROMPT_REGISTRY_NAMES))

PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"

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
    "gemma3-270m": "unsloth/gemma-3-270m-it",
    "qwen3-600m": "unsloth/Qwen3-0.6B",
}
DEFAULT_BASE_MODEL_KEY = "gemma3-270m"


class _ModalStubRun:
    def __enter__(self) -> None:
        raise RuntimeError("Modal is required to run training commands.")

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> bool:
        return False


class _ModalStubApp:
    def function(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def run(self) -> _ModalStubRun:
        return _ModalStubRun()


if modal is None:
    RETRY_POLICY = None
    OUTPUTS_VOLUME = None
    TRAIN_IMAGE = None
    app = _ModalStubApp()
else:
    TRAIN_IMAGE_PACKAGES = (
        "accelerate==1.5.2",
        "bitsandbytes==0.46.0",
        "datasets==3.5.1",
        "hf_transfer==0.1.9",
        "huggingface_hub==0.34.4",
        "laion-clap",
        "numpy==2.2.6",
        "peft==0.17.1",
        "protobuf==5.29.5",
        "pydantic==2.11.7",
        "scipy==1.16.1",
        "sentencepiece==0.2.0",
        "soundfile",
        "torchvision",
        "torchaudio",
        "tqdm==4.67.1",
        "transformers==4.54.1",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
        "weave==0.50.0",
    )
    RETRY_POLICY = modal.Retries(
        max_retries=MAX_RETRIES,
        backoff_coefficient=DEFAULT_RETRY_BACKOFF,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY,
    )
    OUTPUTS_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    TRAIN_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libsndfile1")
        .uv_pip_install(
            *TRAIN_IMAGE_PACKAGES,
            extra_index_url=PYTORCH_CU128_INDEX,
            extra_options="--index-strategy unsafe-best-match",
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
    debug_prompts: bool = False
    debug_prompts_max: int = DEBUG_PROMPT_MAX_DEFAULT


class GrpoConfig(BaseModel):
    """GRPO configuration for modal training."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    base_model: str
    init_adapter_dir: str | None
    data_file: str
    output_dir: str
    system_prompt: str
    prompt_field: str
    response_field: str
    max_seq_length: int
    max_completion_length: int
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
    temperature: float
    top_p: float
    top_k: int
    reward_type: str
    reward_config: RewardConfig
    audio_reward: str | None
    seed: int
    overwrite: bool
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None
    debug_prompts: bool = False
    debug_prompts_max: int = DEBUG_PROMPT_MAX_DEFAULT


_VIBE_TAG_RE = re.compile(r"<vibe>(.*?)</vibe>", re.DOTALL)


def _clean_completion(text: str) -> str:
    cleaned = text.strip()
    if "<|im_end|>" in cleaned:
        cleaned = cleaned.split("<|im_end|>", 1)[0]
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def _truncate_debug_text(value: str, limit: int = DEBUG_PROMPT_TRUNCATE) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    head = value[: limit // 2]
    tail = value[-limit // 2 :]
    return f"{head}...<snip {len(value) - limit} chars>...{tail}"


def _log_prompt_sample(
    *,
    stage: str,
    sample_idx: int,
    system_prompt: str,
    user_prompt: str,
    rendered_prompt: str,
    response: str | None,
) -> None:
    lines = [
        (
            f"[TRAIN_DEBUG] {stage} sample {sample_idx}: system_len={len(system_prompt)} "
            f"user_len={len(user_prompt)} rendered_len={len(rendered_prompt)}"
        ),
        f"[TRAIN_DEBUG] {stage} system_head={_truncate_debug_text(repr(system_prompt))}",
        f"[TRAIN_DEBUG] {stage} user_prompt={_truncate_debug_text(repr(user_prompt))}",
        f"[TRAIN_DEBUG] {stage} rendered_head={_truncate_debug_text(repr(rendered_prompt))}",
    ]
    if response is not None:
        lines.append(f"[TRAIN_DEBUG] {stage} assistant_head={_truncate_debug_text(repr(response))}")
    for line in lines:
        LOGGER.info("%s", line)
        print(line)


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


def _estimate_max_token_length(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    batch_size: int = 256,
) -> int:
    max_length = 0
    batch: list[str] = []

    def flush_batch() -> int:
        if not batch:
            return 0
        encoded = tokenizer(batch, add_special_tokens=False)
        input_ids = encoded.get("input_ids", [])
        return max((len(ids) for ids in input_ids), default=0)

    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            max_length = max(max_length, flush_batch())
            batch.clear()

    max_length = max(max_length, flush_batch())
    return max_length


def _validate_max_seq_length(max_seq_length: int, observed_max_length: int) -> None:
    if observed_max_length > max_seq_length:
        raise SystemExit(
            "Configured max_seq_length is too small for the dataset. "
            f"max_seq_length={max_seq_length}, observed_max_length={observed_max_length}. "
            "Increase --max-seq-length to at least the observed value."
        )


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
    reward_config: RewardConfig | None = None,
) -> Callable[..., list[float]]:
    """Create a reward function for GRPO training.

    Args:
        reward_type: Type of reward ('clap', 'schema_only', 'custom')
        audio_scorer: Optional callable for audio similarity scoring
        wandb_run: Optional W&B run for logging
        reward_config: Optional RewardConfig for configurable weights

    Returns:
        Reward function callable for GRPO trainer
    """
    from data_work.lib.rewards import compute_partial_reward

    if reward_config is None:
        reward_config = DEFAULT_REWARD_CONFIG

    def reward_fn(
        prompts: Sequence[str],
        completions: Sequence[str],
        vibe: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        breakdowns: list[RewardBreakdown] = []
        for idx, completion in enumerate(completions):
            if vibe is not None:
                vibe_text = vibe[idx] if idx < len(vibe) else ""
            else:
                prompt = prompts[idx] if idx < len(prompts) else ""
                vibe_text = _extract_vibe_from_prompt(prompt)
            cleaned = _clean_completion(completion)
            breakdown = compute_partial_reward(
                vibe=vibe_text,
                output=cleaned,
                audio_scorer=audio_scorer,
                config=reward_config,
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
                        "reward/title_similarity_mean": sum(b.title_similarity for b in breakdowns)
                        / len(breakdowns),
                        "reward/title_penalty_mean": sum(b.title_length_penalty for b in breakdowns)
                        / len(breakdowns),
                        "reward/title_score_mean": sum(b.title_score for b in breakdowns)
                        / len(breakdowns),
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
    match parsed:
        case dict():
            return parsed
        case _:
            return None


def _extract_vibe_from_prompt(prompt: str) -> str:
    parsed_prompt = _parse_prompt_json(prompt)
    if parsed_prompt is not None:
        return str(parsed_prompt.get("user", prompt))
    match = _VIBE_TAG_RE.search(prompt)
    if match:
        return match.group(1).strip()
    return prompt


def _ensure_gradient_checkpointing(model: Any) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    try:
        import torch
    except Exception:
        return
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False) and not hasattr(
            module, "_gradient_checkpointing_func"
        ):
            module._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint


def _configure_torch_dynamo() -> None:
    try:
        import torch._dynamo as dynamo
    except Exception:
        return
    if hasattr(dynamo.config, "cache_size_limit"):
        dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)
    if hasattr(dynamo.config, "fail_on_recompile_limit_hit"):
        dynamo.config.fail_on_recompile_limit_hit = False


def _ensure_lora_trainable(model: Any) -> None:
    trainable = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            if not param.requires_grad:
                param.requires_grad = True
            trainable += param.numel()
    if trainable == 0:
        LOGGER.warning("No LoRA parameters marked trainable; GRPO gradients may be empty.")


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
    from unsloth import FastLanguageModel, FastModel
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

    def detect_model_family(model_key: str) -> ModelFamily:
        lower = model_key.lower()
        if "gemma" in lower:
            return "gemma"
        if "qwen" in lower:
            return "qwen"
        raise NotImplementedError(f"Unknown model family for: {model_key}. Supported: gemma, qwen")

    model_family: ModelFamily = detect_model_family(sft.base_model)
    match model_family:
        case "gemma":
            model, tokenizer = FastModel.from_pretrained(
                model_name=sft.base_model,
                max_seq_length=sft.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )
            normalize_tokenizer_for_model(tokenizer, sft.base_model)
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=sft.lora_r,
                lora_alpha=sft.lora_alpha,
                lora_dropout=sft.lora_dropout,
                bias=sft.lora_bias,
                random_state=sft.seed,
            )
        case "qwen":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=sft.base_model,
                max_seq_length=sft.max_seq_length,
                dtype=None,
                load_in_4bit=False,
                load_in_8bit=False,
            )
            normalize_tokenizer_for_model(tokenizer, sft.base_model)
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
        case _:
            raise NotImplementedError(
                f"Model family '{model_family}' not supported. "
                f"Supported families: gemma, qwen. Model key: {sft.base_model}"
            )

    dataset = datasets.load_dataset("json", data_files=sft.data_file, split="train")
    debug_prompts = sft.debug_prompts
    debug_limit = max(0, sft.debug_prompts_max)
    debug_count = 0

    def has_prompt(example: dict[str, Any]) -> bool:
        return example.get(sft.prompt_field) not in (None, "")

    def has_response(example: dict[str, Any]) -> bool:
        return example.get(sft.response_field) not in (None, "")

    dataset = dataset.filter(has_prompt)
    dataset = dataset.filter(has_response)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        nonlocal debug_count
        prompt = str(example[sft.prompt_field])
        response = json.dumps(example[sft.response_field], ensure_ascii=False)
        user_prompt = wrap_vibe_for_chat(prompt)
        text = render_chat_prompt(
            system_prompt=sft.system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            model_name=sft.base_model,
            add_generation_prompt=False,
            assistant=response,
        )
        if debug_prompts and debug_count < debug_limit:
            debug_count += 1
            _log_prompt_sample(
                stage="SFT",
                sample_idx=debug_count,
                system_prompt=sft.system_prompt,
                user_prompt=user_prompt,
                rendered_prompt=text,
                response=response,
            )
        return {DEFAULT_DATASET_TEXT_FIELD: text}

    map_kwargs = {}
    if debug_prompts:
        map_kwargs["load_from_cache_file"] = False
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        **map_kwargs,
    )

    observed_max_length = _estimate_max_token_length(
        (example[DEFAULT_DATASET_TEXT_FIELD] for example in dataset),
        tokenizer,
    )
    _validate_max_seq_length(sft.max_seq_length, observed_max_length)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        bf16=supports_bf16,
        fp16=torch.cuda.is_available() and not supports_bf16,
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
    from unsloth import FastLanguageModel, FastModel
    # isort: on

    import datasets
    import torch
    import wandb
    import weave
    from peft.utils import load_peft_weights, set_peft_model_state_dict
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
    reward_fn = _create_reward_fn(
        grpo.reward_type,
        audio_scorer,
        wandb_run,
        reward_config=grpo.reward_config,
    )

    def detect_model_family(model_key: str) -> ModelFamily:
        lower = model_key.lower()
        if "gemma" in lower:
            return "gemma"
        if "qwen" in lower:
            return "qwen"
        raise NotImplementedError(f"Unknown model family for: {model_key}. Supported: gemma, qwen")

    model_family: ModelFamily = detect_model_family(grpo.base_model)
    match model_family:
        case "gemma":
            model, tokenizer = FastModel.from_pretrained(
                model_name=grpo.model_path,
                max_seq_length=grpo.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )
            normalize_tokenizer_for_model(tokenizer, grpo.base_model)
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=grpo.lora_r,
                lora_alpha=grpo.lora_alpha,
                lora_dropout=grpo.lora_dropout,
                bias=grpo.lora_bias,
                random_state=grpo.seed,
            )
        case "qwen":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=grpo.model_path,
                max_seq_length=grpo.max_seq_length,
                dtype=torch.float16 if torch.cuda.is_available() else None,
                load_in_4bit=False,
                load_in_8bit=False,
            )
            normalize_tokenizer_for_model(tokenizer, grpo.base_model)
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
        case _:
            raise NotImplementedError(
                f"Model family '{model_family}' not supported. "
                f"Supported families: gemma, qwen. Model key: {grpo.base_model}"
            )
    if grpo.init_adapter_dir:
        adapter_weights = load_peft_weights(grpo.init_adapter_dir, device=None)
        set_peft_model_state_dict(model, adapter_weights, adapter_name="default")
        try:
            model.set_adapter("default")
        except Exception:
            pass
    _ensure_gradient_checkpointing(model)
    _ensure_lora_trainable(model)
    _configure_torch_dynamo()

    dataset = datasets.load_dataset("json", data_files=grpo.data_file, split="train")
    debug_prompts = grpo.debug_prompts
    debug_limit = max(0, grpo.debug_prompts_max)
    debug_count = 0

    def has_prompt(example: dict[str, Any]) -> bool:
        return example.get(grpo.prompt_field) not in (None, "")

    dataset = dataset.filter(has_prompt)

    def format_prompts(example: dict[str, Any]) -> dict[str, str]:
        nonlocal debug_count
        vibe = str(example[grpo.prompt_field])
        user_prompt = wrap_vibe_for_chat(vibe)
        text = render_chat_prompt(
            system_prompt=grpo.system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            model_name=grpo.base_model,
            add_generation_prompt=True,
        )
        if debug_prompts and debug_count < debug_limit:
            debug_count += 1
            _log_prompt_sample(
                stage="GRPO",
                sample_idx=debug_count,
                system_prompt=grpo.system_prompt,
                user_prompt=user_prompt,
                rendered_prompt=text,
                response=None,
            )
        return {"prompt": text, "vibe": vibe}

    map_kwargs = {}
    if debug_prompts:
        map_kwargs["load_from_cache_file"] = False
    dataset = dataset.map(
        format_prompts,
        remove_columns=dataset.column_names,
        **map_kwargs,
    )

    observed_max_length = _estimate_max_token_length(
        (example["prompt"] for example in dataset),
        tokenizer,
    )
    _validate_max_seq_length(grpo.max_seq_length, observed_max_length)

    # Gemma models use bf16 due to large activation values; other models use fp16
    use_bf16 = model_family == "gemma" and torch.cuda.is_available()
    use_fp16 = not use_bf16 and torch.cuda.is_available()

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
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": ["wandb"] if wandb_run else [],
        "max_completion_length": grpo.max_completion_length,
        "temperature": grpo.temperature,
        "top_p": grpo.top_p,
        "top_k": grpo.top_k,
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
            "--config-file",
            type=Path,
            default=None,
            help=(
                "Path to JSON config file with TrainingConfig overrides. "
                "CLI args take precedence over config file values."
            ),
        )
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
            default=None,
            help=(
                "Override system prompt used to format training samples. "
                "Defaults to --prompt-version."
            ),
        )
        subparser.add_argument(
            "--prompt-version",
            type=str,
            default=DEFAULT_PROMPT_VERSION,
            help=(
                "Prompt registry key for config generation prompts. "
                f"Available: {PROMPT_REGISTRY_HELP}"
            ),
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
            "--debug-prompts",
            action="store_true",
            help="Log formatted prompt samples (system/user/renderer) for verification.",
        )
        subparser.add_argument(
            "--debug-prompts-max",
            type=int,
            default=DEBUG_PROMPT_MAX_DEFAULT,
            help="Maximum number of prompt samples to log when --debug-prompts is set.",
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
        default=BASE_MODELS[DEFAULT_BASE_MODEL_KEY],
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
        help="Base model path or HF repo (used to initialize GRPO policy).",
    )
    grpo.add_argument(
        "--init-adapter-dir",
        type=str,
        default=None,
        help="Optional path to SFT LoRA adapter to initialize GRPO from.",
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
    grpo.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_GRPO_MAX_COMPLETION_LENGTH,
        help="Maximum completion length for GRPO sampling.",
    )
    grpo.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_GRPO_TEMPERATURE,
        help="Sampling temperature for GRPO generations.",
    )
    grpo.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_GRPO_TOP_P,
        help="Top-p nucleus sampling for GRPO generations.",
    )
    grpo.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_GRPO_TOP_K,
        help="Top-k sampling for GRPO generations.",
    )
    grpo.add_argument(
        "--ablation-preset",
        type=str,
        default=None,
        help=(
            "Use predefined ablation config. Format: 'category:name' "
            "(e.g., 'lora_rank:r32', 'learning_rate:lr1e-4'). "
            "See common.training_config.ABLATION_PRESETS for options."
        )
        if show_advanced
        else argparse.SUPPRESS,
    )
    grpo.add_argument(
        "--format-weight",
        type=float,
        default=None,
        help="Override reward format weight (0.0-1.0)." if show_advanced else argparse.SUPPRESS,
    )
    grpo.add_argument(
        "--schema-weight",
        type=float,
        default=None,
        help="Override reward schema weight (0.0-1.0)." if show_advanced else argparse.SUPPRESS,
    )
    grpo.add_argument(
        "--audio-weight",
        type=float,
        default=None,
        help="Override reward audio weight (0.0-1.0)." if show_advanced else argparse.SUPPRESS,
    )
    grpo.add_argument(
        "--title-similarity-weight",
        type=float,
        default=None,
        help=(
            "Override reward title similarity weight (0.0-1.0)."
            if show_advanced
            else argparse.SUPPRESS
        ),
    )
    grpo.add_argument(
        "--title-length-penalty-weight",
        type=float,
        default=None,
        help=(
            "Override reward title length penalty weight (0.0-1.0)."
            if show_advanced
            else argparse.SUPPRESS
        ),
    )

    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--advanced", action="store_true")
    known, _ = pre_parser.parse_known_args(argv)
    parser = _build_parser(show_advanced=known.advanced)
    return parser.parse_args(argv)


def _load_config_file(config_path: Path | None) -> TrainingConfig | None:
    """Load TrainingConfig from JSON file if provided."""
    if config_path is None:
        return None
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with open(config_path) as f:
        data = json.load(f)
    return TrainingConfig.model_validate(data)


def _apply_ablation_preset(preset_spec: str | None) -> TrainingConfig | None:
    """Apply ablation preset if specified.

    Args:
        preset_spec: Format 'category:name' (e.g., 'lora_rank:r32')

    Returns:
        TrainingConfig with preset applied, or None if no preset
    """
    if preset_spec is None:
        return None

    from common.training_config import ABLATION_PRESETS

    parts = preset_spec.split(":", 1)
    if len(parts) != 2:
        raise SystemExit(
            f"Invalid ablation preset format: '{preset_spec}'. "
            "Expected 'category:name' (e.g., 'lora_rank:r32')."
        )
    category, name = parts
    if category not in ABLATION_PRESETS:
        raise SystemExit(
            f"Unknown ablation category: '{category}'. Available: {list(ABLATION_PRESETS.keys())}"
        )
    if name not in ABLATION_PRESETS[category]:
        raise SystemExit(
            f"Unknown preset '{name}' in category '{category}'. "
            f"Available: {list(ABLATION_PRESETS[category].keys())}"
        )
    return ABLATION_PRESETS[category][name]


def _build_reward_config_from_args(args: argparse.Namespace) -> RewardConfig | None:
    """Build RewardConfig from CLI args if any reward weights are overridden."""
    from common.reward_config import RewardWeights

    format_weight = getattr(args, "format_weight", None)
    schema_weight = getattr(args, "schema_weight", None)
    audio_weight = getattr(args, "audio_weight", None)
    title_similarity_weight = getattr(args, "title_similarity_weight", None)
    title_length_penalty_weight = getattr(args, "title_length_penalty_weight", None)

    if all(
        w is None
        for w in [
            format_weight,
            schema_weight,
            audio_weight,
            title_similarity_weight,
            title_length_penalty_weight,
        ]
    ):
        return None

    # Start with defaults, override specified values
    weights_dict: dict[str, float] = {}
    if format_weight is not None:
        weights_dict["format_weight"] = format_weight
    if schema_weight is not None:
        weights_dict["schema_weight"] = schema_weight
    if audio_weight is not None:
        weights_dict["audio_weight"] = audio_weight
    if title_similarity_weight is not None:
        weights_dict["title_similarity_weight"] = title_similarity_weight
    if title_length_penalty_weight is not None:
        weights_dict["title_length_penalty_weight"] = title_length_penalty_weight

    return RewardConfig(weights=RewardWeights(**weights_dict))


def _resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Resolve effective TrainingConfig from config file, presets, and CLI args.

    Priority (highest to lowest):
    1. CLI args (explicit overrides)
    2. Ablation preset
    3. Config file
    4. Default TrainingConfig
    """
    from common.training_config import DEFAULT_TRAINING_CONFIG

    # Start with default or loaded config
    config_file_path = getattr(args, "config_file", None)
    base_config = _load_config_file(config_file_path) or DEFAULT_TRAINING_CONFIG

    # Apply ablation preset if specified (for GRPO)
    preset_spec = getattr(args, "ablation_preset", None)
    if preset_spec:
        preset_config = _apply_ablation_preset(preset_spec)
        if preset_config:
            # Merge preset into base config
            base_config = base_config.model_copy(
                update={
                    "lora": preset_config.lora,
                    "optimizer": preset_config.optimizer,
                    "batch": preset_config.batch,
                    "grpo": preset_config.grpo,
                }
            )

    # Build updates from CLI args (CLI takes precedence)
    updates: dict[str, Any] = {}

    # Base model
    if hasattr(args, "base_model") and args.base_model != BASE_MODELS.get(DEFAULT_BASE_MODEL_KEY):
        updates["base_model"] = args.base_model

    # System prompt
    system_prompt = render_config_prompt(args.prompt_version)
    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    if system_prompt != base_config.system_prompt:
        updates["system_prompt"] = system_prompt

    # LoRA config - check if any CLI args differ from config
    lora_updates: dict[str, Any] = {}
    if args.lora_r != base_config.lora.r:
        lora_updates["r"] = args.lora_r
    if args.lora_alpha != base_config.lora.alpha:
        lora_updates["alpha"] = args.lora_alpha
    if args.lora_dropout != base_config.lora.dropout:
        lora_updates["dropout"] = args.lora_dropout
    if args.lora_bias != base_config.lora.bias:
        lora_updates["bias"] = args.lora_bias
    if lora_updates:
        updates["lora"] = base_config.lora.model_copy(update=lora_updates)

    # Optimizer config
    optim_updates: dict[str, Any] = {}
    if args.optim != base_config.optimizer.name:
        optim_updates["name"] = args.optim
    if args.lr != base_config.optimizer.learning_rate:
        optim_updates["learning_rate"] = args.lr
    if args.lr_scheduler != base_config.optimizer.lr_scheduler:
        optim_updates["lr_scheduler"] = args.lr_scheduler
    if args.warmup_ratio != base_config.optimizer.warmup_ratio:
        optim_updates["warmup_ratio"] = args.warmup_ratio
    if args.weight_decay != base_config.optimizer.weight_decay:
        optim_updates["weight_decay"] = args.weight_decay
    if optim_updates:
        updates["optimizer"] = base_config.optimizer.model_copy(update=optim_updates)

    # Batch config
    batch_updates: dict[str, Any] = {}
    if args.batch_size != base_config.batch.batch_size:
        batch_updates["batch_size"] = args.batch_size
    if args.grad_accum != base_config.batch.gradient_accumulation_steps:
        batch_updates["gradient_accumulation_steps"] = args.grad_accum
    if batch_updates:
        updates["batch"] = base_config.batch.model_copy(update=batch_updates)

    # Data config
    data_updates: dict[str, Any] = {}
    if args.prompt_field != base_config.data.prompt_field:
        data_updates["prompt_field"] = args.prompt_field
    if args.response_field != base_config.data.response_field:
        data_updates["response_field"] = args.response_field
    if args.max_seq_length != base_config.data.max_seq_length:
        data_updates["max_seq_length"] = args.max_seq_length
    if data_updates:
        updates["data"] = base_config.data.model_copy(update=data_updates)

    # GRPO config (only for GRPO command)
    if args.command == "grpo":
        grpo_updates: dict[str, Any] = {}
        if args.num_generations != base_config.grpo.num_generations:
            grpo_updates["num_generations"] = args.num_generations
        if args.beta != base_config.grpo.beta:
            grpo_updates["beta"] = args.beta
        # Apply reward config from CLI args
        reward_config = _build_reward_config_from_args(args)
        if reward_config:
            grpo_updates["reward"] = reward_config
        if grpo_updates:
            updates["grpo"] = base_config.grpo.model_copy(update=grpo_updates)

    # Logging config
    logging_updates: dict[str, Any] = {}
    if args.wandb_project != base_config.logging.wandb_project:
        logging_updates["wandb_project"] = args.wandb_project
    if args.wandb_entity != base_config.logging.wandb_entity:
        logging_updates["wandb_entity"] = args.wandb_entity
    if logging_updates:
        updates["logging"] = base_config.logging.model_copy(update=logging_updates)

    # Epochs and seed
    if args.epochs != base_config.epochs:
        updates["epochs"] = args.epochs
    if args.seed != base_config.seed:
        updates["seed"] = args.seed

    if updates:
        return base_config.model_copy(update=updates)
    return base_config


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

    # Resolve training config from file, presets, and CLI args
    training_config = _resolve_training_config(args) if args.command in {"sft", "grpo"} else None

    with app.run():
        try:
            match args.command:
                case "check-imports":
                    result = check_imports.remote()
                case "sft":
                    assert training_config is not None
                    config = SftConfig(
                        base_model=training_config.resolve_base_model(),
                        data_file=str(remote_data_file),
                        output_dir=str(output_dir),
                        system_prompt=training_config.system_prompt,
                        prompt_field=training_config.data.prompt_field,
                        response_field=training_config.data.response_field,
                        max_seq_length=training_config.data.max_seq_length,
                        lora_r=training_config.lora.r,
                        lora_alpha=training_config.lora.alpha,
                        lora_dropout=training_config.lora.dropout,
                        lora_bias=training_config.lora.bias,
                        optim=training_config.optimizer.name,
                        batch_size=training_config.batch.batch_size,
                        gradient_accumulation_steps=training_config.batch.gradient_accumulation_steps,
                        learning_rate=training_config.optimizer.learning_rate,
                        lr_scheduler_type=training_config.optimizer.lr_scheduler,
                        warmup_ratio=training_config.optimizer.warmup_ratio,
                        weight_decay=training_config.optimizer.weight_decay,
                        epochs=training_config.epochs,
                        seed=training_config.seed,
                        overwrite=args.overwrite,
                        wandb_project=training_config.logging.wandb_project,
                        wandb_entity=training_config.logging.wandb_entity,
                        wandb_run_name=args.wandb_run_name,
                        debug_prompts=args.debug_prompts,
                        debug_prompts_max=args.debug_prompts_max,
                    )
                    LOGGER.info(
                        "Using TrainingConfig: %s", training_config.model_dump_json(indent=2)
                    )
                    result = run_sft.remote(config.model_dump())
                case "grpo":
                    assert training_config is not None
                    config = GrpoConfig(
                        model_path=_resolve_model_path(args.model),
                        base_model=training_config.resolve_base_model(),
                        init_adapter_dir=(
                            _resolve_model_path(args.init_adapter_dir)
                            if args.init_adapter_dir
                            else None
                        ),
                        data_file=str(remote_data_file),
                        output_dir=str(output_dir),
                        system_prompt=training_config.system_prompt,
                        prompt_field=training_config.data.prompt_field,
                        response_field=training_config.data.response_field,
                        max_seq_length=training_config.data.max_seq_length,
                        max_completion_length=args.max_completion_length,
                        lora_r=training_config.lora.r,
                        lora_alpha=training_config.lora.alpha,
                        lora_dropout=training_config.lora.dropout,
                        lora_bias=training_config.lora.bias,
                        optim=training_config.optimizer.name,
                        batch_size=training_config.batch.batch_size,
                        gradient_accumulation_steps=training_config.batch.gradient_accumulation_steps,
                        learning_rate=training_config.optimizer.learning_rate,
                        lr_scheduler_type=training_config.optimizer.lr_scheduler,
                        warmup_ratio=training_config.optimizer.warmup_ratio,
                        weight_decay=training_config.optimizer.weight_decay,
                        epochs=training_config.epochs,
                        num_generations=training_config.grpo.num_generations,
                        beta=training_config.grpo.beta,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        reward_type=args.reward_type,
                        reward_config=training_config.grpo.reward,
                        audio_reward=getattr(args, "audio_reward", None),
                        seed=training_config.seed,
                        overwrite=args.overwrite,
                        wandb_project=training_config.logging.wandb_project,
                        wandb_entity=training_config.logging.wandb_entity,
                        wandb_run_name=args.wandb_run_name,
                        debug_prompts=args.debug_prompts,
                        debug_prompts_max=args.debug_prompts_max,
                    )
                    LOGGER.info(
                        "Using TrainingConfig: %s", training_config.model_dump_json(indent=2)
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
