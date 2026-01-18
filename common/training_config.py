"""Centralized training configuration for ablation studies.

All training hyperparameters are defined here with sensible defaults.
Override via CLI, config files, or programmatically for ablation experiments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from common.prompts import build_config_generation_prompt
from common.reward_config import RewardConfig

# --- LoRA Configuration ---


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = Field(default=16, ge=1, description="LoRA rank (higher = more capacity)")
    alpha: int = Field(default=16, ge=1, description="LoRA scaling factor")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout rate")
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none", description="Bias training mode"
    )
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Module names to apply LoRA to",
    )


# --- Optimizer Configuration ---


class OptimizerConfig(BaseModel):
    """Optimizer and learning rate configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="adamw_8bit", description="Optimizer name")
    learning_rate: float = Field(default=2e-4, gt=0.0, description="Peak learning rate")
    lr_scheduler: str = Field(default="cosine", description="LR scheduler type")
    warmup_ratio: float = Field(
        default=0.06, ge=0.0, le=1.0, description="Warmup fraction of total steps"
    )
    weight_decay: float = Field(default=0.01, ge=0.0, description="L2 regularization")


# --- Batch Configuration ---


class BatchConfig(BaseModel):
    """Batch size and gradient accumulation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=16, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Gradient accumulation steps"
    )

    @property
    def effective_batch_size(self) -> int:
        """Return effective batch size (batch_size * gradient_accumulation_steps)."""
        return self.batch_size * self.gradient_accumulation_steps


# --- GRPO-Specific Configuration ---


class GRPOConfig(BaseModel):
    """GRPO-specific training parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_generations: int = Field(
        default=4, ge=1, description="Number of candidate generations per prompt"
    )
    beta: float = Field(default=0.04, ge=0.0, description="KL divergence penalty coefficient")
    reward: RewardConfig = Field(default_factory=RewardConfig, description="Reward configuration")


# --- Data Configuration ---


class DataConfig(BaseModel):
    """Data loading and field configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_field: str = Field(default="vibe_noisy", description="Input field name")
    response_field: str = Field(default="config_payload", description="Output field name")
    max_seq_length: int = Field(default=4096, ge=64, description="Maximum sequence length")


# --- Noise Injection Configuration ---


class NoiseConfig(BaseModel):
    """Character-level noise augmentation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error_rate: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Fraction of samples to augment"
    )
    force_augmentation: bool = Field(
        default=True, description="Ensure at least one augmented sample per split"
    )


# --- Logging Configuration ---


class LoggingConfig(BaseModel):
    """Logging and checkpointing settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logging_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    save_strategy: Literal["epoch", "steps", "no"] = Field(
        default="epoch", description="Checkpoint saving strategy"
    )
    save_steps: int = Field(default=500, ge=1, description="Save every N steps (if strategy=steps)")
    wandb_project: str | None = Field(default=None, description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B entity/team name")


# --- Infrastructure Configuration ---


class InfraConfig(BaseModel):
    """Modal/infrastructure settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_type: str = Field(default="L40S", description="GPU accelerator type")
    timeout_hours: int = Field(default=6, ge=1, description="Maximum training duration")
    max_retries: int = Field(default=3, ge=0, description="Retry attempts on failure")
    retry_initial_delay: float = Field(default=1.0, ge=0.0, description="Initial retry delay (s)")
    retry_backoff: float = Field(default=2.0, ge=1.0, description="Exponential backoff factor")
    retry_max_delay: float = Field(default=30.0, ge=0.0, description="Maximum retry delay (s)")


# --- Base Model Registry ---

BASE_MODELS: dict[str, str] = {
    "gemma3-270m": "unsloth/gemma-3-270m-it",
    "qwen3-600m": "unsloth/Qwen3-0.6B",
}


# --- Complete Training Configuration ---


class TrainingConfig(BaseModel):
    """Complete training configuration for SFT and GRPO.

    This is the single source of truth for all training hyperparameters.
    Use this for ablation studies by varying specific fields.

    Example:
        ```python
        from common.training_config import TrainingConfig, LoRAConfig

        # Default config
        config = TrainingConfig()

        # Ablation: try different LoRA ranks
        for lora_r in [4, 8, 16, 32]:
            ablation_config = config.model_copy(
                update={"lora": LoRAConfig(r=lora_r)}
            )
        ```
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Model selection
    base_model: str = Field(
        default="gemma3-270m",
        description="Base model key from BASE_MODELS or HuggingFace path",
    )

    # System prompt (configurable for prompt ablations)
    system_prompt: str = Field(
        default_factory=build_config_generation_prompt,
        description="System prompt for training",
    )

    # Component configs
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)

    # Training loop
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    def resolve_base_model(self) -> str:
        """Resolve base_model key to HuggingFace path."""
        return BASE_MODELS.get(self.base_model, self.base_model)


# Default configuration instance
DEFAULT_TRAINING_CONFIG = TrainingConfig()


# --- Ablation Presets ---

ABLATION_PRESETS: dict[str, dict[str, TrainingConfig]] = {
    "lora_rank": {f"r{r}": TrainingConfig(lora=LoRAConfig(r=r)) for r in [4, 8, 16, 32, 64]},
    "learning_rate": {
        f"lr{lr}": TrainingConfig(optimizer=OptimizerConfig(learning_rate=lr))
        for lr in [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    },
    "grpo_beta": {
        f"beta{beta}": TrainingConfig(grpo=GRPOConfig(beta=beta))
        for beta in [0.01, 0.02, 0.04, 0.08, 0.16]
    },
    "batch_size": {
        f"bs{bs}": TrainingConfig(batch=BatchConfig(batch_size=bs)) for bs in [4, 8, 16, 32]
    },
}
