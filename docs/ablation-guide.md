# Ablation Study Guide

This document covers all configurable parameters for ablation experiments.

## Quick Start

```python
from common.training_config import TrainingConfig, LoRAConfig, ABLATION_PRESETS

# Run LoRA rank ablation
for name, config in ABLATION_PRESETS["lora_rank"].items():
    print(f"Running {name}: lora_r={config.lora.r}")
    # train(config)
```

## Ablation-Worthy Parameters

### High Priority (Model Behavior)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `grpo.beta` | 0.04 | 0.01-0.16 | KL divergence penalty |
| `grpo.reward.weights.format_weight` | 0.2 | 0.0-1.0 | JSON validity importance |
| `grpo.reward.weights.schema_weight` | 0.3 | 0.0-1.0 | Schema correctness importance |
| `grpo.reward.weights.audio_weight` | 0.5 | 0.0-1.0 | Audio similarity importance |
| `noise.error_rate` | 0.15 | 0.0-0.5 | Training noise injection |
| `lora.r` | 16 | 4-64 | Adapter capacity |
| `optimizer.learning_rate` | 2e-4 | 1e-5-5e-4 | Optimization step size |

### Medium Priority (Training Dynamics)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `optimizer.warmup_ratio` | 0.06 | 0.0-0.2 | LR warmup fraction |
| `optimizer.weight_decay` | 0.01 | 0.0-0.1 | L2 regularization |
| `grpo.num_generations` | 4 | 2-8 | Candidates per prompt |
| `batch.batch_size` | 16 | 4-32 | Per-device batch |
| `data.max_seq_length` | 4096 | 256-4096 | Context window |

### Lower Priority (Architecture)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `lora.alpha` | 16 | 8-32 | LoRA scaling |
| `lora.dropout` | 0.0 | 0.0-0.1 | Regularization |
| `base_model` | gemma3-270m | see registry | Model capacity |

## Running Ablations

### Via CLI

```bash
# Single parameter override
python -m data_work.03_modal_train grpo \
  --model checkpoints/sft \
  --data data/train.jsonl \
  --output grpo-beta02 \
  --epochs 3 \
  --beta 0.02 \
  --lora-r 32

# With reward weight overrides (requires --advanced)
python -m data_work.03_modal_train --advanced grpo \
  --model checkpoints/sft \
  --data data/train.jsonl \
  --output grpo-custom-rewards \
  --epochs 3 \
  --format-weight 0.3 \
  --schema-weight 0.4 \
  --audio-weight 0.3

# Using ablation preset (requires --advanced)
python -m data_work.03_modal_train --advanced grpo \
  --model checkpoints/sft \
  --data data/train.jsonl \
  --output grpo-lora-r32 \
  --epochs 3 \
  --ablation-preset lora_rank:r32

# Full config file
python -m data_work.03_modal_train grpo \
  --config-file experiments/ablation_lr.json \
  --model checkpoints/sft \
  --data data/train.jsonl \
  --output grpo-from-config \
  --epochs 3
```

### Via Python

```python
from common.training_config import TrainingConfig, GRPOConfig, LoRAConfig
from common.reward_config import RewardConfig, RewardWeights

# Basic ablation: vary GRPO beta
configs = [
    TrainingConfig(grpo=GRPOConfig(beta=b))
    for b in [0.01, 0.02, 0.04, 0.08]
]
for config in configs:
    print(f"Beta: {config.grpo.beta}")
    # run_grpo(config)

# Combined ablation: LoRA rank + learning rate
from itertools import product

lora_ranks = [8, 16, 32]
learning_rates = [1e-4, 2e-4, 5e-4]

for r, lr in product(lora_ranks, learning_rates):
    config = TrainingConfig(
        lora=LoRAConfig(r=r),
        optimizer=OptimizerConfig(learning_rate=lr),
    )
    print(f"r={r}, lr={lr}")
    # run_training(config)

# Reward weight ablation
reward_configs = [
    RewardConfig(weights=RewardWeights(format_weight=0.1, schema_weight=0.4, audio_weight=0.5)),
    RewardConfig(weights=RewardWeights(format_weight=0.2, schema_weight=0.3, audio_weight=0.5)),
    RewardConfig(weights=RewardWeights(format_weight=0.3, schema_weight=0.2, audio_weight=0.5)),
]
```

## Available Ablation Presets

The following presets are available in `ABLATION_PRESETS`:

### `lora_rank`
- `r4`, `r8`, `r16`, `r32`, `r64`

### `learning_rate`
- `lr1e-05`, `lr5e-05`, `lr0.0001`, `lr0.0002`, `lr0.0005`

### `grpo_beta`
- `beta0.01`, `beta0.02`, `beta0.04`, `beta0.08`, `beta0.16`

### `batch_size`
- `bs4`, `bs8`, `bs16`, `bs32`

## Configuration Files

Create a JSON file with TrainingConfig structure:

```json
{
  "base_model": "gemma3-270m",
  "epochs": 3,
  "seed": 42,
  "lora": {
    "r": 32,
    "alpha": 32,
    "dropout": 0.05
  },
  "optimizer": {
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1
  },
  "grpo": {
    "beta": 0.02,
    "num_generations": 6,
    "reward": {
      "weights": {
        "format_weight": 0.15,
        "schema_weight": 0.35,
        "audio_weight": 0.5
      }
    }
  }
}
```

## Prompt Ablations

Use the prompt registry to swap prompts without code changes:

```python
from common.prompt_registry import get_prompt, register_prompt, PromptVersion

# Get current prompt
current = get_prompt("system_v1")
print(current.template)

# Register a new variant
register_prompt(PromptVersion(
    name="system_v2_detailed",
    version="2.0.0",
    description="More detailed system prompt",
    template="You are an expert ambient music composer...",
))

# Use in training config
config = TrainingConfig(
    system_prompt=get_prompt("system_v2_detailed").template
)
```

## Summary of Configurable Parameters

### Prompts
- `TrainingConfig.system_prompt` - System prompt for training
- `PromptConfig.vibe_extraction` - Vibe extraction prompt version
- `PromptConfig.config_generation` - Config generation prompt version

### Models
- `TrainingConfig.base_model` - Base model (gemma3-270m, qwen3-600m, or HF path)
- `LoRAConfig.*` - All LoRA hyperparameters

### Rewards
- `RewardWeights.*` - All reward component weights
- `RewardConfig.reward_type` - "clap", "schema_only", or "custom"
- `RewardConfig.custom_scorer_path` - Path to custom scorer function
- `ClapScorerConfig.*` - CLAP scoring parameters

### Training
- `OptimizerConfig.*` - LR, scheduler, warmup, weight decay
- `BatchConfig.*` - Batch size, gradient accumulation
- `GRPOConfig.*` - num_generations, beta
- `NoiseConfig.*` - error_rate, force_augmentation
- `DataConfig.*` - Field names, max_seq_length

### Infrastructure
- `InfraConfig.*` - GPU, timeout, retries
- `LoggingConfig.*` - logging_steps, save_strategy, W&B
