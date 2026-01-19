# LatentScore

> ⚠️ **Alpha**: This library is under active development. API may change between versions.

Generate ambient music from text descriptions. Locally. No GPU required.

Read more about how it works [here](https://substack.com/home/post/p-184245090).
```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

## Install

### Conda
```bash
conda create -n latentscore python=3.10
conda activate latentscore
conda install pip

pip install latentscore
```

### Pip
```bash
python -m venv .venv
source .venv/bin/activate

pip install latentscore
```

> Requires Python 3.10. If you don't have it: `brew install python@3.10` (macOS) or `pyenv install 3.10`

## Usage
```python
import latentscore as ls

# Render and play
audio = ls.render("warm sunrise over water")
audio.play()
audio.save("output.wav")
```

### Streaming
```python
import latentscore as ls

# Stream a single vibe
ls.stream("warm sunset over water", duration=120).play()

# Stream multiple vibes with crossfade
ls.stream(
    "morning coffee",
    "afternoon focus", 
    "evening wind-down",
    duration=60,
    transition=5.0,
).play()
```

### Async Streaming
```python
import latentscore as ls
import asyncio

async def main():
    items = [
        ls.Streamable(content="morning coffee", duration=30),
        ls.Streamable(content="afternoon focus", duration=30),
    ]
    async for chunk in ls.astream(items):
        # Process chunks as they arrive
        print(f"Got {len(chunk)} samples")

asyncio.run(main())
```

### Playlists
```python
import latentscore as ls

playlist = ls.Playlist(tracks=(
    ls.Track(content="morning energy", duration=60),
    ls.Track(content="deep focus", duration=120),
    ls.Track(content="evening calm", duration=60),
))
playlist.play()
```

### Modes

- **fast** (default): Embedding lookup. Instant.
- **expressive**: Local LLM. Slower, more creative. Run `latentscore download expressive` first.
- **external**: Route through Claude, Gemini, etc. Best quality, needs API key.
```python
# Use expressive mode
ls.render("jazz cafe at midnight", model="expressive").play()

# Use external LLM
ls.render(
    "cyberpunk rain",
    model="external:gemini/gemini-3-pro-preview",
    api_key="..."
).play()
```

## CLI
```bash
latentscore demo                  # Generate and play a sample
latentscore download expressive   # Fetch local LLM weights
latentscore doctor                # Check setup
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the data_work pipeline map and environment notes.

---

## Research & Training Pipeline (`data_work/`)

The `data_work/` folder contains a complete research pipeline for training and evaluating vibe-to-music-config models. This is separate from the core library and is used for:

- Downloading and processing training data
- Training SFT and GRPO models on Modal
- Benchmarking with CLAP audio-text similarity
- Evaluating models against baselines

### Quick Start

```bash
# Activate the environment
conda activate latentscore

# 1. Download sample data (10 texts per dataset)
python -m data_work.01_download_base_data \
  --output-dir data_work/.outputs \
  --sample-size 10

# 2. Extract vibes (requires API key)
export OPENROUTER_API_KEY="your-key"
python -m data_work.02a_extract_vibes \
  --input-dir data_work/.outputs \
  --output-dir data_work/.vibes \
  --limit 5

# 3. Generate configs (requires API key)
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --limit 5

# 4. Run baseline evaluation (no API needed)
python -m data_work.06_eval_suite \
  --eval-set test_prompts \
  --baseline random \
  --baseline rule_based \
  -v
```

### Pipeline Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `01_download_base_data.py` | Download & sample text datasets | `--sample-size`, `--output-dir` |
| `02a_extract_vibes.py` | Extract vibes only | `--model`, `--limit` |
| `02b_generate_configs.py` | Generate configs (Best-of-N) | `--model`, `--num-candidates` |
| `03_modal_train.py` | Train SFT/GRPO on Modal cloud | `--config-file`, `--ablation-preset` |
| `04_clap_benchmark.py` | CLAP audio-text scoring | `--litellm-model`, `--local-model` |
| `05_export_models.py` | Merge LoRA adapters | `--input-dir`, `--output-dir` |
| `06_eval_suite.py` | Evaluate models vs baselines | `--baseline`, `--include-clap` |

### Training with Config Files

You can now use JSON config files instead of CLI args:

```bash
# Create a training config
cat > my_config.json << 'EOF'
{
  "base_model": "qwen3-600m",
  "lora": {"r": 32, "alpha": 32},
  "optimizer": {"learning_rate": 1e-4},
  "batch": {"batch_size": 8},
  "epochs": 3
}
EOF

# Train with config file (CLI args override config file)
python -m data_work.03_modal_train sft \
  --config-file my_config.json \
  --data data_work/.processed/SFT-Train.jsonl \
  --output my-sft-run
```

### Ablation Presets

Use predefined ablation presets for systematic experiments:

```bash
# LoRA rank ablation
python -m data_work.03_modal_train grpo \
  --ablation-preset lora_rank:r32 \
  --data data_work/.processed/GRPO.jsonl \
  --output grpo-r32

# Learning rate ablation
python -m data_work.03_modal_train sft \
  --ablation-preset learning_rate:lr1e-4 \
  --data data_work/.processed/SFT-Train.jsonl \
  --output sft-lr1e-4
```

Available presets: `lora_rank` (r4-r64), `learning_rate` (1e-5 to 5e-4), `grpo_beta` (0.01-0.16), `batch_size` (4-32)

### Evaluation Suite

Evaluate models against baselines:

```bash
# Create an eval set (JSONL format)
cat > data_work/eval_sets/my_eval.jsonl << 'EOF'
{"id": "001", "prompt": "warm sunset", "category": "short", "expected_fields": {"brightness": "medium"}, "difficulty": "easy", "source": "manual"}
{"id": "002", "prompt": "dark ambient", "category": "short", "expected_fields": {"brightness": "dark"}, "difficulty": "easy", "source": "manual"}
EOF

# Run evaluation with baselines
python -m data_work.06_eval_suite \
  --eval-set my_eval \
  --baseline random \
  --baseline rule_based \
  -v

# Evaluate a trained model
python -m data_work.06_eval_suite \
  --eval-set my_eval \
  --local-model path/to/model:my_model \
  --include-clap
```

### Baselines

Two baseline generators for comparison:

- **`random`**: Generates valid configs with random field values
- **`rule_based`**: Extracts keywords from prompts to set tempo, brightness, mode, etc.

```python
from data_work.lib.baselines import RandomBaseline, RuleBasedBaseline

# Generate a random config
random_gen = RandomBaseline(seed=42)
config = random_gen.generate("any vibe text")

# Generate a rule-based config
rule_gen = RuleBasedBaseline(seed=42)
config = rule_gen.generate("dark slow ambient drone")
print(config.config.tempo)      # "slow"
print(config.config.brightness) # "dark"
```

### CLAP Scoring

Benchmark audio-text alignment with LAION-CLAP:

```bash
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --limit 20 \
  --dataset-field config_payload:synthetic \
  --litellm-model openrouter/openai/gpt-4o-mini:gpt4o \
  --baseline random \
  --baseline rule_based
```

### Reward Functions

For GRPO training, configurable reward weights:

```bash
python -m data_work.03_modal_train grpo \
  --format-weight 0.3 \
  --schema-weight 0.5 \
  --audio-weight 0.2 \
  ...
```

### Full E2E Experiment Example

```bash
# 1. Download minimal data
python -m data_work.01_download_base_data \
  --output-dir data_work/.outputs \
  --sample-size 50

# 2. Extract vibes
python -m data_work.02a_extract_vibes \
  --input-dir data_work/.outputs \
  --output-dir data_work/.vibes \
  --limit 20

# 3. Generate configs
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --limit 20

# 4. Train SFT model on Modal
python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output sft-experiment \
  --epochs 1

# 5. Train GRPO model on Modal
python -m data_work.03_modal_train grpo \
  --model outputs/sft-experiment \
  --data data_work/.processed/GRPO.jsonl \
  --output grpo-experiment \
  --epochs 1

# 6. Export merged model
python -m data_work.05_export_models \
  --input-dir outputs/grpo-experiment \
  --output-dir exports/final-model

# 7. Benchmark with CLAP
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --local-model exports/final-model:trained \
  --baseline rule_based \
  --limit 50

# 7. Run eval suite
python -m data_work.06_eval_suite \
  --eval-set test_prompts \
  --local-model exports/final-model:trained \
  --baseline rule_based \
  --include-clap
```

### Library Modules

| Module | Purpose |
|--------|---------|
| `lib/baselines.py` | Random and rule-based baseline generators |
| `lib/rewards.py` | Reward functions for GRPO (format, schema, audio) |
| `lib/clap_scorer.py` | LAION-CLAP audio-text similarity scoring |
| `lib/eval_schema.py` | Evaluation prompt and result schemas |
| `lib/vibe_schema.py` | Vibe extraction schema (5-level descriptors) |
| `lib/music_schema.py` | Music config schema (re-exports from common) |
| `lib/llm_client.py` | LiteLLM wrapper with structured outputs |
| `lib/llm_cache.py` | SQLite caching for LLM responses |
| `lib/resilience.py` | Retry, rate limiting, error tracking |
| `lib/dedupe.py` | Embedding-based deduplication |

### Training Config Schema

The `TrainingConfig` in `common/training_config.py` defines all hyperparameters:

```python
from common.training_config import TrainingConfig, LoRAConfig

# Default config
config = TrainingConfig()

# Custom config
config = TrainingConfig(
    base_model="qwen3-600m",
    lora=LoRAConfig(r=32, alpha=32, dropout=0.1),
    epochs=5,
    seed=123,
)

# Save to JSON
config_json = config.model_dump_json(indent=2)
```

---

## Contributing

See `CONTRIBUTE.md` for environment setup and contribution guidelines.

See [`docs/coding-guidelines.md`](docs/coding-guidelines.md) for code style requirements.
