# Data Work

This folder is intended for researchers who want to generate data, train models, or otherwise
run data preparation scripts outside of the core LatentScore package.

## Environment setup (separate from core LatentScore)

This data pipeline uses a dedicated conda environment to avoid pulling data-only dependencies
into the main LatentScore install.

```bash
conda env create -f data_work/environment.yml
conda activate latentscore-data
```

You can also use the local venv helper:

```bash
./data_work/setup_python_env.sh
source data_work/.venv/bin/activate
```

Or install manually:

```bash
python3 -m venv data_work/.venv
source data_work/.venv/bin/activate
pip install -r data_work/requirements.txt
```

## Scripts

Prefer the numbered modules when running from the project root:
`python -m data_work.01_download_base_data`, `data_work.02a_extract_vibes`,
`data_work.02b_generate_configs`, `data_work.02c_score_configs`, `data_work.03_modal_train`,
`data_work.04_clap_benchmark`, `data_work.05_export_models`.

### `01_download_base_data`

Downloads Common Pile datasets, samples 1,000 texts from each, and writes JSONL files with
standardized fields (`created`, `metadata`, `dataset`, `id_in_dataset`, `text`). The script
prints approximate download sizes and prompts for confirmation before downloading.

Usage example:

```bash
python -m data_work.01_download_base_data \
  --seed 123 \
  --sample-size 1000 \
  --output-dir data_work/.outputs
```

### `02a_extract_vibes`

Extracts structured vibe objects from raw texts using a cheap LLM (vibe extraction is
straightforward and doesn't need SOTA intelligence). This is the first step of a two-step
pipeline that separates vibe extraction from config generation.

Key features:
- Extracts scene vibes, character vibes, and tags per text
- Injects deterministic noise into ~15% of vibes for robustness training
- **Dedupes on vibe content** (not raw text) - two different books about "love" will be
  deduped if their vibes are similar
- Splits into SFT-Train/SFT-Val/GRPO/TEST with proper ordering (eval sets first)
- **Diversity sampling for GRPO** - uses farthest-point algorithm on vibe embeddings
- Writes incrementally to prevent data loss on crash
- Uses SQLite caching for resumability

Usage example:

```bash
python -m data_work.02a_extract_vibes \
  --input-dir data_work/.outputs \
  --output-dir data_work/.vibes \
  --model openrouter/openai/gpt-oss-120b \
  --seed 42 \
  --error-rate 0.15 \
  --dedupe-threshold 0.95
```

### `02b_generate_configs`

Generates music configs from vibes using a SOTA model (config generation requires understanding
musical semantics and producing valid JSON matching strict schema). This is the second step
of the two-step pipeline.

Key features:
- **Best-of-N sampling** (default N=5) - generates multiple config candidates per vibe
- Temperature 0.8 for diversity in candidates
- Scores each candidate on: `format_valid`, `schema_valid`, `palette_valid`
- Stores all N candidates + scores for analysis
- **Validation-only selection**: picks first valid config (02c refines with CLAP quality scores)
- Role-based prompting: system prompt holds instructions + schema, user message holds `<vibe>...</vibe>`
- LiteLLM prompt caching enabled on the system message by default (disable with `--no-prompt-caching`)
- Optional batching via `--batch-size` and `--batch-wait-ms`
- Writes incrementally with resume support (`--resume` flag)
- Uses SQLite caching

Usage example:

```bash
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --model gemini/gemini-3-flash-preview \
  --num-candidates 3 \
  --temperature 0.8 \
  --max-concurrency 10 \
  --max-qps 5.0 \
  --seed 42
```

### `02c_score_configs`

Scores generated configs using multiple scoring methods and performs **quality-based Best-of-N
selection**. This step scores ALL valid candidates from 02b (not just the pre-selected one)
and re-selects the winner based on the highest quality score.

Key features:
- **Quality-based Best-of-N selection**: Scores all N candidates and picks the best one
- 02b does validation-only selection (first valid); 02c refines with CLAP-based quality scores
- Supports multiple scorers: CLAP, LLM-as-judge, custom Python functions
- Primary scorer (first in list) determines Best-of-N winner selection
- Stores per-candidate scores in `candidate_scores` for analysis
- Updates `best_index` and `config_payload` to the actual quality-based winner
- All scorers must return a `final_score` (enforced by `ScoreResult` protocol)
- Writes incrementally with resume support
- TQDM progress bars for monitoring

Usage example:

```bash
python -m data_work.02c_score_configs \
  --input-dir data_work/.processed \
  --output-dir data_work/.scored \
  --scorers 'clap,llm_judge' \
  --env-file .env
```

Custom scorer example:

```bash
python -m data_work.02c_score_configs \
  --input-dir data_work/.processed \
  --output-dir data_work/.scored \
  --scorers 'my_scorer.py:score_fn' \
  --overwrite
```

Custom scorers must have signature `def score_fn(vibe: str, config: dict) -> dict` and
return a dict containing at least `{"final_score": float}` (higher = better, unbounded).

## CLI usage

### Two-step pipeline (recommended)

**Step 1: Extract vibes** (uses cheap model):

```bash
python -m data_work.02a_extract_vibes \
  --input-dir data_work/.outputs \
  --output-dir data_work/.vibes \
  --env-file .env \
  --model openrouter/openai/gpt-oss-120b \
  --api-key-env OPENROUTER_API_KEY \
  --seed 42
```

**Step 2: Generate configs** (Gemini 3 Flash - performs as well as Opus at ~200x lower cost):

```bash
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --env-file .env \
  --model gemini/gemini-3-flash-preview \
  --api-key-env GEMINI_API_KEY \
  --num-candidates 3 \
  --max-concurrency 10 \
  --max-qps 5.0
```

Quick E2E smoke run:

```bash
# Step 1: Extract vibes (limit 15 input records = 0.5% of 3000)
python -m data_work.02a_extract_vibes \
  --input-dir data_work/.outputs \
  --output-dir data_work/.vibes \
  --env-file .env \
  --api-key-env OPENROUTER_API_KEY \
  --limit 15

# Step 2: Generate configs (limit 5 rows)
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --env-file .env \
  --api-key-env ANTHROPIC_API_KEY \
  --limit 5

# Step 3 (optional): Score configs with custom scorer
python -m data_work.02c_score_configs \
  --input-dir data_work/.processed \
  --output-dir data_work/.scored \
  --scorers 'data_work/test_scorer.py:simple_score' \
  --overwrite
```

Resume interrupted config generation:

```bash
python -m data_work.02b_generate_configs \
  --input-dir data_work/.vibes \
  --output-dir data_work/.processed \
  --env-file .env \
  --resume
```

Show all options:

```bash
python -m data_work.02a_extract_vibes --help
python -m data_work.02b_generate_configs --help
```

### `debug_vibe_noisy`

Quick sanity check for `vibe_noisy` differences in existing outputs plus an
nlpaug noise demo.

```bash
python -m data_work.lib.debug_vibe_noisy \
  --input data_work/.processed_smoke2/SFT-Train.jsonl \
  --limit 200
```

### `03_modal_train`

Launches Modal-based SFT + GRPO training for tiny models. Training defaults to
the noisy vibe column (`vibe_noisy`) and full config payload (`config_payload`),
so the model learns to handle corrupted inputs while inference remains clean.
Training outputs are LoRA adapter directories; use `data_work.05_export_models`
to merge adapters into full-precision checkpoints when needed.
For GRPO, `--model` should point at the base HF repo (e.g. `unsloth/gemma-3-270m-it`),
and `--init-adapter-dir` should point at the SFT LoRA adapter. If the base model
is private or only available locally, merge the adapter first and pass the merged
checkpoint path instead so Modal can resolve it.
Use `--debug-prompts` (optionally with `--debug-prompts-max`) to log a few formatted
system/user prompt samples so you can verify chat roles and prompt content.

**Note:** GRPO runs are currently skipped due to compute constraints (see `METHODOLOGY.md`).

If `--download-dir` is set, outputs are downloaded into `<download-dir>/<output>`.
Use `--delete-remote` to remove the Modal volume output after a successful download.

#### Configuration System

Training hyperparameters are centralized in `common/training_config.py`. You can
override them via CLI flags, config files, or ablation presets.

**Config file:** Load a JSON file with `TrainingConfig` structure:

```bash
python -m data_work.03_modal_train sft \
  --config-file my_config.json \
  --data data_work/.processed/SFT-Train.jsonl \
  --output gemma3-sft \
  --epochs 1
```

**Ablation presets:** Use predefined configs for common ablation studies (requires `--advanced`):

```bash
python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir gemma3-sft \
  --output gemma3-grpo-beta02 \
  --epochs 1 \
  --ablation-preset grpo_beta:beta0.02
```

Available presets: `lora_rank:{r4,r8,r16,r32,r64}`, `learning_rate:{lr1e-05,...}`,
`grpo_beta:{beta0.01,...}`, `batch_size:{bs4,bs8,bs16,bs32}`.

**Reward weight overrides:** Customize GRPO reward weights (requires `--advanced`):

```bash
python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir gemma3-sft \
  --output gemma3-grpo-custom \
  --epochs 1 \
  --format-weight 0.3 \
  --schema-weight 0.4 \
  --audio-weight 0.3
```

See `docs/ablation-guide.md` for full parameter reference.

#### Usage Examples

```bash
python -m data_work.03_modal_train check-imports
```

```bash
python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output gemma3-sft \
  --epochs 1 \
  --max-seq-length 4096 \
  --download-dir data_work/.modal_outputs \
  --delete-remote
```

```bash
python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir gemma3-sft \
  --output gemma3-grpo \
  --epochs 1 \
  --max-seq-length 4096 \
  --download-dir data_work/.modal_outputs \
  --delete-remote
```

### `07_modal_infer_eval`

Runs Modal-based SFT inference with **batched, constrained decoding** (Outlines JSON schema).
Outputs a JSONL file under `/outputs/<name>` and downloads it locally to `data_work/.modal_outputs`.

Usage example (2026-01-27 run):

```bash
conda run -n latentscore-data python -m data_work.07_modal_infer_eval \
  --adapter prod-sft-gemma3-270m-v5 \
  --input data_work/2026-01-26_scored/SFT-Val.jsonl \
  --limit 100 \
  --prompt-field vibe_noisy \
  --score-vibe-field vibe_original \
  --do-sample \
  --max-retries 3 \
  --batch-size 16 \
  --log-first-n 10 \
  --log-every 10 \
  --output sftval-100-v5-infer-batch
```

Notes:
- Default GPU is H100 (override with `MODAL_GPU_TYPE` or `MODAL_GPU_COUNT` env vars).
- Use `--batch-size` for throughput and `--tqdm/--no-tqdm` for progress display.

### `09_render_audio_from_results`

Renders WAVs from inference JSONL outputs (config payloads).

Usage example (30s renders):

```bash
conda run -n latentscore-data python -m data_work.09_render_audio_from_results \
  --input data_work/.modal_outputs/sftval-100-v5-infer-batch \
  --output-dir data_work/.audio/sftval-100-v5-30s \
  --limit 100 \
  --duration 30
```

### `10_export_embedding_map`

Builds a lookup table of **vibe embeddings + best config payloads** from
scored split files. Uses a sentence‑transformers encoder to embed `vibe_original`
and writes a JSONL file that can be used for fast retrieval.

The canonical output is `vibe_and_embeddings_to_config_map.jsonl` (10,558 rows
with correct split assignments). Legacy `_progress_embeddings.jsonl` had empty
splits and should not be used.

Usage example:

```bash
conda run -n latentscore-data python -m data_work.10_export_embedding_map \
  --input data_work/2026-01-26_scored/_progress.jsonl \
  --output data_work/2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl \
  --batch-size 64
```

### `04_clap_benchmark`

Scores configs with LAION-CLAP. You can benchmark dataset configs, LiteLLM
models, and local HF models by supplying multiple sources.
Defaults to the clean vibe column (`vibe_original`) for inference.
Supports `--workers N` for multiprocess parallelism (splits rows across N
workers, each loading its own models). Tracks per-sample timing
(`config_gen_s`, `audio_synth_s`, `elapsed_s`) and a `success` flag per
result, with `success_rate` reported in the summary.

**Label syntax:** Arguments use `value:label` format where `:label` is an optional
display name for the benchmark output. For example:
- `--dataset-field config_payload:synthetic` → uses `config_payload` column, labeled "synthetic"
- `--litellm-model openrouter/openai/gpt-oss-20b:teacher` → benchmarks model, labeled "teacher"
- `--local-model /path/to/model:finetuned` → benchmarks local model, labeled "finetuned"

Usage example (synthetic configs + teacher model):

```bash
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --dataset-field config_payload:synthetic \
  --litellm-model openrouter/openai/gpt-oss-20b:teacher \
  --api-key-env OPENROUTER_API_KEY \
  --env-file .env \
  --limit 10
```

Usage example (local model):

```bash
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --local-model /path/to/sft-model:finetuned \
  --limit 10
```

### `05_export_models`

Merges LoRA adapters into full-precision checkpoints and optionally exports
GGUF/MLC artifacts. The default merged output path is `data_work/.exports/combined-model`.

### Notes on defaults

**Vibe extraction (02a):**
- Recommended model: `openrouter/openai/gpt-oss-120b` (cheap, better quality than 20b)
- Default max input tokens: 80,000 (whitespace-tokenized)
- Truncation preferred over exclusion to avoid selection bias toward shorter texts
- Dedupe threshold: 0.95 (vibes with similarity ≥ 0.95 are considered duplicates)

**Config generation (02b):**
- Recommended model: `gemini/gemini-3-flash-preview` (performs as well as Opus at ~200x lower cost)
- Default candidates: 3 (Best-of-N sampling)
- Default temperature: 0.8 (for diversity in candidates)

**API keys:**
- Native providers (`anthropic/`, `gemini/`) auto-detect API keys from
  environment variables via LiteLLM, so `--api-key-env` is optional for them.
- Other providers require explicit `--api-key-env`:
  - `openai/...` models: `--api-key-env OPENAI_API_KEY`
  - `openrouter/...` models: `--api-key-env OPENROUTER_API_KEY`
- The script will error if a required env var is not set.

### Output format

**Vibe extraction output (02a):**

Each split produces `SFT-Train.jsonl`, `SFT-Val.jsonl`, `GRPO.jsonl`, `TEST.jsonl` with one
row per vibe level (xl/lg/m/sm/xs). Rows include:

- `dataset`, `id_in_dataset`, `split`, `vibe_index`, `text_page`
- `vibe_scope` (`scene` or `character`) and optional `character_name`
- `vibe_level`, `vibe_original`, `vibe_noisy`
- `tags_original`, `tags_noisy`
- `vibe_model`

The output directory also contains `run_config.json` with the non-secret configuration used
to produce the dataset.

**Config generation output (02b):**

Takes vibe JSONL from 02a and adds config fields. Each row includes all vibe fields plus:

- `config_model`: Model used for config generation
- `config_candidates`: Array of N config candidates (each with `thinking`, `title`, `config`, `palettes`)
- `scores`: Dict with binary scores per candidate:
  - `format_valid`: 1 if parses as JSON
  - `schema_valid`: 1 if validates against MusicConfigPromptPayload
  - `palette_valid`: 1 if exactly 3 palettes with 5 colors each
- `best_index`: Index of the selected best candidate
- `config_payload`: The selected best config
- `config_error`: Error message if all candidates failed

### Helper modules

Non-entrypoint helpers live under `data_work/lib/` and are imported by the entrypoints.

Key modules:
- `scoring_types.py`: Defines `ScoreResult` protocol for strict typing across all scorers
- `clap_scorer.py`: CLAP-based audio-text similarity scoring
- `llm_scorer.py`: LLM-as-judge multimodal scoring
- `rewards.py`: Reward computation for GRPO training
- `eval_schema.py`: Evaluation result schemas

### Scoring types and the `ScoreResult` protocol

All scoring results in data_work implement the `ScoreResult` protocol, which requires a
`final_score` property returning a float (higher = better, unbounded). Only relative ordering
matters for GRPO and Best-of-N selection. This ensures consistent interfaces across CLAP
scoring, LLM-as-judge scoring, reward computation, and custom scorers.

```python
from data_work.lib.scoring_types import ScoreResult, validate_score_result

# All these implement ScoreResult:
from data_work.lib.clap_scorer import ClapScore       # final_score = final_reward
from data_work.lib.llm_scorer import LLMScoreResult   # final_score = harmonic mean (vibe_match + audio_quality)
from data_work.lib.rewards import RewardBreakdown     # final_score = total
from data_work.lib.eval_schema import EvalResult      # final_score = best available

# Validate any scorer result
validate_score_result(my_result, source="my_scorer")  # Raises if invalid
```

Custom scorers for 02c_score_configs must return a dict with `final_score` key. The script
validates this using `DictScoreResult` wrapper.
