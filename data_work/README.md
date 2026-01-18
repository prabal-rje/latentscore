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
`python -m data_work.01_download_base_data`, `data_work.02_process_base_data`,
`data_work.03_modal_train`, `data_work.04_clap_benchmark`, `data_work.05_export_models`.

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

### `02_process_base_data`

Validates that downloaded JSONL files match the expected schema, splits them into
SFT-Train/SFT-Val/GRPO/TEST, and uses LiteLLM to:

- extract structured vibe objects per text
- inject deterministic noise into ~15 percent of vibes
- generate music configs per vibe level with a separate model
- dedupe near-identical vibes per dataset using embeddings
- write one JSONL per split + a `run_config.json` alongside the outputs

The script prints a configuration hash at startup and uses SQLite caching to make repeated
runs deterministic and fast.

Noise injection requires `nlpaug` (the run exits if it's missing and `--error-rate > 0`).
If a split produces zero noisy rows, one row is force-noised to ensure the pipeline is
exercised.

## CLI usage

Basic processing (uses `OPENROUTER_API_KEY` from `.env` at the repo root):

```bash
python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.processed \
  --env-file .env
```

Quick E2E smoke run (10 rows total, 2 per split):

```bash
python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.processed \
  --env-file .env \
  --limit-per-split 2
```

Run from a saved config file (auto-generated in the output folder):

```bash
python -m data_work.02_process_base_data \
  --config-file data_work/.processed/run_config.json \
  --input-dir data_work/.outputs \
  --output-dir data_work/.processed
```

Model kwargs (pass JSON strings through to LiteLLM):

```bash
python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.processed \
  --env-file .env \
  --model-kwargs '{"timeout": 120}' \
  --config-model-kwargs '{"max_tokens": 512}'
```

Show advanced options (rate limits, retries, dedupe thresholds, etc.):

```bash
python -m data_work.02_process_base_data --extra-help
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

### `04_clap_benchmark`

Scores configs with LAION-CLAP. You can benchmark dataset configs, LiteLLM
models, and local HF models by supplying multiple sources.
Defaults to the clean vibe column (`vibe_original`) for inference.

Usage example (synthetic configs + teacher model):

```bash
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --dataset-field config_payload:synthetic \
  --litellm-model openrouter/openai/gpt-oss-20b:teacher \
  --env-file .env \
  --limit 10
```

Usage example (local model):

```bash
python -m data_work.04_clap_benchmark \
  --input data_work/.processed/TEST.jsonl \
  --local-model /path/to/sft-model:local \
  --limit 10
```

### `05_export_models`

Merges LoRA adapters into full-precision checkpoints and optionally exports
GGUF/MLC artifacts. The default merged output path is `data_work/.exports/combined-model`.

### Notes on defaults

- Default vibe model: `openrouter/openai/gpt-oss-20b`
- Default config model: `openrouter/openai/gpt-oss-20b`
- Default context window: `--max-input-tokens 100000`
- OpenRouter GPT-OSS currently caps at ~131k tokens; set `--max-input-tokens 131072` if needed.
- If you override the default vibe model, the script logs a warning with the current
  context size so you can adjust it for the new model.
- If `--api-key` is not provided, the script falls back to `OPENROUTER_API_KEY` and
  logs a masked key preview (prefix/suffix only).

### Output format

Each split produces `SFT-Train.jsonl`, `SFT-Val.jsonl`, `GRPO.jsonl`, `TEST.jsonl` with one
row per vibe level (xl/lg/m/sm/xs). Rows include:

- `dataset`, `id_in_dataset`, `split`, `vibe_index`, `text_page`
- `vibe_scope` (`scene` or `character`) and optional `character_name`
- `vibe_level`, `vibe_original`, `vibe_noisy`
- `tags_original`, `tags_noisy`
- `vibe_model`, `config_model`, `config_payload`
- `config_error` if config generation failed

The output directory also contains `run_config.json` with the non-secret configuration used
to produce the dataset.

### Helper modules

Non-entrypoint helpers live under `data_work/lib/` and are imported by the entrypoints.
