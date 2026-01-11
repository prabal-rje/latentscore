# Data Work

This folder is intended for researchers who want to generate data, train models, or otherwise
run data preparation scripts outside of the core LatentScore package.

## Environment setup

Create a local Python virtual environment and install dependencies:

```bash
./data_work/setup_python_env.sh
source data_work/.venv/bin/activate
```

You can also do it manually:

```bash
python3 -m venv data_work/.venv
source data_work/.venv/bin/activate
pip install -r data_work/requirements.txt
```

## Scripts

### `0_download_base_data.py`

Downloads Common Pile datasets, samples 1,000 texts from each, and writes JSONL files with
standardized fields (`created`, `metadata`, `dataset`, `id_in_dataset`, `text`). The script
prints approximate download sizes and prompts for confirmation before downloading.

Usage example:

```bash
python data_work/0_download_base_data.py --seed 123 --sample-size 1000 --output-dir data_work/.outputs
```

### `1_process_base_data.py`

Validates that downloaded JSONL files match the expected schema, splits them into
SFT-Train/SFT-Val/GRPO/TEST, and uses LiteLLM to generate vibe objects per text with
caching, rate limiting, and deterministic noise injection.

Usage example:

```bash
python data_work/1_process_base_data.py \
  --input-dir data_work/.outputs \
  --output-dir data_work/.processed \
  --model openrouter/openai/gpt-oss-20b \
  --env-file examples/.env
```

Tip: use `--limit-per-split 10` for quick E2E smoke runs, and tune `--max-concurrency`
and `--max-qps` to respect rate limits.
Note: OpenRouter's `openai/gpt-oss-20b` currently caps at ~131k tokens, so set
`--max-input-tokens 131072` to avoid 400 errors when using the default 200k limit.

Use `-h` on each script for detailed CLI help and arguments.
