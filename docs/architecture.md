# Architecture

## Data Work Architecture Map

The `data_work/` folder is a standalone data pipeline outside the core package.

**Pipeline flow**
- `data_work/01_download_base_data.py` → download + sample raw text → `data_work/.outputs/`
- `data_work/02_process_base_data.py` → schema validation + vibe/config generation → `data_work/.processed/`
- `data_work/03_modal_train.py` → Modal SFT/GRPO training → `data_work/.modal_outputs/`
- `data_work/04_clap_benchmark.py` → CLAP scoring + model comparisons
- `data_work/05_export_models.py` → merge LoRA adapters → `data_work/.exports/`

**Supporting modules**
- `data_work/lib/config_io.py` / `jsonl_io.py` handle config + JSONL IO
- `data_work/lib/llm_client.py` / `llm_cache.py` wrap LiteLLM requests + caching
- `data_work/lib/record_builder.py` assembles dataset rows per vibe level
- `data_work/lib/vibe_schema.py` / `music_schema.py` define data contracts
- `data_work/lib/dedupe.py`, `rewards.py`, `resilience.py`, `music_prompt.py` provide
  embedding dedupe, reward helpers, retry logic, and prompt helpers

**Environment**
- `data_work/environment.yml` and `data_work/requirements.txt` define the data-only deps
- `data_work/setup_python_env.sh` provisions a local venv
- Prefer the `latentscore-data` environment for `data_work/` scripts and tooling
