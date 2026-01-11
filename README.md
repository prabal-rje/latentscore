# LatentScore

LatentScore is a local-first audio synthesis library. It turns short text vibes or
structured configs into short audio clips, with a DX layer for rendering, streaming,
and composing playlists. The repo also contains a separate data pipeline for
building datasets, benchmarking configs, and training small models.

## Repo layout

- `latentscore/`: core library + CLI demo
- `data_work/`: data prep, benchmarking, and Modal training workflows
- `docs/`: API/DX docs and examples
- `tests/`: unit + smoke tests

## Install

### Conda

```bash
conda env create -f environment.yml
conda activate latentscore
pip install -e .
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Library usage

```python
import latentscore as ls

audio = ls.render("warm sunrise over water")
audio.play()
audio.save(".examples/quickstart.wav")
```

- Local-first by default (no API keys required).
- Streaming supports speculative preview while slower models load; see `docs/latentscore-dx.md`.
- Expressive local model download: `latentscore download expressive`.
- Health check: `latentscore doctor`.

## Demo CLI

```bash
python -m latentscore.demo --model fast --save
```

Outputs land in `.examples/` (gitignored). For external LLM demos, create a `.env`
file at the repo root and set `GEMINI_API_KEY` (or override via `--api-key`).

## Data work

The data pipeline uses its own environment and scripts. See `data_work/README.md`
for full details, including Modal training and benchmarking.

## Tiny smoke commands

These are minimal, small-footprint runs that exercise each outward-facing script.

```bash
# Demo
python -m latentscore.demo --model fast --save

# Data work: download tiny samples
python -m data_work.01_download_base_data \
  --seed 1 \
  --sample-size 1 \
  --output-dir data_work/.outputs_smoke

# Data work: process tiny samples (requires OPENROUTER_API_KEY in .env)
python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs_smoke \
  --output-dir data_work/.processed_smoke \
  --env-file .env \
  --limit-per-split 1 \
  --error-rate 0

# Data work: debug noisy vibes
python -m data_work.lib.debug_vibe_noisy \
  --input data_work/.processed_smoke/SFT-Train.jsonl \
  --limit 5

# Data work: Modal import/mount check (requires Modal credentials)
python -m data_work.03_modal_train check-imports

# Data work: CLAP benchmark on dataset configs (requires laion-clap + torchvision)
python -m data_work.04_clap_benchmark \
  --input data_work/.processed_smoke/SFT-Train.jsonl \
  --dataset-field config_payload:synthetic \
  --limit 1
```

## Tooling

- `make check` runs ruff, pyright, and pytest.
- `make format` applies `ruff format`.

## Contributing

See `CONTRIBUTE.md` for environment setup and contribution guidelines.
