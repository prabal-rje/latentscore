# LatentScore

> ⚠️ **Alpha**: This library is under active development. API may change between versions.

Generate ambient music from text descriptions. Locally. No GPU required.

Read more about how it works [here](https://substack.com/home/post/p-184245090).

```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

## Repo layout

- `latentscore/`: core library + CLI demo
- `data_work/`: data prep, benchmarking, and Modal training workflows
- `docs/`: API/DX docs and examples
- `tests/`: unit + smoke tests

## Install

### Conda

```bash
# download the repo
git clone https://github.com/prabal-rje/latentscore
cd latentscore

# create the env, install dependencies
conda env create -f environment.yml
conda activate latentscore

# install latentscore
pip install -e .
```

### Pip

```bash
# download the repo
git clone https://github.com/prabal-rje/latentscore
cd latentscore

# create the env
python -m venv .venv
source .venv/bin/activate

# install dependencies, install latentscore
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

Live streaming example:

```bash
python -m latentscore.demo --live --model fast
```

## Data work

See `data_work/README.md` for environment setup, pipeline scripts, benchmarks, and training.

## Tooling

- `make check` runs ruff, pyright, and pytest.
- `make format` applies `ruff format`.

## Contributing

See `CONTRIBUTE.md` for environment setup and contribution guidelines.
