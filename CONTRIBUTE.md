# Contributing to LatentScore

## Setting Up Your Environment

### 1. Install Conda (recommended)

Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you don't have it already. The project requires Python 3.11–3.12. For artifact review and the full demo stack, use Docker.

### 2. Create the environment

```bash
conda create -n latentscore python=3.11 -y
conda activate latentscore
```

(`environment.yml` is also present for `conda env create -f environment.yml`, but the explicit two-liner above is the same thing and easier to update.)

### 3. Install the package in editable mode with dev extras

This is the canonical contributor install. It pulls everything you need
to run the full test suite, build the docs, exercise the demo, and
iterate on the library:

```bash
pip install -e ".[external,heavy,expressive,dev]"
```

The extras:

- `[dev]` — pytest, ruff, pyright (the `make check` toolchain).
- `[external]` — LiteLLM bridge for the `external:<model>` model spec.
- `[heavy]` — laion-clap for `fast_heavy` retrieval.
- `[expressive]` — local LLM (270M Gemma 3). Currently runs through the CPU `transformers` backend on every platform; MLX / llama-cpp markers exist in `pyproject.toml` but aren't wired into the runtime yet, so renders take ~30–100&nbsp;s.

For a lighter setup that skips local LLM tooling:

```bash
pip install -e ".[external,heavy,dev]"
```

### 4. System prerequisites for audio playback

**macOS:**

```bash
brew install sox
```

**Linux:**

```bash
sudo apt-get update
sudo apt-get install -y sox libasound2-dev
```

**Windows:** install SoX with your preferred package manager (e.g.
`winget install sox`). If you're using WSL, follow the Linux steps inside
the WSL distro.

### 5. Next time

You only set this up once. Future sessions just need:

```bash
conda activate latentscore
```

## Development Loop

Before opening a pull request, run the full suite:

```bash
make check     # ruff lint + format + pyright + pytest
```

Style and review rules live in
[`docs/contribute/coding-guidelines.md`](docs/contribute/coding-guidelines.md);
code samples illustrating the conventions live in
[`docs/contribute/examples.md`](docs/contribute/examples.md).

## Working on the demo

The interactive demo (`demo/`) is its own React + FastAPI artifact. See
[`demo/README.md`](demo/README.md) for backend-only, frontend-only, and
Docker Compose setup paths.

## Working on the data pipeline

The research pipeline lives under `data_work/` and uses a separate
conda env (`latentscore-data`). See [`data_work/README.md`](data_work/README.md).
