# LatentScore

[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart.ipynb) [![Listen to Demo](https://img.shields.io/badge/▶_Listen_to_Demo-latentscore.com-8A2BE2?style=flat)](https://latentscore.com/demo) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](LICENSE)

[![Presenting at SIGGRAPH 2026](https://img.shields.io/badge/Presenting_at-SIGGRAPH_2026-B91C1C?style=flat)](https://s2026.siggraph.org/program/talks/) [![Presenting at NIME 2026](https://img.shields.io/badge/Presenting_at-NIME_2026-5B21B6?style=flat)](https://nime2026.org/)

**Generate ambient music from text. Locally. No GPU required.** &mdash; [Read more about how it works](https://prabal.ca/posts/latentscore-research/).

https://private-user-images.githubusercontent.com/140295281/557606724-22889dcc-9287-4712-8ffb-ec19381444c9.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzI1NTQ3OTcsIm5iZiI6MTc3MjU1NDQ5NywicGF0aCI6Ii8xNDAyOTUyODEvNTU3NjA2NzI0LTIyODg5ZGNjLTkyODctNDcxMi04ZmZiLWVjMTkzODE0NDRjOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMwM1QxNjE0NTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03Y2IzMzNlMzhiYTAyZDRmMjE0OWY4ZmNiMmUxNzE4ZTE0YjMzZWU0OGE5MmQyOWYzMzYzZWViYmY5MWNjZWM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VDGlSlEVudf_rmavZpjUqkImO0O1gUYxEqDvDx6PQNo

---

## Try it now

**Four ways:**

- 🎧 **Hear it now** &mdash; [latentscore.com/demo](https://latentscore.com/demo). Browser, no install.
- 🐳 **Run the demo locally** &mdash; `docker compose` on any OS. See [Try the demo](#try-the-demo).
- 📓 **Try the SDK in Colab** &mdash; [open the quickstart notebook](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart.ipynb). No install, runs in your browser.
- 🛠 **Build with it locally** &mdash; `pip install latentscore` on macOS, Linux, or Windows WSL2. See [Install the SDK](#install-the-sdk).

```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

That's it. One line. You get audio playing on your speakers.

---

## Try the demo

**No install needed:** open **[latentscore.com/demo](https://latentscore.com/demo)** in your browser.

**Run it locally** &mdash; works on macOS, Linux, or Windows (any flavor) with [Docker Desktop](https://www.docker.com/products/docker-desktop/):

```bash
docker compose -f demo/docker-compose.yml up --build
```

Then open [`localhost:3002`](http://localhost:3002). More details in [demo/](demo/).

---

## Contents

- [Try it now](#try-it-now) &mdash; four ways to get going (browser, Docker, Colab, pip)
- [Try the demo](#try-the-demo) &mdash; deeper Docker setup
- [Install the SDK](#install-the-sdk) &mdash; pip, 30 seconds
- [Quick start](#quick-start) &mdash; Python in 5 lines
- [Controlling the sound](#controlling-the-sound) &mdash; `MusicConfig` parameters
- [Documentation](docs/latentscore-dx.md) &mdash; streaming, live playlists, async, bring-your-own-LLM
- [How it works](docs/architecture.md#how-it-works) &mdash; embedding retrieval, no LLM hallucinations
- [FAQ](docs/FAQ.md) &mdash; common questions
- [Citation](#citation) &mdash; SIGGRAPH Talks '26 BibTeX

---

## Install the SDK

### Requirements

- **OS** &mdash; macOS, Linux, or Windows **ONLY** via [WSL2](docs/FAQ.md#can-i-run-this-on-windows). For the web UI on any OS, use the [Docker demo](#try-the-demo) instead.
- **Python 3.10&ndash;3.12** &mdash; we test against [3.12](https://www.python.org/downloads/release/python-3120/) (matches our Docker image). Or use [conda](https://docs.conda.io/projects/miniconda/en/latest/) for environment management.

### Install

With **venv** (regular Python):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install latentscore
```

With **conda**:

```bash
conda create -n latentscore python=3.12 -y
conda activate latentscore
pip install latentscore
```

### What you get

`pip install latentscore` gives you text prompts (`ls.render("vibe")`), `MusicConfig` rendering, and local playback &mdash; the [Quick Start](#quick-start) below.

Optional extras &mdash; install with `pip install "latentscore[<extra>]"`:

| Extra | Adds |
|---|---|
| `external` | bring-your-own hosted LLM via [LiteLLM](https://docs.litellm.ai/) (Anthropic, Gemini, OpenAI, &hellip;) |
| `heavy` | CLAP audio-based retrieval (`fast_heavy` model) |
| `expressive` | local LLM inference |

### Verify your install

```bash
latentscore doctor --strict --offline
```

Exits non-zero with a clear hint if anything's broken. Add `--json` for machine-readable output.


---

## CLI

```bash
# Verify your install
latentscore doctor                       # human-readable summary
latentscore doctor --strict --offline    # nonzero exit if anything's broken
latentscore doctor --json                # machine-readable output

# Pre-download model assets (otherwise the first render call appears to hang)
latentscore download fast                # ~90 MB, MiniLM embedding model
latentscore download fast_heavy          # ~1.8 GB, LAION-CLAP weights

# Render a sample clip
latentscore demo                         # play a short ambient clip
latentscore demo --duration 30 --output ambient.wav   # 30 seconds, save to file
```

---

## Quick Start

### Render and play

```python
import latentscore as ls

# Optional one-time setup: pre-download the embedding model (~90 MB) so
# the first render() call doesn't appear to hang. The download happens
# on the first render anyway; this just makes it explicit and visible.
ls.prefetch("fast")

audio = ls.render("warm sunset over water", duration=10.0)
audio.play()              # plays on your speakers
audio.save("output.wav")  # save to WAV
```

### Different vibes

```python
ls.render("jazz cafe at midnight").play()
ls.render("thunderstorm on a tin roof").play()
ls.render("lo-fi study beats").play()
```

---

## Controlling the sound

Beyond text prompts, you can drive synthesis directly:

```python
import latentscore as ls

# Full control: build a MusicConfig with human-readable labels
config = ls.MusicConfig(
    tempo="slow", mode="dorian", root="d",
    bass="drone", pad="ambient_drift", melody="contemplative",
    rhythm="minimal", texture="shimmer", echo="heavy",
    density=3, brightness="dark", space="vast",
)
ls.render(config, duration=10.0).play()

# Or start from a vibe and nudge specific parameters
ls.render(
    "morning coffee shop",
    update=ls.MusicConfigUpdate(brightness="very_bright", rhythm="electronic"),
).play()
```

See the [full documentation](docs/latentscore-dx.md) for the parameter
reference, relative-step updates, streaming, live playlists, async
API, and bring-your-own-LLM cookbook.

---

## Read more

- [Documentation](docs/latentscore-dx.md) &mdash; parameter reference, streaming, async API, bring-your-own-LLM
- [How it works](docs/architecture.md) &mdash; embedding retrieval, explained
- [FAQ](docs/FAQ.md) &mdash; first-call hang, system deps, Windows, citation, …
- [Research pipeline](data_work/README.md) &mdash; how the dataset was built
- [Contributing](CONTRIBUTE.md) &mdash; setup + [style rules](docs/contribute/coding-guidelines.md)
- [Demo](demo/) &mdash; run the web demo locally

---


## Citation

If you use LatentScore in your research, please cite the SIGGRAPH Talks '26 paper:

```bibtex
@inproceedings{gupta2026latentscore,
  author    = {Gupta, Prabal},
  title     = {LatentScore: Sketching Soundscapes with LLM-Distilled Retrieval for Procedural Synthesis},
  booktitle = {SIGGRAPH Talks '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799818.3812120}
}
```

---

## License

LatentScore is released under the [Apache License 2.0](LICENSE).
