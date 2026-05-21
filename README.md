# LatentScore

[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart.ipynb) [![Listen to Demo](https://img.shields.io/badge/▶_Listen_to_Demo-latentscore.com-8A2BE2?style=flat)](https://latentscore.com/demo) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](LICENSE)

[![Presenting at SIGGRAPH 2026](https://img.shields.io/badge/Presenting_at-SIGGRAPH_2026-B91C1C?style=flat)](https://s2026.siggraph.org/program/talks/) [![Presenting at NIME 2026](https://img.shields.io/badge/Presenting_at-NIME_2026-5B21B6?style=flat)](https://nime2026.org/)

**Generate ambient music from text. Locally. No GPU required.**

```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

That's it. One line. You get audio playing on your speakers.

> ⚠️ **Alpha** &mdash; under active development. API may change between versions. [Read more about how it works](https://substack.com/home/post/p-184245090).

https://private-user-images.githubusercontent.com/140295281/557606724-22889dcc-9287-4712-8ffb-ec19381444c9.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzI1NTQ3OTcsIm5iZiI6MTc3MjU1NDQ5NywicGF0aCI6Ii8xNDAyOTUyODEvNTU3NjA2NzI0LTIyODg5ZGNjLTkyODctNDcxMi04ZmZiLWVjMTkzODE0NDRjOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMwM1QxNjE0NTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03Y2IzMzNlMzhiYTAyZDRmMjE0OWY4ZmNiMmUxNzE4ZTE0YjMzZWU0OGE5MmQyOWYzMzYzZWViYmY5MWNjZWM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VDGlSlEVudf_rmavZpjUqkImO0O1gUYxEqDvDx6PQNo

---

## Live demo

The interactive demo at **[latentscore.com/demo](https://latentscore.com/demo)** lets you type vibes and stream audio in the browser &mdash; no install needed. The full source (FastAPI backend + React frontend + Dockerfile) lives in [`demo/`](demo/). To run it locally: `docker compose -f demo/docker-compose.yml up --build` (see [`demo/README.md`](demo/README.md) for details).

---

## Contents

- [Install](#install) &mdash; 30 seconds
- [Quick start](#quick-start) &mdash; Python in 5 lines
- [Controlling the sound](#controlling-the-sound) &mdash; `MusicConfig` parameters
- [Library DX](docs/latentscore-dx.md) &mdash; streaming, live playlists, async API, bring-your-own-LLM
- [How it works](docs/architecture.md#how-it-works) &mdash; embedding retrieval, no LLM hallucinations
- [Help / FAQ](docs/FAQ.md) &mdash; common questions
- [Citation](#citation) &mdash; SIGGRAPH Talks '26 BibTeX

---

## Install

**Requires Python 3.10&ndash;3.12.** If you don't have it: `brew install python@3.10` (macOS) or `pyenv install 3.10`.

```bash
pip install latentscore
```

Or with conda:

```bash
conda create -n latentscore python=3.10 -y
conda activate latentscore
pip install latentscore
```

The baseline install gives you embedding-match text prompts (`ls.render("vibe")`), `MusicConfig` rendering, and local playback &mdash; everything in the Quick Start below. Optional extras: `[external]` for bring-your-own hosted LLMs (LiteLLM), `[heavy]` for CLAP audio retrieval, `[expressive]` for local LLM inference.

### Verify your install

```bash
latentscore doctor --strict --offline
```

Runs ten checks (Python version, package metadata, license, audio I/O, schema export, core synth render, retrieval render, optional-extra availability) and exits non-zero if anything required is broken. Add `--require-external`/`--require-heavy`/`--require-expressive` to also fail when those extras aren't installed. Add `--json` for machine-readable output.

### Platform support

| Platform | Core (`pip install latentscore`) | `[external]` | `[heavy]` | `[expressive]` | Demo (Docker) |
|---|---|---|---|---|---|
| macOS arm64 | ✅ tested | ✅ tested | ✅ tested | ✅ tested (MLX) | ✅ tested |
| Linux x86_64 | ✅ tested | ✅ tested | ✅ tested | ✅ tested | ✅ tested |
| Windows native | should work | should work | untested | should work (transformers backend; mlx + llama-cpp excluded by markers) | ✅ via Docker Desktop |
| Windows WSL2 | as Linux | as Linux | as Linux | as Linux | as Linux |

If you're on native Windows and hit an issue, the cleanest fallback is **WSL2** (then follow the Linux instructions inside it) or **Docker Desktop** for the demo. If you find that core or `[external]` actually works on native Windows, please open an issue so we can mark it tested.

---

## CLI

```bash
latentscore doctor                       # run 10 install health checks
latentscore doctor --strict --offline    # CI-friendly: nonzero on required-fail
latentscore doctor --json                # machine-readable JSON
latentscore download fast                # prefetch the default model assets
latentscore demo                         # render and play a sample
latentscore demo --duration 30           # 30-second demo
latentscore demo --output ambient.wav    # save to file
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

See [`docs/latentscore-dx.md`](docs/latentscore-dx.md) for the full
parameter reference, relative-step updates, streaming, live playlists,
async API, and bring-your-own-LLM cookbook.

---

## Read more

- [`docs/latentscore-dx.md`](docs/latentscore-dx.md) &mdash; full library DX: parameter reference, streaming, live playlists, async API, bring-your-own-LLM cookbook, audio contract.
- [`docs/architecture.md`](docs/architecture.md) &mdash; system architecture + how the retrieval works under the hood.
- [`docs/FAQ.md`](docs/FAQ.md) &mdash; common questions (first-call hang, system deps, Windows support, citation, …).
- [`data_work/README.md`](data_work/README.md) &mdash; research / training pipeline.
- [`CONTRIBUTE.md`](CONTRIBUTE.md) &mdash; contributor setup; [`docs/contribute/coding-guidelines.md`](docs/contribute/coding-guidelines.md) for style rules.
- [`demo/README.md`](demo/README.md) &mdash; the bundled FastAPI + React demo.

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
