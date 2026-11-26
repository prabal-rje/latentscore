# LatentScore

[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb) [![Listen to Demo](https://img.shields.io/badge/▶_Listen_to_Demo-latentscore.com-8A2BE2?style=flat)](https://latentscore.com/demo) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](LICENSE)

[![Presenting at SIGGRAPH 2026](https://img.shields.io/badge/Presenting_at-SIGGRAPH_2026-B91C1C?style=flat)](https://s2026.siggraph.org/program/talks/) [![Presenting at NIME 2026](https://img.shields.io/badge/Presenting_at-NIME_2026-5B21B6?style=flat)](https://nime2026.org/)

LatentScore generates calm ambient music from text prompts on your own machine without a GPU. You can use it in Python, JavaScript, TypeScript, or Rust, or [read how the system works](https://prabal.ca/posts/latentscore-research/).

https://private-user-images.githubusercontent.com/140295281/557606724-22889dcc-9287-4712-8ffb-ec19381444c9.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzI1NTQ3OTcsIm5iZiI6MTc3MjU1NDQ5NywicGF0aCI6Ii8xNDAyOTUyODEvNTU3NjA2NzI0LTIyODg5ZGNjLTkyODctNDcxMi04ZmZiLWVjMTkzODE0NDRjOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMwM1QxNjE0NTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03Y2IzMzNlMzhiYTAyZDRmMjE0OWY4ZmNiMmUxNzE4ZTE0YjMzZWU0OGE5MmQyOWYzMzYzZWViYmY5MWNjZWM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VDGlSlEVudf_rmavZpjUqkImO0O1gUYxEqDvDx6PQNo

[Try it now](#try-it-now) | [Build reactive local apps](#build-reactive-local-apps) | [Try the web demo](#try-the-web-demo) | [Python SDK and CLI](#python-sdk-and-cli) | [Controlling the sound](#controlling-the-sound) | [Repository layout](#repository-layout) | [Read more](#read-more) | [Citation](#citation)

## Try it now

You can try LatentScore in five ways:

- Open [latentscore.com/demo](https://latentscore.com/demo) in your browser to listen right away without installing anything.

- Install the Python package with `pip install latentscore` on macOS or Linux (Python 3.11 or 3.12).

- Install the TypeScript and JavaScript package with `npm install latentscore` for Node.js, Bun, Deno, or browsers. Read [packages/latentscore/README.md](packages/latentscore/README.md) for details.

- Add `latentscore = "0.1"` to `Cargo.toml` for offline Rust rendering, or enable `features = ["playback"]` for live audio output. Read [crates/latentscore/README.md](crates/latentscore/README.md) for details.

- Run the [Google Colab quickstart notebook](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb) on a free CPU runtime.

```python
import latentscore as ls

ls.render("morning coffee shop").play()
```

[Back to top](#latentscore)

## Build reactive local apps

Each language SDK includes a stateful `Score` engine for interactive tools such as writing editors, timers, or games where music responds as the user types.

Calling `steer(text)` is idempotent. Calling it once or ten times with `"rainy midnight coffee shop"` targets the exact same sound. The engine glides smoothly from the current sound over `glide_seconds`. For example, at 100 BPM one bar lasts 2.4 seconds, so a `3.5`-second glide crosses roughly one and a half bars.

### Python

```python
import time
import latentscore as ls

with ls.Score("quiet morning desk", glide_seconds=3.5).play() as score:
    time.sleep(6)
    score.steer("rainy midnight coffee shop")  # glides smoothly while playing
    time.sleep(6)
    score.nudge(space=ls.Step(+1))             # widen the reverb space by one step
    time.sleep(4)
    score.drift()                              # glide toward a neighboring mood
    time.sleep(6)
```

### JavaScript and TypeScript (browser)

```ts
import { createBrowserScore } from "latentscore/browser";

const music = createBrowserScore("quiet morning desk", { glideSeconds: 3.5 });

playButton.addEventListener("click", () => music.play());
editor.addEventListener("input", () => music.steerOnType(editor.value));

const drift = music.startAutoDrift({ intervalMs: 12000 });
stopButton.addEventListener("click", () => {
  drift.stop();
  music.stop(0.5);
});
```

### Rust

```rust
use latentscore::Score;
use std::thread;
use std::time::Duration;

let handle = Score::from_text("quiet morning desk")
    .with_glide_seconds(3.5)
    .play()?;

thread::sleep(Duration::from_secs(6));
handle.steer("rainy midnight coffee shop");
thread::sleep(Duration::from_secs(6));
handle.drift_next();
thread::sleep(Duration::from_secs(6));
handle.stop(0.5);
```

All three SDKs read the same shared specification in `spec/` (`config.schema.json`, `mood_lexicon.json`, and `golden_cases.json`).

[Back to top](#latentscore)

## Try the web demo

Open [latentscore.com/demo](https://latentscore.com/demo) to run the browser version directly. It synthesizes audio locally with Pyodide, searches 14,035 moods with Transformers.js, and loops smoothly with Auto Drift.

### Run the full server and JupyterLab stack with Docker

You can run the full demo stack on Linux, macOS, or Windows with [Docker Desktop](https://www.docker.com/products/docker-desktop/) (using WSL2 on Windows). The first build takes about 15 minutes, and later starts are fast.

```bash
docker compose -f demo/docker-compose.yml up --build
```

On Apple Silicon Macs, set the arm64 platform so Docker builds natively instead of emulating x86_64:

```bash
LATENTSCORE_DOCKER_PLATFORM=linux/arm64 docker compose -f demo/docker-compose.yml up --build
```

After the containers start, open these local addresses:

- `http://localhost:4242` for the web demo UI.

- `http://localhost:8889` for the JupyterLab playground with `latentscore` pre-installed.

### Run the lightweight static browser demo

You can also serve the zero-backend browser build inside a small Nginx container:

```bash
docker compose -f demo/docker-compose.yml --profile static up --build static
```

Then open `http://localhost:8080`. Read [demo/README.md](demo/README.md) for local development instructions.

[Back to top](#latentscore)

## Python SDK and CLI

Local Python installs work on macOS and Linux with Python 3.11 or 3.12. On Windows, use Docker Desktop or Colab.

On Ubuntu or Debian Linux, install system audio libraries first:

```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 libportaudio2
```

Create a virtual environment and install the package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install latentscore
```

On Linux, install CPU-only PyTorch first so pip skips roughly 5 GB of unused CUDA packages:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3"
pip install latentscore
```

Optional extras add more retrieval and LLM options:

- Run `pip install "latentscore[external]"` to connect your own hosted LLM through LiteLLM (such as Anthropic, Gemini, or OpenAI).

- Run `pip install "latentscore[heavy]"` for experimental CLAP audio-based retrieval (`fast_heavy`).

- Run `pip install "latentscore[expressive]"` for experimental local LLM inference.

Verify your installation and render a sample track with the CLI:

```bash
latentscore doctor --strict --offline
latentscore download fast
latentscore demo --duration 30 --output ambient.wav
```

You can also render and save a clip in four lines of Python:

```python
import latentscore as ls

ls.prefetch("fast")
audio = ls.render("morning coffee shop", duration=10.0)
audio.play()
audio.save("output.wav")
```

[Back to top](#latentscore)

## Controlling the sound

Pass a structured `MusicConfig` or a partial `MusicConfigUpdate` when you want direct control over key, mode, tempo, space, or instrument layers:

```python
import latentscore as ls

config = ls.MusicConfig(
    tempo="slow",
    mode="dorian",
    root="d",
    bass="drone",
    pad="ambient_drift",
    melody="contemplative",
    rhythm="minimal",
    texture="shimmer",
    echo="heavy",
    density=3,
    brightness="dark",
    space="vast",
)
ls.render(config, duration=10.0).play()

ls.render(
    "morning coffee shop",
    update=ls.MusicConfigUpdate(
        tempo="very_fast",
        brightness="very_dark",
        echo="infinite",
    ),
).play()
```

Read [docs/library.md](docs/library.md) for the full parameter reference, streaming guide, async API, and custom LLM recipes.

[Back to top](#latentscore)

## Repository layout

- `latentscore/`: Python synthesizer, retrieval models, streaming engine, CLI, and reactive `Score`.

- `packages/latentscore/`: Pure TypeScript and JavaScript SDK (dual ESM and CJS, zero runtime dependencies, Web Audio browser adapter).

- `crates/latentscore/`: Pure Rust crate with minimal `serde` dependencies and optional lock-free `cpal` live playback.

- `spec/`: Shared 34-field config schema, deterministic mood lexicon, and cross-language DSP test cases.

- `demo/`: Static browser demo (Alba UI, Auto Drift, Share card) plus Docker server and JupyterLab stacks.

[Back to top](#latentscore)

## Read more

- [Library documentation](docs/library.md) covers every music parameter, streaming, async usage, and custom LLMs.

- [FAQ](docs/FAQ.md) covers first-call warmup, system audio libraries, Windows setup, and troubleshooting.

- [Retrieval map card](RETRIEVAL_MAP_CARD.md) documents the schema, models, biases, and license for the 14,035-mood embedding map.

- [Research pipeline](data_work/README.md) holds the dataset construction and scoring scripts.

- [Contributing](CONTRIBUTE.md) explains the local dev environment and coding checks.

[Back to top](#latentscore)

## Citation

If you use LatentScore in your research, cite the SIGGRAPH Talks '26 paper:

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

## License

LatentScore uses the [Apache License 2.0](LICENSE).

[Back to top](#latentscore)
