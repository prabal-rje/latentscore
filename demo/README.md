# LatentScore Demo

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-latentscore.com-8A2BE2?style=flat)](https://latentscore.com/demo) [![Jupyter Bundled](https://img.shields.io/badge/📓_JupyterLab-Bundled-F37626?style=flat)](#2-what-you-get) [![GHCR](https://img.shields.io/badge/🐳_Multi--arch_images-CI-2496ED?style=flat)](https://github.com/prabal-rje/latentscore/actions/workflows/publish-images.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](../LICENSE)

**A bundled, runnable artifact of LatentScore.** One
`docker compose up --build` builds the demo from source and gives you
the web UI **and** a JupyterLab playground with the SDK pre-installed —
on your machine, no signup, no `pip install`.

---

## Contents

- [Try it](#1-try-it) - hosted, or local Docker
- [What you get](#2-what-you-get) - the three URLs and what each does
- [Reproducibility](#3-reproducibility) - build from source
- [Read more](#4-read-more) - architecture, dev setup, troubleshooting

---

## 1. Try it

**Hosted - zero install:** [latentscore.com/demo](https://latentscore.com/demo)
opens in your browser.

**Locally — build from source:**

```bash
docker compose up --build
```

(from this `demo/` folder, with [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed). First build takes ~15 min; the first prompt takes ~30
seconds as model weights load. Tested on Linux, macOS, and Windows
(Docker Desktop with WSL2) — both Intel and Apple Silicon.

> ⚠️ **Apple Silicon (M-series) Macs:** the build defaults to `linux/amd64` for reproducibility,
> which runs under emulation and is slow. For a native, much faster build, prefix the command:
>
> ```bash
> LATENTSCORE_DOCKER_PLATFORM=linux/arm64 docker compose up --build
> ```

---

## 2. What you get

Once the stack is up, three services are running:

| Service     | URL                                         | What's there                                |
|-------------|---------------------------------------------|---------------------------------------------|
| **Demo UI** | <http://localhost:4242>                     | The interactive web app (vibe input → audio) |
| **JupyterLab** 📓 | <http://localhost:8889>               | Pre-installed SDK + `quickstart.ipynb`, runs locally - the same notebook content as the [Colab](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb) version |
| Backend API | <http://localhost:4244>                     | Server backend (only relevant if you're poking the API directly) |

JupyterLab is bound to `127.0.0.1` only and runs token-less for
local convenience. Do **not** publish port 8889 to the internet.

---

## 3. Reproducibility

Build from source:

```bash
docker compose up --build
```

That uses the `build:` block in each service definition. No file edits
required. First build takes ~15 min because the backend image
pre-downloads model weights (MiniLM ~90 MB, LAION-CLAP ~1.8 GB).
Subsequent rebuilds hit the Docker layer cache.

---

## 4. Read more

Deeper reading lives under [`docs/`](docs/) so this page stays the
landing page:

- [Architecture](docs/architecture.md) - what the three services do
  and how they connect.
- [Development](docs/development.md) - backend-only / frontend-only
  setups for working on the demo itself.
- [Troubleshooting](docs/troubleshooting.md) - port collisions,
  Docker daemon errors, audio playback quirks.

For everything else, the top-level
[LatentScore README](../README.md) has the library install, API
reference, FAQ, and citation.
