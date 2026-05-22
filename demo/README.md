# LatentScore Demo

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-latentscore.com-8A2BE2?style=flat)](https://latentscore.com/demo) [![Jupyter Bundled](https://img.shields.io/badge/📓_JupyterLab-Bundled-F37626?style=flat)](#2-what-you-get) [![GHCR](https://img.shields.io/badge/🐳_Multi--arch_images-CI-2496ED?style=flat)](https://github.com/prabal-rje/latentscore/actions/workflows/publish-images.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat)](../LICENSE)

**A bundled, runnable artifact of LatentScore.** One `docker compose up`
gives you the web UI **and** a JupyterLab playground with the SDK
pre-installed - both on your machine, no signup, no `pip install`.

---

## Contents

- [Try it](#1-try-it) - hosted, or local Docker
- [What you get](#2-what-you-get) - the three URLs and what each does
- [Reproducibility](#3-reproducibility) - pull pre-built vs build from source
- [Read more](#4-read-more) - architecture, dev setup, troubleshooting

---

## 1. Try it

**Hosted - zero install:** [latentscore.com/demo](https://latentscore.com/demo)
opens in your browser.

**Locally - one command:**

```bash
docker compose up
```

(from this `demo/` folder, with [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed). Pulls 3 pre-built multi-arch images from GHCR. About 60
seconds on a warm network. Works on macOS, Linux, and Windows WSL2 -
both Intel and Apple Silicon.

---

## 2. What you get

Once the stack is up, three services are running:

| Service     | URL                                         | What's there                                |
|-------------|---------------------------------------------|---------------------------------------------|
| **Demo UI** | <http://localhost:4242>                     | The interactive web app (vibe input → audio) |
| **JupyterLab** 📓 | <http://localhost:8889>               | Pre-installed SDK + `quickstart.ipynb`, runs locally - the same notebook content as the [Colab](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb) version |
| Backend API | <http://localhost:4244>                     | FastAPI server (only relevant if you're poking the API directly) |

JupyterLab is bound to `127.0.0.1` only and runs token-less for
local convenience. Do **not** publish port 8889 to the internet.

---

## 3. Reproducibility

The default `docker compose up` pulls finished images (~60 s). That's
the fast path.

To verify the published images match this source - the typical
OSS-track-reviewer concern - just add `--build`:

```bash
docker compose up --build
```

That ignores the `image:` line and uses the `build:` block in each
service definition. No file edits required. First build takes
~5 min because the backend image pre-downloads model weights
(MiniLM ~90 MB, LAION-CLAP ~1.8 GB). Subsequent rebuilds hit the
Docker layer cache.

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
