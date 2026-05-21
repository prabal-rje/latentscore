# LatentScore Demo

The web-based interactive demo that powers [latentscore.com/demo](https://latentscore.com/demo).
A FastAPI + WebSocket backend wraps the `latentscore` library; a React + Vite frontend lets
visitors type vibes, tweak `MusicConfig` parameters, and stream live audio.

```
demo/
├── backend/                 FastAPI app
│   ├── main.py              the whole server (~900 LOC)
│   ├── requirements.txt     fastapi + uvicorn + torchvision + laion-clap
│   └── Dockerfile           per-service image (used by docker-compose)
├── frontend/                React 19 + Vite + TypeScript
│   ├── src/                 components, hooks, store, api client
│   ├── package.json
│   └── Dockerfile           nginx-served static bundle
├── docker-compose.yml       2-service local stack (backend :8001, frontend :3002)
└── Dockerfile               single-image multi-stage build (used by Railway)
```

## Quick start: local dev (no Docker)

From the **repo root**:

```bash
# 1. Install latentscore (editable, so the backend sees your local changes).
pip install -e .

# 2. Install backend extras and run.
pip install -r demo/backend/requirements.txt
cd demo/backend && uvicorn main:app --reload
```

In another shell:

```bash
cd demo/frontend
npm install
npm run dev      # serves on http://localhost:5173, proxies API to backend on 8000
```

## Local Docker stack

From the `demo/` directory:

```bash
docker compose up --build
```

- Backend: <http://localhost:8001>
- Frontend: <http://localhost:3002>

The first build takes 10-15 minutes because the backend image pre-downloads the
MiniLM (~90 MB) and LAION-CLAP (~1.8 GB) model assets at build time. After that,
the image is cached.

## Single-image production build

The top-level `Dockerfile` produces one image that serves both the API and the
static frontend bundle on a single port. This is what Railway deploys.

```bash
# Build context is the REPO ROOT, not demo/
docker build -f demo/Dockerfile -t latentscore-demo .
docker run -p 8000:8000 latentscore-demo
```

## Live deployment

The production demo runs at [latentscore.com/demo](https://latentscore.com/demo) (Railway,
served by the multi-stage `demo/Dockerfile`). Reviewers can use the hosted version
directly without running the stack locally.

## Why the backend lives in this repo

Previously the demo lived in a separate `latentscore-demo` repository. As of the
ACM MM OSS submission, it's been vendored in alongside the library so the artifact
is self-contained: one `git clone` gets the library, the demo, the paper code, and
the install harness. The library itself (`latentscore/`) does not depend on
anything in `demo/`; the published wheel (`pip install latentscore`) does not
include the demo.
