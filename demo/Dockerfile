# Single-image deploy of the latentscore demo. Build context must be the repo
# ROOT so we can pip-install the bundled latentscore source.
#
# Local build:
#   docker build -f demo/Dockerfile -t latentscore-demo .
#
# Railway:
#   rootDirectory=""
#   dockerfilePath="demo/Dockerfile"
#
# Multi-stage: frontend build + Python deps (with compilers) are built in
# throwaway stages; the final image ships neither node nor compilers.

# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:22-alpine@sha256:968df39aedcea65eeb078fb336ed7191baf48f972b4479711397108be0966920 AS frontend

ARG VITE_ANONYMOUS=false
ENV VITE_ANONYMOUS=${VITE_ANONYMOUS}

WORKDIR /app

COPY demo/frontend/package.json demo/frontend/package-lock.json* ./
RUN npm ci

COPY demo/frontend/ .
RUN npm run build

# ── Stage 2: Build Python deps (with compilers) ──────────────────────────────
FROM python:3.12-slim@sha256:9d3abd9fc11d06998ccdbdd93b4dd49b5ad7d67fcbbc11c016eb0eb2c2194891 AS pydeps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LATENTSCORE_MODEL_DIR=/root/.cache/latentscore/models

ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0
# 1 = also serve fast_heavy and best-of-N scoring (CLAP): +2.3 GB image, +1.4 GB RAM.
ARG LATENTSCORE_DEMO_HEAVY=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc g++ make cmake libc6-dev \
    libasound2-dev libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Critical: CPU-only torch/torchvision BEFORE installing latentscore.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

# Fail the build loudly if a CUDA torch wheel slipped in.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__); assert torch.version.cuda is None, torch.version.cuda"

# Install latentscore from the bundled source.
COPY pyproject.toml README.md LICENSE /src/
COPY latentscore /src/latentscore
COPY common /src/common
RUN pip install /src

# Backend-specific deps (torch/torchvision already satisfied by CPU wheels).
COPY demo/backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Bake model assets at build time so runtime startup needs no network. Fail
# the build if the map did not load: warmup() would quietly use 5 examples.
RUN python -c "from latentscore.models import FastEmbeddingModel; m = FastEmbeddingModel(); m.warmup(); assert m._embed_map_examples()[1] is not None, 'MiniLM embedding map missing'"
RUN if [ "$LATENTSCORE_DEMO_HEAVY" = "1" ]; then \
      python -c "from latentscore.models import FastHeavyModel; FastHeavyModel().warmup()"; \
    fi

# ── Stage 3: Final slim image ────────────────────────────────────────────────
FROM python:3.12-slim@sha256:9d3abd9fc11d06998ccdbdd93b4dd49b5ad7d67fcbbc11c016eb0eb2c2194891

ARG LATENTSCORE_DEMO_HEAVY=0

# Thread and malloc-arena caps: torch, tokenizers and glibc size their pools by
# the host's CPU count, and Railway hosts expose 48+ CPUs to each container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LATENTSCORE_MODEL_DIR=/root/.cache/latentscore/models \
    LATENTSCORE_DEMO_HEAVY=${LATENTSCORE_DEMO_HEAVY} \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    RAYON_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Runtime libs only; no compilers in the final image. NOTE: libasound2t64 is the
# correct runtime ALSA package name on the slim base (libasound2 was renamed in
# the 64-bit time_t transition).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libasound2t64 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and cached models from the build stage.
COPY --from=pydeps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=pydeps /usr/local/bin /usr/local/bin
COPY --from=pydeps /root/.cache /root/.cache

COPY demo/backend/main.py .
COPY --from=frontend /app/dist ./static

EXPOSE 8000

# Railway sets PORT; local Docker falls back to 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
