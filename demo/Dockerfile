# Single-image deploy of the latentscore demo. Build context must be the
# latentscore REPO ROOT so we can pip-install the bundled latentscore source.
#
# Local build:
#   docker build -f demo/Dockerfile -t latentscore-demo .
#
# Railway: configure the service to build with rootDirectory="" and
# dockerfilePath="demo/Dockerfile".

# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:22-alpine@sha256:968df39aedcea65eeb078fb336ed7191baf48f972b4479711397108be0966920 AS frontend
WORKDIR /app
COPY demo/frontend/package.json demo/frontend/package-lock.json* ./
RUN npm ci
COPY demo/frontend/ .
RUN npm run build

# ── Stage 2: Build Python deps (with compilers) ─────────────────────────────
FROM python:3.12-slim@sha256:9d3abd9fc11d06998ccdbdd93b4dd49b5ad7d67fcbbc11c016eb0eb2c2194891 AS pydeps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake libc6-dev libasound2-dev libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install latentscore from the bundled source.
COPY pyproject.toml README.md LICENSE /src/
COPY latentscore /src/latentscore
COPY common /src/common
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /src

# Backend-specific deps.
COPY demo/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download model assets at build time so first-request latency is low.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import laion_clap; m = laion_clap.CLAP_Module(enable_fusion=False); m.load_ckpt()"
RUN python -c "from latentscore.models import FastHeavyModel; FastHeavyModel().warmup()"

# ── Stage 3: Final slim image ────────────────────────────────────────────────
FROM python:3.12-slim@sha256:9d3abd9fc11d06998ccdbdd93b4dd49b5ad7d67fcbbc11c016eb0eb2c2194891
WORKDIR /app

# Runtime libs only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2t64 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and cached models from build stage
COPY --from=pydeps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=pydeps /usr/local/bin /usr/local/bin
COPY --from=pydeps /root/.cache /root/.cache

COPY demo/backend/main.py .
COPY --from=frontend /app/dist ./static

EXPOSE 8000

# Railway sets PORT env var; fall back to 8000 for local use
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
