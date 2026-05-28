# Backend image for the Docker Compose demo. Build context: repo ROOT.
#
# Why CPU torch is installed FIRST:
#   latentscore depends on torch, and backend extras pull torchvision/laion-clap.
#   If pip resolves those before torch is present, Linux builds drag in the
#   ~5 GB CUDA wheel stack. Installing explicit CPU-only torch/torchvision first
#   means every later install reuses them.
#
# Single-stage on purpose: the baked model weights live in one place, so there
# is no cross-stage cache-copy step that could silently fail and cause runtime
# re-downloads. (The single-image demo/Dockerfile IS multi-stage because it has
# a separate frontend build to keep out of the runtime image.)

FROM python:3.12-slim@sha256:9d3abd9fc11d06998ccdbdd93b4dd49b5ad7d67fcbbc11c016eb0eb2c2194891

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Pinned for reproducibility. No "+cpu" suffix: --index-url already forces CPU
# wheels, "==2.12.0" matches "2.12.0+cpu" on x86_64, and the plain version also
# resolves the aarch64 wheel (so linux/arm64 builds work). If an arm64 build
# can't find this exact version, relax to "torch>=2.1,<3".
ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0

WORKDIR /app

# Build + runtime deps for the scipy/audio/laion-clap stack.
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

# Install latentscore from the bundled source. Copy only what the package needs
# so Docker layer caching is not invalidated by frontend/demo-only changes.
COPY pyproject.toml README.md LICENSE ./latentscore-src/
COPY latentscore ./latentscore-src/latentscore
COPY common ./latentscore-src/common
RUN pip install ./latentscore-src

# Backend-specific deps (torch/torchvision already satisfied by CPU wheels).
COPY demo/backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Bake model assets at build time so runtime startup isn't network-bound.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import laion_clap; m = laion_clap.CLAP_Module(enable_fusion=False); m.load_ckpt()"
RUN python -c "from latentscore.models import FastHeavyModel; FastHeavyModel().warmup()"

COPY demo/backend/main.py .

EXPOSE 8000

# Image-level default healthcheck (Compose overrides this with its own).
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=600s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
