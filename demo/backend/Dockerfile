# Backend image. Build context is the latentscore REPO ROOT (see
# demo/docker-compose.yml), so we have access to the latentscore source
# at ./latentscore/ relative to the build context and install it directly.

FROM python:3.12-slim

WORKDIR /app

# System deps for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake libc6-dev \
    libasound2-dev libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install latentscore from the bundled source. Copy only what the package
# needs (keeps the layer cache stable when demo/ changes).
COPY pyproject.toml README.md LICENSE ./latentscore-src/
COPY latentscore ./latentscore-src/latentscore
COPY common ./latentscore-src/common
RUN pip install --no-cache-dir ./latentscore-src

# Backend-specific deps.
COPY demo/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model assets at build time so first-request latency is low.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import laion_clap; m = laion_clap.CLAP_Module(enable_fusion=False); m.load_ckpt()"
RUN python -c "from latentscore.models import FastHeavyModel; FastHeavyModel().warmup()"

COPY demo/backend/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
