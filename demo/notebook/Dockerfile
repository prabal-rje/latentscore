# JupyterLab image with latentscore pre-installed. Build context: repo ROOT.
#
# Why CPU torch is installed FIRST:
#   The notebook installs latentscore[external,heavy]; the [heavy] extra includes
#   torchvision/laion-clap, which can otherwise make pip pull CUDA torch wheels.

FROM quay.io/jupyter/scipy-notebook:python-3.12@sha256:15b50ef89a316b483279aa5f689c5b26947e586ddb403c504e65a73b47ff8569

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0

# System libs that latentscore's audio deps link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsndfile1 libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Critical: CPU-only torch/torchvision BEFORE installing [external,heavy].
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

# Fail the build loudly if a CUDA torch wheel slipped in.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__); assert torch.version.cuda is None, torch.version.cuda"

# Install latentscore + the [external] and [heavy] extras from bundled source.
COPY pyproject.toml README.md LICENSE ./latentscore-src/
COPY latentscore ./latentscore-src/latentscore
COPY common ./latentscore-src/common
RUN pip install "./latentscore-src[external,heavy]"

# Pre-download MiniLM so the first ls.render() in the notebook doesn't hang.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Disable token + password auth so users can open localhost:8889 directly.
# Safe only because docker-compose binds 8889 to 127.0.0.1.
RUN mkdir -p /home/jovyan/.jupyter && \
    printf "c.ServerApp.token = ''\nc.ServerApp.password = ''\nc.ServerApp.ip = '0.0.0.0'\nc.ServerApp.default_url = '/lab/tree/quickstart.ipynb'\nc.LabApp.default_url = '/lab/tree/quickstart.ipynb'\nc.LabServerApp.default_url = '/lab/tree/quickstart.ipynb'\n" \
        > /home/jovyan/.jupyter/jupyter_server_config.py && \
    chown -R jovyan:users /home/jovyan/.jupyter

# Pre-seed a notebook-trust secret so the signed bundled notebook stays trusted
# (otherwise the live server generates a fresh secret and rejects the signature).
RUN mkdir -p /home/jovyan/.local/share/jupyter && \
    python -c "import secrets; open('/home/jovyan/.local/share/jupyter/notebook_secret','wb').write(secrets.token_bytes(64))" && \
    chown -R jovyan:users /home/jovyan/.local

# IPython startup script that pre-imports helpers on every kernel boot.
RUN mkdir -p /home/jovyan/.ipython/profile_default/startup && \
    printf 'import latentscore as ls  # noqa: F401\nfrom latentscore.notebook_helpers import listen, check_key  # noqa: F401\n' \
        > /home/jovyan/.ipython/profile_default/startup/00-latentscore-helpers.py && \
    chown -R jovyan:users /home/jovyan/.ipython

# Copy the Docker-specific quickstart notebook.
COPY --chown=jovyan:users notebooks/quickstart-docker.ipynb /home/jovyan/work/quickstart.ipynb

# The Jupyter base image writes caches as root during build; hand home back.
RUN chown -R jovyan:users /home/jovyan

USER jovyan
WORKDIR /home/jovyan/work

# Sign the bundled notebook so JupyterLab treats its baked outputs as trusted.
RUN python -c "import nbformat; from nbformat.sign import NotebookNotary; nb = nbformat.read('/home/jovyan/work/quickstart.ipynb', as_version=4); NotebookNotary().sign(nb); nbformat.write(nb, '/home/jovyan/work/quickstart.ipynb')"

EXPOSE 8888

CMD ["start-notebook.sh", \
     "--ServerApp.root_dir=/home/jovyan/work", \
     "--ServerApp.default_url=/lab/tree/quickstart.ipynb", \
     "--LabApp.default_url=/lab/tree/quickstart.ipynb"]
