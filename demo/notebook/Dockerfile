# JupyterLab image with latentscore pre-installed. Pairs with the
# demo's docker-compose so anyone running the demo locally also gets
# a working SDK playground at http://localhost:8889 without touching
# their host Python install.
#
# Same quickstart.ipynb as the Colab notebook (one source of truth).

FROM quay.io/jupyter/scipy-notebook:python-3.12@sha256:15b50ef89a316b483279aa5f689c5b26947e586ddb403c504e65a73b47ff8569

USER root

# System libs that latentscore's audio deps link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install latentscore + the [external] and [heavy] extras from the
# bundled source. Build context is the repo root (see
# demo/docker-compose.yml), so the paths below are relative to that.
COPY pyproject.toml README.md LICENSE ./latentscore-src/
COPY latentscore ./latentscore-src/latentscore
COPY common ./latentscore-src/common
RUN pip install --no-cache-dir "./latentscore-src[external,heavy]"

# Pre-download MiniLM so the first ls.render() call in the notebook
# doesn't silently hang on a 90 MB download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Drop a config file that disables token + password auth so users can
# open localhost:8889 and land directly in JupyterLab. Safe on localhost
# only -- the docker-compose mapping intentionally binds to the host's
# loopback. Do NOT expose this port to the internet.
RUN mkdir -p /home/jovyan/.jupyter && \
    printf "c.ServerApp.token = ''\nc.ServerApp.password = ''\nc.ServerApp.ip = '0.0.0.0'\nc.ServerApp.default_url = '/lab/tree/quickstart.ipynb'\nc.LabApp.default_url = '/lab/tree/quickstart.ipynb'\nc.LabServerApp.default_url = '/lab/tree/quickstart.ipynb'\n" \
        > /home/jovyan/.jupyter/jupyter_server_config.py && \
    chown -R jovyan:users /home/jovyan/.jupyter

# Pre-seed a Jupyter notebook-trust secret so the `jupyter trust` call
# below uses the same secret the running JupyterLab will use for
# signature verification (otherwise the live server would generate
# a fresh secret on first launch and refuse to trust the bundled
# notebook's signature).
RUN mkdir -p /home/jovyan/.local/share/jupyter && \
    python -c "import secrets; open('/home/jovyan/.local/share/jupyter/notebook_secret','wb').write(secrets.token_bytes(64))" && \
    chown -R jovyan:users /home/jovyan/.local

# IPython startup script that pre-imports `latentscore as ls` and the
# notebook helpers on every kernel boot, so the user can run any cell
# in any order without first running the RUN ME FIRST cell. Helpers
# now ship inside the latentscore package itself (no separate module).
RUN mkdir -p /home/jovyan/.ipython/profile_default/startup && \
    printf 'import latentscore as ls  # noqa: F401\nfrom latentscore.notebook_helpers import listen, check_key  # noqa: F401\n' \
        > /home/jovyan/.ipython/profile_default/startup/00-latentscore-helpers.py && \
    chown -R jovyan:users /home/jovyan/.ipython

# Copy the Docker-specific quickstart notebook (no Colab `!pip install`
# / `os.kill` cell; latentscore is already pre-installed in this image).
# The Colab path uses notebooks/quickstart-colab.ipynb instead.
COPY --chown=jovyan:users notebooks/quickstart-docker.ipynb /home/jovyan/work/quickstart.ipynb

# The jupyter base image sets HOME=/home/jovyan even for root-run
# commands, so the MiniLM pre-download above wrote
# /home/jovyan/.cache/huggingface/ as root. Hand the whole home dir
# back to jovyan so runtime writes (latentscore's log dir,
# huggingface_hub's cache updates, etc.) don't hit Errno 13.
RUN chown -R jovyan:users /home/jovyan

USER jovyan
WORKDIR /home/jovyan/work

# Sign the bundled notebook so JupyterLab treats its baked outputs as
# trusted on load. Without this, audio players in pre-baked cells show
# up but refuse to play (untrusted HTML is sanitized client-side).
# Sign via nbformat directly using the secret we seeded above, so the
# running JupyterLab (which reads the same secret) verifies the sig
# and renders the outputs as trusted.
RUN python -c "import nbformat; from nbformat.sign import NotebookNotary; \
    nb = nbformat.read('/home/jovyan/work/quickstart.ipynb', as_version=4); \
    NotebookNotary().sign(nb); \
    nbformat.write(nb, '/home/jovyan/work/quickstart.ipynb')"

EXPOSE 8888

# Use the jupyter docker stacks' start.sh entrypoint. The CLI
# --ServerApp.default_url flag wins over the entrypoint's built-in
# default (/lab?), which would otherwise drop users into the file
# browser instead of the quickstart notebook.
CMD ["start-notebook.sh", \
     "--ServerApp.root_dir=/home/jovyan/work", \
     "--ServerApp.default_url=/lab/tree/quickstart.ipynb", \
     "--LabApp.default_url=/lab/tree/quickstart.ipynb"]
