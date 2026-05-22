# JupyterLab image with latentscore pre-installed. Pairs with the
# demo's docker-compose so anyone running the demo locally also gets
# a working SDK playground at http://localhost:8889 without touching
# their host Python install.
#
# Same quickstart.ipynb as the Colab notebook (one source of truth).

FROM quay.io/jupyter/scipy-notebook:python-3.12

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
    printf "c.ServerApp.token = ''\nc.ServerApp.password = ''\nc.ServerApp.ip = '0.0.0.0'\n" \
        > /home/jovyan/.jupyter/jupyter_server_config.py && \
    chown -R jovyan:users /home/jovyan/.jupyter

# Copy the quickstart notebook into the default working directory.
COPY --chown=jovyan:users notebooks/quickstart.ipynb /home/jovyan/work/quickstart.ipynb

USER jovyan
WORKDIR /home/jovyan/work
EXPOSE 8888

# Use the jupyter docker stacks' start.sh entrypoint (default).
CMD ["start-notebook.sh", "--ServerApp.root_dir=/home/jovyan/work"]
