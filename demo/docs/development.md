# Development

Only relevant if you're modifying the demo source. To **use** the demo,
the [README](../README.md) Docker Compose path is enough:

```bash
docker compose -f demo/docker-compose.yml up --build
```

## Backend only

From the **latentscore repo root**:

```bash
pip install -e ".[external,heavy]"
pip install -r demo/backend/requirements.txt

cd demo/backend
uvicorn main:app --reload --port 8000
```

API at <http://localhost:8000>. `curl http://localhost:8000/api/health`
to confirm it's up.

The first launch downloads the LAION-CLAP checkpoint (~1.8 GB) the
first time `laion_clap.CLAP_Module().load_ckpt()` is called. On a slow
or blocked network this looks like the server is failing - give it a
few minutes, or pre-download with `latentscore download fast_heavy`.

## Frontend only

```bash
cd demo/frontend
npm install        # Node 22+
npm run dev
```

Vite dev server at <http://localhost:5173> with hot-reload. Vite
proxies `/api/*` calls to `http://localhost:8000`, so run the backend
in another terminal first.

## Notebook only

Build the notebook image from source:

```bash
cd demo
docker compose up notebook --build
```

JupyterLab at <http://localhost:8889>.

## Full stack from source

Build all three services from local source:

```bash
cd demo
docker compose up --build
```

That uses each service's `build:` block. First build takes ~15 min
because the backend image pre-downloads model weights (MiniLM ~90 MB,
LAION-CLAP ~1.8 GB). Subsequent rebuilds hit the Docker layer cache.

## Common dev gotchas

- **`pip install` requires Python 3.11-3.12.** For other Python versions, use Docker or Colab.
- ⚠️ **Apple Silicon: prefix builds with `LATENTSCORE_DOCKER_PLATFORM=linux/arm64`.** The default `linux/amd64` runs under emulation and is slow; the override builds natively.
- **`npm install` engine errors.** Use Node 22+. Vite 7 requires
  Node 18+ but the Dockerfile pins to Node 22, so match it to avoid
  bundle differences.
- **Hot reload not picking up Python changes.** `uvicorn --reload`
  only watches the cwd. Run it from `demo/backend/` so it watches
  `main.py` and the latentscore source via the editable install.
