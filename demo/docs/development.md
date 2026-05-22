# Development

Only relevant if you're modifying the demo source. To **use** the demo,
the [README](../README.md) `docker compose up` path is enough.

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

```bash
cd demo
docker compose up notebook
```

JupyterLab at <http://localhost:8889>. The image is rebuilt from
`demo/notebook/Dockerfile` if you uncomment its `build:` block in
`docker-compose.yml`.

## Full stack with from-source builds

To build all three services from your local source instead of
pulling pre-built images from GHCR, just pass `--build`:

```bash
cd demo
docker compose up --build
```

That ignores each service's `image:` line and uses the matching
`build:` block. First build takes ~5 min because the backend image
pre-downloads model weights (MiniLM ~90 MB, LAION-CLAP ~1.8 GB).
Subsequent rebuilds hit the Docker layer cache.

## Common dev gotchas

- **`pip install` fails on Python 3.13.** The library targets
  Python 3.10-3.12. Some of `laion-clap`'s transitive deps don't
  have 3.13 wheels yet.
- **`npm install` engine errors.** Use Node 22+. Vite 7 requires
  Node 18+ but the Dockerfile pins to Node 22, so match it to avoid
  bundle differences.
- **Hot reload not picking up Python changes.** `uvicorn --reload`
  only watches the cwd. Run it from `demo/backend/` so it watches
  `main.py` and the latentscore source via the editable install.
