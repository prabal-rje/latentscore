# Troubleshooting

If your problem isn't here, the [project FAQ](../../docs/FAQ.md) covers
broader library issues.

## `docker compose up` fails: "Cannot connect to Docker daemon."

Docker Desktop (or Docker Engine on Linux) isn't running. Start it,
then verify with `docker ps`.

## `docker compose up` fails: "manifest unknown" on pull.

The pre-built images haven't been published yet for the version your
compose file references. Two options:

1. **Wait for a tagged release** - tagged releases trigger a CI build
   that publishes the images to GHCR.
2. **Build from source instead** - edit `docker-compose.yml`, comment
   each `image:`, uncomment the matching `build:`, then
   `docker compose up --build`.

## Backend stuck on "Waiting: Healthy" for several minutes.

Normal on the first run. The backend container loads MiniLM and
LAION-CLAP at startup; on a cold cache that's a multi-minute pause.
The healthcheck retries every 10s. If it's still red after ~2 min,
`docker compose logs backend` will show whether it's a network issue
or insufficient disk.

## Backend build (from-source) fails on `laion_clap` pre-download step.

The build is downloading the CLAP checkpoint into the image. Common
causes:

- Blocked outbound HTTPS (firewall / VPN).
- Disk full on the Docker daemon. `docker system prune -af` frees
  unused layers.
- OOM during the model load on a small Docker VM. Bump Docker
  Desktop's memory allocation to 8 GB+.

## Audio doesn't play but the network tab shows the WAV arriving.

Browser autoplay policy. Click something on the page (not just type)
before the first render so the gesture-required check passes.

On iOS, also turn the silent-mode switch off - the demo shows a
banner about this on iPhones.

## Port already in use.

The compose ports are:

- `4244` - backend
- `4242` - frontend
- `127.0.0.1:8889` - JupyterLab (loopback only)

If something else owns one of them, edit `docker-compose.yml` or
stop the conflicting process.

## JupyterLab loads but the notebook doesn't open automatically.

The hero button on the demo's landing page deep-links to
`http://localhost:8889/lab/tree/quickstart.ipynb`. If you got to
JupyterLab via a different URL (e.g. just `http://localhost:8889`),
the file browser opens at `/home/jovyan/work/` - click
`quickstart.ipynb` from there.

## `fast_heavy` model fails with `laion_clap` ImportError (local dev).

You're running the backend locally without the `[heavy]` extra.
`pip install -e ".[heavy]"` from the repo root, or stick to the
default `fast` model.
