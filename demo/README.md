# LatentScore Demo

An interactive web demo that turns short text descriptions ("warm rain on a tin
roof", "jazz cafe at midnight") into continuously-rendered ambient music. It's
the artifact hosted at **[latentscore.com/demo](https://latentscore.com/demo)**
and the same code is bundled here so you can read it end-to-end and run it
offline.

If you've never used LatentScore before, this folder is the easiest place to
see what the library can do: type a vibe, hear something reasonable, tweak the
parameters that produced it, save the WAV.

---

## Try it without installing anything

The hosted version at **[latentscore.com/demo](https://latentscore.com/demo)**
runs the exact code in this folder. Open it in a browser — no signup, no API
key, nothing to install.

If you only have 30 seconds, that's the right path. Everything below is for
people who want to read the source, run it offline, or modify it.

---

## What the demo actually does

After a hero / landing page, you reach a single interactive surface with four
moving parts:

1. **Vibe input.** A text box. Type anything — "late-night neon", "rain on a
   tin roof", "lo-fi study beats". When you submit, the backend embeds your
   text with the [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
   text encoder and finds the closest configuration in a precomputed library
   of music configs (cosine similarity over the embedding map). It's a
   similarity lookup, not an LLM — fast and deterministic, but bounded by what
   the library knows about. There's also an opt-in "Bring your own LLM" mode
   that routes your prompt through a hosted model (Gemini, Claude, GPT, …) for
   more open-ended generation.
2. **Preset grid.** 12 ready-made vibes you can one-click to skip the typing
   (5 visible at a time, the rest behind a scroller). Useful for getting a
   feel before you experiment.
3. **Parameter panel.** Button option groups over the underlying
   `MusicConfig`: tempo, brightness, density, bass style, melody style, echo,
   mode (major / minor / dorian / …), root note, and a few more. Changes are
   *staged* in the UI; click **Apply Changes** to re-render with the new
   parameters. This is where the library's actual control surface is exposed
   — the text-prompt path above is a shortcut that fills these in for you.
4. **Audio player + visualizer.** Inline playback with a particle
   visualization that reacts to the current `MusicConfig` (its palette and
   density), not to the audio waveform itself. You can save the rendered
   clip as a WAV file.

Two playback modes are available via a top-right **Render / Live** toggle:

- **Render** (default) — fetch a full ~60-second WAV, then play it. Simple,
  reliable, what every browser supports.
- **Live** — opt-in WebSocket mode that streams short phase-continuous audio
  chunks as they're rendered, so you can hear updates faster after each
  Apply Changes. Best on a fast local network or when you're iterating
  quickly.

There's also a playlist mode for queueing multiple vibes and advancing
through them (no crossfade — each track starts when the previous one
finishes).

---

## Recommended setup paths (pick one)

| You want to... | Do this | Time | Notes |
|---|---|---|---|
| Just see it work | Open <https://latentscore.com/demo> | 10 sec | Zero install. Same code as this folder. |
| Run it offline with Docker | `cd demo && docker compose up --build` | **15-20 min** | First build pre-downloads ~2&nbsp;GB of model weights. Cached after that. |
| Develop / iterate on the demo | See [Local development](#local-development) below | ~5 min | Backend + frontend separately, hot reload. |

For ACM MM OSS reviewers: option 1 is enough to evaluate the working artifact.
Option 2 proves the bundled source is self-contained. Option 3 is only for
people who want to modify the demo itself.

---

## Architecture

```
┌──────────────────────────────────┐
│  Browser                         │  React 19 + Vite + TypeScript
│  http://localhost:4242 (Compose) │  demo/frontend/
└────────────────┬─────────────────┘
                 │  HTTP for Render mode, WebSocket for Live mode
                 ▼
┌──────────────────────────────────┐
│  FastAPI server                  │  ~900 LOC, single main.py
│  http://localhost:4244 (Compose) │  demo/backend/main.py
│                                  │
│  • routes (REST + WS)            │
│  • session state (in-memory)     │
│  • DSP process pool              │
│  • WAV encoding                  │
│  • CLAP-based candidate scoring  │
└────────────────┬─────────────────┘
                 │  Python imports
                 ▼
┌──────────────────────────────────┐
│  latentscore library             │  ../latentscore/
│  • text → MusicConfig retrieval  │  (FastEmbeddingModel, FastHeavyModel)
│  • audio synthesis (`assemble`)  │  installed editably from repo root
│  • streaming generator (`live`)  │
└────────────────┬─────────────────┘
                 │
                 ▼
            audio samples
            (float32 mono @ 44.1 kHz)
            → encoded to WAV by backend → sent to browser
```

**What the library does:** vibe text → embedding → nearest-neighbor lookup
returning a `MusicConfig` (a structured description of the piece — tempo,
mode, bass style, etc.); procedural audio synthesis from that config.

**What the backend adds:** session state keyed by short IDs; a CPU-bound
worker pool so render requests don't block the event loop; WAV byte-encoding
for HTTP and base64-over-WebSocket delivery; an optional CLAP-based
candidate scorer (`POST /api/select-best-config`) that renders a few
variations and picks the one whose audio embedding best matches the prompt.

There is no database. Session state lives in an in-memory dict; restart the
server and it's gone. That's intentional — the demo is a stateless
interactive artifact, not a service.

### Terms used above

- **MiniLM** — a 384-dim sentence-transformer used to embed the user's text.
  ~90&nbsp;MB download on first use.
- **CLAP** — *Contrastive Language-Audio Pre-training* (LAION); a joint
  audio+text encoder. The `fast_heavy` model uses it to match the prompt
  against the *sound* of each library config, not just its description.
  ~1.8&nbsp;GB checkpoint, only loaded when used.
- **MusicConfig** — the library's central data type. A frozen Pydantic model
  describing one piece of music (tempo, mode, instrument styles, density,
  etc.). See `latentscore/config.py` for fields.
- **WAV @ 44.1 kHz / float32 / mono** — the audio format the library produces
  and the backend sends. Standard browser-playable; ~7&nbsp;MB per 60-second
  clip.

---

## Project structure

```
demo/
├── README.md            this file
├── docker-compose.yml   3-service local stack (backend + frontend + Jupyter)
├── Dockerfile           single-image multi-stage build (production deploy)
│
├── backend/
│   ├── main.py          FastAPI app: REST + WebSocket + static frontend
│   ├── requirements.txt fastapi + uvicorn + torchvision + laion-clap
│   └── Dockerfile       backend-only image (used by docker-compose)
│
└── frontend/
    ├── src/
    │   ├── App.tsx               router + layout
    │   ├── main.tsx              entrypoint
    │   ├── api.ts                typed client for the FastAPI endpoints
    │   ├── store.ts              zustand store for session + parameter state
    │   ├── types.ts              shared TypeScript types
    │   ├── components/
    │   │   ├── LandingPage.tsx       hero + presets
    │   │   ├── DemoPage.tsx          the main interactive surface
    │   │   ├── VibeInput.tsx         text input for prompts
    │   │   ├── PresetGrid.tsx        clickable preset tiles (12 entries)
    │   │   ├── ParamPanel.tsx        button-group MusicConfig editor
    │   │   └── ParticleVisualizer.tsx config-reactive particle canvas
    │   └── hooks/
    │       └── useAudioPlayer.ts     Web Audio playback + WS streaming
    ├── package.json     React 19 + Vite 7 + tanstack-query + zustand
    ├── vite.config.ts   dev server proxy to backend
    ├── nginx.conf       production-image static serving
    └── Dockerfile       frontend-only image (used by docker-compose)
```

---

## How features map to code

If you want to read the source, start here:

| User-visible feature | Backend handler | Frontend component |
|---|---|---|
| Type a vibe → audio | `POST /api/generate` (main.py:598) | `VibeInput.tsx` |
| Click a preset | Same `/api/generate` with preset's vibe | `PresetGrid.tsx` |
| Edit parameters → Apply | `POST /api/update-params` (main.py:656) | `ParamPanel.tsx` |
| Live-stream WebSocket | `WS /ws/stream/{session_id}` (main.py:808) | `useAudioPlayer.ts` |
| Queue a playlist | `POST /api/playlist/create` (main.py:882) | `DemoPage.tsx` (inline) |
| Save WAV | `GET /api/render/{session_id}` (main.py:790) | `DemoPage.tsx` (inline) |
| Best-of-N candidate scoring | `POST /api/select-best-config` (main.py:690) | (called internally) |
| See available models | `GET /api/capabilities` (main.py:576) | `App.tsx` |

The `latentscore` calls themselves are mostly in `main.py` lines ~150–550 —
look for `ls.live(...)`, `FastEmbeddingModel`, `FastHeavyModel`, and the
parameter-mapping helpers imported from `latentscore.config`.

---

## Local development

### Backend only (recommended for library work)

From the **repo root**:

```bash
# 1. Install latentscore in editable mode so the backend sees your changes.
pip install -e ".[external,heavy]"

# 2. Backend extras (fastapi, uvicorn).
pip install -r demo/backend/requirements.txt

# 3. Run.
cd demo/backend && uvicorn main:app --reload --port 8000
```

The API is now at <http://localhost:8000>. Try
`curl http://localhost:8000/api/health` to confirm it's up.

**Heads up:** the backend imports `laion_clap` at module load and calls
`load_ckpt()` during startup. If the LAION-CLAP weights aren't cached yet
(~1.8&nbsp;GB), startup will hang on the first launch while it downloads.
On a slow or blocked network this looks like the server is failing — give it
a few minutes, or pre-download with `latentscore download fast_heavy` first.

### Frontend only (recommended for UI work)

```bash
cd demo/frontend
npm install        # Node 22+ recommended (matches the Docker image)
npm run dev        # Vite dev server with HMR
```

Vite proxies API calls to `http://localhost:8000` (configured in
`vite.config.ts`), so run the backend in another shell first.

The Vite dev server prints its URL — usually <http://localhost:5173>.

### Full Docker Compose stack

```bash
cd demo
docker compose up --build
```

Backend on <http://localhost:4244>, frontend on <http://localhost:4242>, and JupyterLab on <http://localhost:8889> (token-less, localhost only).

The first build is slow because the backend image pre-downloads model
weights at build time (MiniLM ~90&nbsp;MB, LAION-CLAP ~1.8&nbsp;GB, plus a
CLAP embedding map). Subsequent rebuilds hit the Docker layer cache and
finish in seconds unless you change the backend `requirements.txt` or the
latentscore source.

### Single-image production build (what Railway deploys)

The `Dockerfile` at the top of this folder combines the frontend bundle
and the backend into one image, serving both on a single port. Build
context must be the **latentscore repo root** so the image can pip-install
the bundled latentscore source.

```bash
# From the repo root:
docker build -f demo/Dockerfile -t latentscore-demo .
docker run -p 8000:8000 latentscore-demo
```

Open <http://localhost:8000>.

---

## Troubleshooting

**`docker compose up` fails immediately with "Cannot connect to Docker daemon."**
Make sure Docker Desktop (or Docker Engine on Linux) is running. `docker ps`
should succeed.

**Backend container is stuck on "Waiting: Healthy" for several minutes.**
This is normal on the first run — the backend image's startup also triggers
`laion_clap.CLAP_Module.load_ckpt()` and the SentenceTransformer load. The
healthcheck retries every 10s with up to 3 attempts; if you see consistent
failures after ~30s, check `docker compose logs backend` for network errors
or insufficient disk space.

**Backend build fails on `RUN python -c "import laion_clap; ..."` step.**
The pre-download step is downloading the CLAP checkpoint into the image.
Common causes: blocked outbound HTTPS, disk full on the Docker daemon,
out-of-memory during the model load. Free up disk space (`docker system
prune`), make sure your network can reach Hugging Face, and retry.

**`pip install -r demo/backend/requirements.txt` complains about Python 3.13.**
The library targets Python 3.10–3.12. Some of `laion-clap`'s transitive
dependencies don't have 3.13 wheels yet. Use a 3.10–3.12 venv or conda env.

**Frontend `npm install` fails with engine errors.**
Use Node 22 or newer (the Docker image uses `node:22-alpine`, and Vite 7
requires Node 18.0+). On older Node, the install or build may fail
silently or produce a broken bundle.

**Audio doesn't play but the network tab shows the WAV arriving.**
Some browsers require a user gesture (a click) before they'll play audio.
Make sure you've clicked something on the page (not just typed) before the
first render. On iOS, also turn the silent-mode switch off — the demo
shows a banner about this on iPhones.

**Port already in use.** The compose ports (4244 backend, 4242 frontend,
8889 JupyterLab) are mapped from the host. If something's already on
those ports, edit `demo/docker-compose.yml` or stop the conflicting
process.

**`fast_heavy` model fails with `laion_clap` ImportError.** You're running
locally without the `[heavy]` extra. Either install with `[heavy]` or stick
to the default `fast` model.

---

## Where this fits in the LatentScore project

This `demo/` folder is one piece of the broader artifact. Other entry points:

- **Library**: `pip install latentscore`, then `import latentscore as ls`.
  Top-level [README.md](../README.md) has the API and the Colab quickstart.
- **Paper code**: the data pipeline that built the embedding map lives in
  `data_work/`.
- **Doctor**: `latentscore doctor --strict --offline` verifies a clean install
  from the CLI.

Both the demo and the library are released under Apache 2.0 (see
[LICENSE](../LICENSE)).
