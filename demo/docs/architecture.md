# Architecture

How the three services in `docker compose up` fit together.

```
┌──────────────────────────────────┐
│  Browser                         │  React 19 + Vite + TypeScript
│  http://localhost:4242           │
└────────────────┬─────────────────┘
                 │  HTTP (Render mode) or WebSocket (Live mode)
                 ▼
┌──────────────────────────────────┐
│  FastAPI server                  │  ~900 LOC, single main.py
│  http://localhost:4244           │
│                                  │
│  • REST + WebSocket routes       │
│  • Session state (in-memory)     │
│  • DSP process pool              │
│  • WAV encoding                  │
│  • CLAP candidate scoring        │
└────────────────┬─────────────────┘
                 │  Python imports
                 ▼
┌──────────────────────────────────┐
│  latentscore library             │
│  • text → MusicConfig retrieval  │
│  • audio synthesis (`assemble`)  │
│  • streaming generator (`live`)  │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  JupyterLab (separate service)   │  quay.io/jupyter/scipy-notebook
│  http://localhost:8889           │  Same latentscore install as the
│                                  │  backend. Notebook lives at
│  • quickstart.ipynb              │  /home/jovyan/work/.
└──────────────────────────────────┘
```

The JupyterLab service is not in the request path of the demo UI. It
runs in parallel, bound to `127.0.0.1` only, with token auth disabled
so the URL just works. It exists so anyone running the demo also gets
a fully-working Python playground without `pip install`.

There is no database. Backend session state lives in an in-memory dict;
restart the backend container and it's gone. The demo is a stateless
interactive artifact, not a service.

## Terms

- **MiniLM** - a 384-dim sentence-transformer used to embed the user's
  text. ~90 MB on first download.
- **CLAP** - *Contrastive Language-Audio Pre-training* (LAION). A joint
  audio + text encoder. The `fast_heavy` model uses it to match a
  prompt against the *sound* of each library config, not just its
  description. ~1.8 GB checkpoint, only loaded when the `[heavy]`
  extra is installed.
- **MusicConfig** - the library's central data type. A frozen Pydantic
  model describing one piece of music (tempo, mode, instrument styles,
  density, etc.). See `latentscore/config.py`.
- **WAV @ 44.1 kHz / float32 / mono** - the audio format the library
  produces and the backend sends. ~7 MB per 60-second clip.

## Project structure

```
demo/
├── README.md            landing page
├── docker-compose.yml   3-service local stack
├── Dockerfile           single-image multi-stage build (kept for
│                        platforms that want a one-image deploy)
│
├── backend/             FastAPI service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/            React + Vite app served by nginx in prod
│   ├── src/
│   ├── package.json
│   └── Dockerfile
│
├── notebook/            JupyterLab with latentscore pre-installed
│   └── Dockerfile
│
└── docs/                deeper reading (you are here)
    ├── architecture.md
    ├── development.md
    └── troubleshooting.md
```

## How features map to code

| Feature | Backend handler | Frontend component |
|---|---|---|
| Type a vibe → audio | `POST /api/generate` (main.py:598) | `VibeInput.tsx` |
| Click a preset | Same `/api/generate` | `PresetGrid.tsx` |
| Edit parameters → Apply | `POST /api/update-params` (main.py:656) | `ParamPanel.tsx` |
| Live-stream WebSocket | `WS /ws/stream/{session_id}` (main.py:808) | `useAudioPlayer.ts` |
| Queue a playlist | `POST /api/playlist/create` (main.py:882) | `DemoPage.tsx` |
| Save WAV | `GET /api/render/{session_id}` (main.py:790) | `DemoPage.tsx` |
| Best-of-N candidate scoring | `POST /api/select-best-config` (main.py:690) | (internal) |
| Available models | `GET /api/capabilities` (main.py:576) | `App.tsx` |

The `latentscore` calls themselves are mostly in `main.py` lines
~150-550 - `ls.live(...)`, `FastEmbeddingModel`, `FastHeavyModel`, and
the parameter-mapping helpers from `latentscore.config`.
