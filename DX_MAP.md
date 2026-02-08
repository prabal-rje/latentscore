# DX Map

Purpose: live map of `latentscore/dx.py` and its integration points.

## Overview
- `latentscore/dx.py` provides the higher-level DX layer on top of `latentscore.main`.
- It wraps raw streaming/render APIs into `Audio`/`AudioStream`, and offers `Track`/`Playlist` helpers.

## Key Types
- `TrackContent`: `str | MusicConfig | MusicConfigUpdate`.
- `Audio`: wrapper over numpy samples; supports `play`, `save`, `to_numpy`.
- `AudioStream`: wrapper that exposes sync + async iteration, plus `collect`, `save`, `play`.
- `Track`: `{content, duration, transition, name}`; converts to `Streamable`.
- `Playlist`: collection of `Track` with `stream`, `render`, `play` helpers.

## Entry Points
- `render(vibe_or_config, ...)`:
  - If `vibe_or_config` is `str`: delegates to `latentscore.main.render_raw`.
  - If config/update: merges into internal config then synthesizes via `assemble`.
- `stream(*items, duration=None, transition=5.0, transition_seconds=None, ...)`:
  - If `duration` is set: splits time across items (legacy behavior) and converts to `Streamable`.
  - If `duration` is `None`: treats items as instructions and streams indefinitely until new items arrive.
  - Items can be `Track`, `Streamable`, or `TrackContent`.
- `_stream_from_items(...)`: core bridge into `stream_raw`/`astream_raw`.

## Main API Dependencies
- `latentscore.main.stream_raw`/`astream_raw` for streaming.
- `latentscore.main.render_raw` for prompt-based render.
- `Streamable`, `StreamHooks`, `RenderHooks` for events + lifecycle.

## Playback + IO
- Uses `latentscore.playback.play_stream` and `write_wav` for output.

## Current Simplification Plan
- Remove speculative preview pathways from DX surface.
- Make streaming instruction-based by default (like `new.py`).
- Keep fallback behavior, but avoid prefetch/speculative execution.
- Keep `Track`/`Playlist` as optional helpers (compat), but allow plain instruction iterables.

## Update Log
- 2026-02-04: created map.
- 2026-02-04: DX simplified (instruction-based streaming, preview removed; fallback retained).
