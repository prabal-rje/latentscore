# LatentScore Library DX

## Quickstart

```python
from latentscore import MusicConfigUpdate, render, save_wav

update = MusicConfigUpdate(tempo="medium")
audio = render("warm sunrise over water", duration=8, model="fast", update=update)
save_wav("output.wav", audio)
```

- Default models run locally (no API keys required).
- First expressive-model run may download weights (~1.2GB) and cache them at `~/.cache/latentscore/models`.

## Streaming

```python
from latentscore import (
    FirstAudioSpinner,
    StreamHooks,
    Streamable,
    MusicConfigUpdate,
    stream,
    stream_texts,
    astream,
)

hooks = StreamHooks(on_event=lambda event: print(event.kind))
spinner = FirstAudioSpinner(delay=0.35)

for chunk in stream_texts(
    ["warm sunrise", "darker, slower"],
    duration=60.0,
    transition_duration=1.0,
    chunk_seconds=0.5,
    model="fast",
    prefetch_depth=1,
    hooks=hooks,
):
    ...

items = [
    Streamable(content="warm sunrise", duration=30.0),
    Streamable(content=MusicConfigUpdate(tempo="slow"), duration=30.0),
]

for chunk in stream(
    items,
    chunk_seconds=0.5,
    model="fast",
    prefetch_depth=1,
    preview_policy="embedding",
    fallback="keep_last",
    hooks=spinner.hooks(),
):
    ...

async for chunk in astream(
    items,
    chunk_seconds=0.5,
    model="fast",
    prefetch_depth=1,
    preview_policy="embedding",
    fallback="keep_last",
    hooks=spinner.hooks(),
):
    ...
```

- `prefetch_depth` resolves future vibe strings in the background to reduce transition stalls.
- `preview_policy="embedding"` plays a fast preview config while waiting on slow LLMs.
- `fallback="keep_last"` keeps the current config if an LLM fails (other options: `embedding`, `none`, or a fixed config/update).
- `StreamHooks` emits lifecycle events like `first_config_ready` and `first_audio_chunk`.
- `FirstAudioSpinner` shows a delayed spinner while the first audio chunk is preparing; when preview mode is active, it explains that speculative preview is running and how to disable it (`preview_policy="none"`).

## Audio Contract

- dtype: `float32`
- range: `[-1, 1]`
- sample rate: `44100`
- shape: `(n,)` (mono)

## Advanced

- `latentscore demo` renders a short clip to `demo.wav`.
- `latentscore download expressive` prefetches the expressive model weights.
- `latentscore doctor` prints cache paths and availability hints.
- First-time expressive-model downloads show a spinner; run `latentscore doctor` and prefetch missing models in production to avoid runtime downloads.
- For numeric control, use `MusicConfigInternal` or `MusicConfigUpdateInternal` instead of the string-literal configs.

### Bring Your Own LLM

```python
from latentscore import render
from latentscore.providers import LiteLLMAdapter

adapter = LiteLLMAdapter(model="gpt-4o-mini")
audio = render("late night neon", model=adapter)
```
