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
from latentscore import Streamable, MusicConfigUpdate, stream, stream_texts, astream

for chunk in stream_texts(
    ["warm sunrise", "darker, slower"],
    duration=60.0,
    transition_duration=1.0,
    chunk_seconds=0.5,
    model="fast",
):
    ...

items = [
    Streamable(content="warm sunrise", duration=30.0),
    Streamable(content=MusicConfigUpdate(tempo="slow"), duration=30.0),
]

for chunk in stream(items, chunk_seconds=0.5, model="fast"):
    ...

async for chunk in astream(items, chunk_seconds=0.5, model="fast"):
    ...
```

## Audio Contract

- dtype: `float32`
- range: `[-1, 1]`
- sample rate: `44100`
- shape: `(n,)` (mono)

## Advanced

- `latentscore demo` renders a short clip to `demo.wav`.
- `latentscore download expressive` prefetches the expressive model weights.
- `latentscore doctor` prints cache paths and availability hints.
- For numeric control, use `MusicConfigInternal` or `MusicConfigUpdateInternal` instead of the string-literal configs.

### Bring Your Own LLM

```python
from latentscore import render
from latentscore.providers import LiteLLMAdapter

adapter = LiteLLMAdapter(model="gpt-4o-mini")
audio = render("late night neon", model=adapter)
```
