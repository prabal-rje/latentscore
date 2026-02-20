# LatentScore Library DX

## Tier 0: I just want sound

```python
import latentscore as ls

audio = ls.render("underwater cave")
audio.play()
audio.save("cave.wav")
```

- `render(...)` returns an `Audio` object (not a raw numpy array).
- `Audio` supports `.play()`, `.save()`, and `np.asarray(audio)`.

## Tier 1: I want a stream of chunks

```python
import latentscore as ls

for chunk in ls.stream("dark ambient", "sunrise"):
    speaker.write(chunk)
```

- `stream(...)` yields `np.float32` mono chunks.
- `duration` is total duration across all items (split evenly).
- `AudioStream` also supports `.save()` and `.play()`.
- `stream(...)` accepts a single sequence of items (e.g., `ls.stream(["dark ambient", "sunrise"])`).

## Live generator stream (dynamic playlist)

```python
import latentscore as ls
from collections.abc import Iterable
from latentscore.playback import play_stream

def live_items() -> Iterable[ls.Streamable]:
    for vibe in ["misty harbor", "neon rain", "quiet orchard"]:
        yield ls.Streamable(content=vibe, duration=6.0, transition_duration=1.5)

chunks = ls.stream_raw(live_items(), chunk_seconds=1.0, model="fast")
play_stream(chunks, sample_rate=ls.SAMPLE_RATE)
```

## Tier 2: Same stream, but with knobs

```python
import latentscore as ls

async for chunk in ls.stream(
    "dark ambient",
    "sunrise",
    duration=120,
    transition=5,
    chunk_seconds=1.0,
    model="fast",
):
    await speaker.write(chunk)
```

- `stream(...)` supports both `for` and `async for`.
- `chunk_seconds` controls chunk sizing.
- `preview=True` uses the fast model as a speculative preview while a slower model loads.

## Tier 3: Composition primitives

```python
import latentscore as ls

playlist = ls.Playlist(
    tracks=[
        ls.Track(content="dark ambient", duration=60),
        ls.Track(content="sunrise", duration=120, transition=10),
        ls.Track(content=ls.MusicConfig(tempo="fast", mode="minor"), duration=60),
        ls.Track(content=ls.MusicConfigUpdate(tempo="slow", brightness="dark"), duration=60),
    ]
)
playlist.stream().play()
```

- `Track` accepts `str`, `MusicConfig`, or `MusicConfigUpdate`.
- `Playlist.stream()` returns the same dual sync/async `AudioStream`.

## Model selection

- `"fast"` (default): MiniLM text-embedding retrieval (384-dim, sub-second).
- `"fast_heavy"`: LAION-CLAP audio-embedding retrieval (512-dim, matches text against rendered audio).
- `"expressive"` or `"local"`: local LLM (Apple Silicon uses MLX; other platforms use transformers with CUDA if available, otherwise CPU; 4-bit bitsandbytes is used when available).
- `"external:<model-name>"`: shorthand for `LiteLLMAdapter`.

```python
import latentscore as ls

audio = ls.render("late night neon", model="external:gemini/gemini-3-flash-preview")
```

For advanced LiteLLM control (timeouts, API keys, etc.), instantiate the adapter:

```python
import os
import latentscore as ls
from latentscore.providers.litellm import LiteLLMAdapter

adapter = LiteLLMAdapter(
    model="external:gemini/gemini-3-flash-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
    litellm_kwargs={"timeout": 60},
)

audio = ls.render("late night neon", model=adapter)
```

You can also pass a typed external spec instead of instantiating the adapter:

```python
import latentscore as ls

spec = ls.ExternalModelSpec(
    model="gemini/gemini-3-flash-preview",
    api_key=None,
    litellm_kwargs={"timeout": 60},
)
audio = ls.render("late night neon", model=spec)
```

## Playback notes

CLI playback uses sounddevice/simpleaudio by default. Install `ipython` if you want inline notebook playback. If playback is unavailable, `.play()` raises a friendly error that suggests using `.save()`.

## Progress indicators

- `render(...)` and `stream(...)` show Rich spinners in TTYs (model load, LLM config, audio generation).
- `.play()` shows a progress bar for buffered audio and a music-note spinner for streams.
- To silence indicators, pass empty hooks: `hooks=ls.RenderHooks()` or `hooks=ls.StreamHooks()`.

## Advanced: raw API

Core functions remain available for advanced use:

```python
from latentscore import render_raw, stream_raw, astream_raw, Streamable
```

- `stream_raw(...)` expects an iterable of `Streamable`.
- `astream_raw(...)` yields chunks asynchronously without the `AudioStream` wrapper.

## Render hooks

Render hooks help surface progress during blocking renders:

```python
import latentscore as ls

events: list[str] = []
hooks = ls.RenderHooks(
    on_start=lambda: events.append("start"),
    on_model_start=lambda model: events.append(f"model:{model}"),
    on_synth_start=lambda: events.append("synth_start"),
    on_end=lambda: events.append("end"),
)

audio = ls.render("underwater cave", hooks=hooks)
```

## Audio Contract

- dtype: `float32`
- range: `[-1, 1]`
- sample rate: `44100`
- shape: `(n,)` (mono)
