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

- `"fast"` (default): MiniLM text-embedding retrieval (384-dim, sub-second). Included in the core install.
- `"fast_heavy"`: LAION-CLAP audio-embedding retrieval (512-dim, matches text against rendered audio). Requires `pip install "latentscore[heavy]"`.
- `"expressive"` or `"local"`: local LLM (270M Gemma 3). Always runs through the CPU `transformers` backend in the current release &mdash; MLX integration is declared in pyproject markers but not yet wired into the runtime, so even on Apple Silicon you're on CPU. Expect ~30&ndash;100&nbsp;s per render. Requires `pip install "latentscore[expressive]"`.
- `"external:<model-name>"`: shorthand for `LiteLLMAdapter`. Requires `pip install "latentscore[external]"`.

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

```python
import numpy as np
import latentscore as ls

audio = ls.render("deep ocean")
samples = np.asarray(audio)  # NDArray[np.float32]
```

---

## Controlling the sound

### `MusicConfig` (full control)

Build a config directly with human-readable labels and skip the
text-prompt retrieval step entirely:

```python
import latentscore as ls

config = ls.MusicConfig(
    tempo="slow",
    brightness="dark",
    space="vast",
    density=3,
    bass="drone",
    pad="ambient_drift",
    melody="contemplative",
    rhythm="minimal",
    texture="shimmer",
    echo="heavy",
    root="d",
    mode="minor",
)

ls.render(config, duration=10.0).play()
```

### `MusicConfigUpdate` (tweak a vibe)

Start from a vibe and override specific parameters:

```python
import latentscore as ls

audio = ls.render(
    "morning coffee shop",
    duration=10.0,
    update=ls.MusicConfigUpdate(
        brightness="very_bright",
        rhythm="electronic",
    ),
)
audio.play()
```

### Relative steps

`Step(+1)` moves one level up the scale, `Step(-1)` moves one down.
Saturates at the boundaries.

```python
from latentscore.config import Step

audio = ls.render(
    "morning coffee shop",
    duration=10.0,
    update=ls.MusicConfigUpdate(
        brightness=Step(+2),   # two levels brighter
        space=Step(-1),        # one level less spacious
    ),
)
audio.play()
```

## Parameter reference

Every `MusicConfig` field uses human-readable labels.

| Field | Labels |
|-------|--------|
| `tempo` | `very_slow` `slow` `medium` `fast` `very_fast` |
| `brightness` | `very_dark` `dark` `medium` `bright` `very_bright` |
| `space` | `dry` `small` `medium` `large` `vast` |
| `motion` | `static` `slow` `medium` `fast` `chaotic` |
| `stereo` | `mono` `narrow` `medium` `wide` `ultra_wide` |
| `echo` | `none` `subtle` `medium` `heavy` `infinite` |
| `human` | `robotic` `tight` `natural` `loose` `drunk` |
| `attack` | `soft` `medium` `sharp` |
| `grain` | `clean` `warm` `gritty` |
| `density` | `2` `3` `4` `5` `6` |
| `root` | `c` `c#` `d` ... `a#` `b` |
| `mode` | `major` `minor` `dorian` `mixolydian` |

**Layer styles:**

| Layer | Styles |
|-------|--------|
| `bass` | `drone` `sustained` `pulsing` `walking` `fifth_drone` `sub_pulse` `octave` `arp_bass` |
| `pad` | `warm_slow` `dark_sustained` `cinematic` `thin_high` `ambient_drift` `stacked_fifths` `bright_open` |
| `melody` | `procedural` `contemplative` `rising` `falling` `minimal` `ornamental` `arp_melody` `contemplative_minor` `call_response` `heroic` |
| `rhythm` | `none` `minimal` `heartbeat` `soft_four` `hats_only` `electronic` `kit_light` `kit_medium` `military` `tabla_essence` `brush` |
| `texture` | `none` `shimmer` `shimmer_slow` `vinyl_crackle` `breath` `stars` `glitch` `noise_wash` `crystal` `pad_whisper` |
| `accent` | `none` `bells` `pluck` `chime` `bells_dense` `blip` `blip_random` `brass_hit` `wind` `arp_accent` `piano_note` |

## Bring your own LLM

Use any LLM through [LiteLLM](https://docs.litellm.ai/docs/providers)
&mdash; OpenAI, Anthropic, Google, Mistral, Groq, and
[100+ others](https://docs.litellm.ai/docs/providers). Install with
`pip install "latentscore[external]"`.

```python
import latentscore as ls

# Gemini (free tier available)
ls.render("cyberpunk rain on neon streets", model="external:gemini/gemini-3-flash-preview").play()

# Claude
ls.render("cozy library with rain outside", model="external:anthropic/claude-sonnet-4-5-20250929").play()

# GPT
ls.render("space station ambient", model="external:openai/gpt-4o").play()
```

API keys are read from environment variables automatically
(`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

### LLM metadata

External models return rich metadata alongside audio:

```python
audio = ls.render("cyberpunk rain", model="external:gemini/gemini-3-flash-preview")

if audio.metadata is not None:
    print(audio.metadata.title)      # e.g. "Neon Rain Drift"
    print(audio.metadata.thinking)   # the LLM's reasoning
    print(audio.metadata.config)     # the MusicConfig it chose
    for palette in audio.metadata.palettes:
        print([c.hex for c in palette.colors])
```

> **Note:** LLM models are slower than the default `fast` model
> (network round-trips) and can occasionally produce invalid configs.
> The built-in `fast` model is recommended for production use.

## Local LLM (`expressive` / `local`) ⚠️

> **Not recommended for general use.** The default `fast` and
> `fast_heavy` models are faster, more reliable, and produce
> higher-quality results. Expressive mode exists for experimentation.

Runs a 270M-parameter Gemma 3 LLM locally via the `transformers`
backend on CPU &mdash; **including on Apple Silicon**. MLX integration
is declared in pyproject markers but isn't actually wired into the
runtime yet, so every platform falls back to CPU `transformers`.
Expect ~30&ndash;100&nbsp;seconds per render on a laptop. The local
model can also produce invalid configs and our benchmarks showed it
barely outperforms a random baseline.

```bash
pip install 'latentscore[expressive]'
latentscore download expressive
```

```python
ls.render("jazz cafe at midnight", model="expressive").play()
```
