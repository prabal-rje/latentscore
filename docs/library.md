# LatentScore Library Guide

> 🟧 **Don't want to install?** [Try the SDK in Colab](https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb) — free CPU runtime, no setup, no key required.

## Contents

**Scenarios:**

- ["I just want sound"](#i-just-want-sound)
- ["I'm gonna play this live"](#im-gonna-play-this-live)
- ["I want to steer it live"](#i-want-to-steer-it-live)
- ["I want to nudge a vibe"](#i-want-to-nudge-a-vibe)
- ["I want full control over every knob"](#i-want-full-control-over-every-knob)
- ["I want a smarter model (BYO LLM)"](#i-want-a-smarter-model-byo-llm)

**More:**

- [Other built-in models](#other-built-in-models)
- [Advanced](#advanced)

**Reference:**

- [Parameter reference](#parameter-reference)
- [Audio contract](#audio-contract)

---

## "I just want sound"

```python
import latentscore as ls

audio = ls.render("deep underwater cave")
audio.play()
audio.save("cave.wav")
```

- `render(...)` returns an `Audio` object.
- `Audio` supports `.play()`, `.save("file.wav")`, and `np.asarray(audio)`.

## "I'm gonna play this live"

Chain a few vibes together with crossfades. One call, one continuous stream:

```python
import latentscore as ls

ls.stream(
    "morning coffee shop",
    "critical alert",
    "tension over a treasured object",
    duration=45,        # total seconds, split evenly across the vibes
    transition=3.0,     # crossfade seconds
).play()
```

## "I want to steer it live"

When you don't know the next vibe up front — driving it off user input,
sensor data, an LLM, whatever — hand `ls.live(...)` a generator instead
of fixed arguments. The engine pulls the next item when it's ready to
transition, and crossfades into it.

```python
import asyncio
import latentscore as ls
from collections.abc import AsyncIterator


async def my_set() -> AsyncIterator[str]:
    yield "morning coffee shop"
    await asyncio.sleep(10)

    yield "nintendo nes mario game"
    await asyncio.sleep(10)

    yield "critical alert"


ls.live(my_set(), transition_seconds=3.0).play()
```

Sync generators work too (just `def` / `yield`, no `async`). The
`await asyncio.sleep(...)` calls above are stand-ins for "wait for the
next event" — replace them with whatever real signal your application
uses (a `queue.get()`, a websocket message, a UI button, an LLM call).

> `.play()` is what makes time pass for the generator — it consumes
> chunks at the speaker's real-time rate, which gives the generator
> wall-clock time to yield the next item. `.collect()` won't do that
> (no backpressure → generator never gets to advance).

## "I want to nudge a vibe"

Start from a vibe string, override specific knobs:

```python
import latentscore as ls

ls.render(
    "morning coffee shop",
    update=ls.MusicConfigUpdate(
        tempo="very_fast",
        brightness="very_dark",
        echo="infinite",
    ),
).play()
```

Or the equivalent with relative `Step(±N)` (stops at the highest or lowest value):

```python
from latentscore.config import Step

ls.render(
    "morning coffee shop",
    update=ls.MusicConfigUpdate(
        tempo=Step(+4),
        brightness=Step(-4),
        echo=Step(+4),
    ),
).play()
```

## "I want full control over every knob"

Skip the vibe string and the retrieval step entirely. Build a
`MusicConfig` directly with human-readable labels:

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

Valid labels per field are in the [parameter reference](#parameter-reference) below.

## "I want a smarter model (BYO LLM)"

The default `"fast"` model is a CPU embedding lookup — instant, no key
required, ships with the core install. To route through an LLM instead,
install the extra and name a model:

```bash
pip install "latentscore[external]"
```

This example uses Google's Gemini, so it needs `GEMINI_API_KEY`. Grab
a free one at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey):

```python
import os
import latentscore as ls

os.environ["GEMINI_API_KEY"] = "your-key-here"

ls.render(
    "a father's eternal memory",
    model="external:gemini/gemini-3-flash-preview",
).play()
```

For other providers, set their `{PROVIDER}_API_KEY` (e.g.
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — full list of [100+ providers](https://docs.litellm.ai/docs/providers).

LLM responses carry richer metadata than the lookup path:

```python
audio = ls.render("a father's eternal memory", model="external:gemini/gemini-3-flash-preview")

if audio.metadata is not None:
    print(audio.metadata.title)      # the LLM's chosen title
    print(audio.metadata.thinking)   # the LLM's reasoning
    print(audio.metadata.config)     # the MusicConfig it chose
```

> LLM models are slower than `"fast"` (network round-trips) and can
> occasionally return invalid configs. `"fast"` is recommended for
> production use.

## Other built-in models

Beyond the default `"fast"` and `"external:*"` for BYO LLM, two more
models ship with the library:

### `"fast_heavy"` — CLAP audio-embedding retrieval

Same retrieval idea as `fast`, but matches against CLAP audio
embeddings (512-dim) instead of MiniLM text embeddings.

Often sharper for sonically-specific prompts ("muffled bass through a
wall", "glass shattering in slow motion") than scene descriptions.

```bash
pip install "latentscore[heavy]"
```

```python
import latentscore as ls

ls.render("muffled bass through a wall", model="fast_heavy").play()
```

Trade-off: CLAP weights are ~1.8 GB (vs ~90 MB for MiniLM), so first-call
download is slower and disk footprint is bigger.

### 🚧 `"expressive"` — local LLM (Gemma 3 270M, CPU)

> [!WARNING]
> 🚧 ⚠️ **Experimental — not recommended for production.**
> ~30–100 s per render and barely beats a random baseline on the
> CLAP benchmark. Useful only if you want fully offline operation
> and don't care about latency or quality.

Runs a small Gemma 3 LLM locally via `transformers` on CPU.

```bash
pip install "latentscore[expressive]"
latentscore download expressive
```

```python
ls.render("a father's eternal memory", model="expressive").play()
```

## Advanced

### Prefetching model assets

The first `render()` call downloads model weights and the embedding map
on demand, which can look like a hang for 30–60 seconds. Call
`ls.prefetch(...)` once at startup if you want that download to be
explicit:

```python
import latentscore as ls

ls.prefetch("fast")           # ~90 MB MiniLM + embedding map
ls.prefetch("fast_heavy")     # ~1.8 GB LAION-CLAP weights
```

### Render hooks

`render(...)` accepts a `hooks` argument that fires a callback at each
lifecycle stage — useful for progress UIs or measuring which stage
takes how long:

```python
import latentscore as ls

hooks = ls.RenderHooks(
    on_start=lambda: print("start"),
    on_model_start=lambda model: print(f"model: {model}"),
    on_synth_start=lambda: print("synth_start"),
    on_end=lambda: print("end"),
)

ls.render("underwater cave", hooks=hooks).play()
```

Pass an empty `RenderHooks()` to suppress the default Rich indicators.

**Streaming?** `ls.stream(...)` and `ls.live(...)` use a different
hooks shape — `ls.StreamHooks(on_event=callback)`, a single callback
receiving a tagged `StreamEvent` you pattern-match on. See
[`data_work/13_live_timing.py`](../data_work/13_live_timing.py) for a
worked example with timestamp logging across the full set of streaming
events.

---

## Parameter reference

Full `MusicConfig` schema — 34 fields across five groups. Fields marked
⋆ accept relative `Step(±N)` adjustments in `MusicConfigUpdate`
(everything else takes absolute labels only).

**Type column key:** *Ord. label* = ordered enum (`Step` works if ⋆);
*Enum* = unordered choices; *Bnd. int* = bounded integer;
*Boolean* = `true` / `false`; *Style sel.* = unordered layer-style selection.

### Global parameters (8)

| Field | Type | Allowed values |
|-------|------|----------------|
| `tempo` ⋆ | Ord. label | `very_slow` `slow` `medium` `fast` `very_fast` |
| `root` | Enum | `c` `c#` `d` `d#` `e` `f` `f#` `g` `g#` `a` `a#` `b` |
| `mode` | Enum | `major` `minor` `dorian` `mixolydian` |
| `brightness` ⋆ | Ord. label | `very_dark` `dark` `medium` `bright` `very_bright` |
| `space` ⋆ | Ord. label | `dry` `small` `medium` `large` `vast` |
| `density` ⋆ | Bnd. int | `2` `3` `4` `5` `6` |
| `motion` ⋆ | Ord. label | `static` `slow` `medium` `fast` `chaotic` |
| `attack` | Enum | `soft` `medium` `sharp` |

### Orchestration layers (6)

| Field | Type | Allowed values |
|-------|------|----------------|
| `bass` | Style sel. | `drone` `sustained` `pulsing` `walking` `fifth_drone` `sub_pulse` `octave` `arp_bass` |
| `pad` | Style sel. | `warm_slow` `dark_sustained` `cinematic` `thin_high` `ambient_drift` `stacked_fifths` `bright_open` |
| `melody` | Style sel. | `procedural` `contemplative` `rising` `falling` `minimal` `ornamental` `arp_melody` `contemplative_minor` `call_response` `heroic` |
| `rhythm` | Style sel. | `none` `minimal` `heartbeat` `soft_four` `hats_only` `electronic` `kit_light` `kit_medium` `military` `tabla_essence` `brush` |
| `texture` | Style sel. | `none` `shimmer` `shimmer_slow` `vinyl_crackle` `breath` `stars` `glitch` `noise_wash` `crystal` `pad_whisper` |
| `accent` | Style sel. | `none` `bells` `pluck` `chime` `bells_dense` `blip` `blip_random` `brass_hit` `wind` `arp_accent` `piano_note` |

### Spatial / texture (5)

| Field | Type | Allowed values |
|-------|------|----------------|
| `stereo` ⋆ | Ord. label | `mono` `narrow` `medium` `wide` `ultra_wide` |
| `depth` | Boolean | `true` `false` |
| `echo` ⋆ | Ord. label | `none` `subtle` `medium` `heavy` `infinite` |
| `human` ⋆ | Ord. label | `robotic` `tight` `natural` `loose` `drunk` |
| `grain` | Enum | `clean` `warm` `gritty` |

### Melody generation (10)

| Field | Type | Allowed values |
|-------|------|----------------|
| `melody_engine` | Enum | `pattern` `procedural` |
| `phrase_len_bars` | Bnd. int | `2` `4` `8` |
| `melody_density` | Ord. label | `very_sparse` `sparse` `medium` `busy` `very_busy` |
| `syncopation` | Ord. label | `straight` `light` `medium` `heavy` |
| `swing` | Ord. label | `none` `light` `medium` `heavy` |
| `motif_repeat_prob` | Ord. label | `rare` `sometimes` `often` |
| `step_bias` | Enum | `step` `balanced` `leapy` |
| `chromatic_prob` | Ord. label | `none` `light` `medium` `heavy` |
| `register_min_oct` | Bnd. int | `1` – `8` |
| `register_max_oct` | Bnd. int | `1` – `8` (must exceed `register_min_oct`) |

### Harmony (5)

| Field | Type | Allowed values |
|-------|------|----------------|
| `cadence_strength` | Ord. label | `weak` `medium` `strong` |
| `tension_curve` | Enum | `arc` `ramp` `waves` |
| `harmony_style` | Enum | `auto` `pop` `jazz` `cinematic` `ambient` |
| `chord_change_bars` | Ord. label | `very_slow` `slow` `medium` `fast` |
| `chord_extensions` | Enum | `triads` `sevenths` `lush` |

## Audio contract

- dtype: `float32`
- range: `[-1, 1]`
- sample rate: `44100` (`ls.SAMPLE_RATE`)
- shape: `(n,)` (mono)

```python
import numpy as np
import latentscore as ls

samples = np.asarray(ls.render("deep underwater cave"))  # NDArray[np.float32]
```
