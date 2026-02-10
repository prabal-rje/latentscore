# LatentScore

**Generate ambient music from text. Locally. No GPU required.**

```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

That's it. One line. You get audio playing on your speakers.

> **Alpha** &mdash; under active development. API may change between versions. [Read more about how it works](https://substack.com/home/post/p-184245090).

---

## Install

**Requires Python 3.10&ndash;3.12.** If you don't have it: `brew install python@3.10` (macOS) or `pyenv install 3.10`.

```bash
pip install latentscore
```

Or with conda:

```bash
conda create -n latentscore python=3.10 -y
conda activate latentscore
pip install latentscore
```

Verify it works:

```bash
latentscore doctor    # check your setup
latentscore demo      # render a sample and play it
```

---

## Quick Start

### Render and save

```python
import latentscore as ls
from latentscore import Audio

audio: Audio = ls.render("warm sunrise over water")
audio.play()              # plays on your speakers
audio.save("output.wav")  # save to WAV
```

### Stream multiple vibes with crossfade

```python
import latentscore as ls
from latentscore import Audio, AudioStream

stream: AudioStream = ls.stream(
    "morning coffee",
    "afternoon focus",
    "evening wind-down",
    duration=60,       # 60 seconds per vibe
    transition=5.0,    # 5-second crossfade between vibes
)
stream.play()

# Or collect to a single Audio and save
collected: Audio = stream.collect()
collected.save("session.wav")
```

---

## How It Works

You give LatentScore a **vibe** (a short text description) and it generates ambient music that matches.

Under the hood, the default `fast` model uses **embedding-based retrieval**: your vibe text gets embedded with a sentence transformer, then matched against a curated library of 10,000+ music configurations using cosine similarity. The best-matching config drives a real-time audio synthesizer.

This approach is **instant** (sub-second), **100% reliable** (no LLM hallucinations), and produces the highest-quality results. Our [research benchmarks](data_work/README.md) showed that this simple retrieval approach outperforms GPT, Claude, and Gemini at mapping vibes to music configurations.

---

## Controlling the Sound

### Use a vibe string (easiest)

Just describe what you want:

```python
import latentscore as ls

ls.render("jazz cafe at midnight").play()
ls.render("thunderstorm on a tin roof").play()
ls.render("lo-fi study beats").play()
```

### Use MusicConfig (full control)

For precise control, build a `MusicConfig` directly:

```python
from latentscore import Audio, MusicConfig
import latentscore as ls

config: MusicConfig = MusicConfig(
    tempo="slow",            # very_slow, slow, medium, fast, very_fast
    brightness="dark",       # very_dark, dark, medium, bright, very_bright
    space="vast",            # dry, small, medium, large, vast
    density=3,               # 2 (sparse) to 6 (dense)
    bass="drone",            # drone, sustained, pulsing, walking, ...
    pad="ambient_drift",     # warm_slow, dark_sustained, cinematic, ...
    melody="contemplative",  # contemplative, rising, falling, minimal, ...
    rhythm="minimal",        # none, minimal, heartbeat, electronic, ...
    texture="shimmer",       # none, shimmer, vinyl_crackle, breath, stars, ...
    echo="heavy",            # none, subtle, medium, heavy, infinite
    root="d",                # c, c#, d, d#, e, f, f#, g, g#, a, a#, b
    mode="minor",            # major, minor, dorian, mixolydian
)

audio: Audio = ls.render(config)
audio.play()
```

### Tweak a vibe with MusicConfigUpdate

Start from a vibe and nudge specific parameters using relative steps:

```python
import latentscore as ls
from latentscore import Audio, MusicConfigUpdate
from latentscore.config import Step

audio: Audio = ls.render(
    "morning coffee shop",
    update=MusicConfigUpdate(
        brightness=Step(+2),   # two levels brighter
        space=Step(-1),        # one level less spacious
    ),
)
audio.play()
```

`Step(+1)` moves one level up the scale, `Step(-1)` moves one level down. Scales saturate at their boundaries (you can't go brighter than `very_bright`).

You can also set absolute values instead of relative steps:

```python
from latentscore import MusicConfigUpdate

update: MusicConfigUpdate = MusicConfigUpdate(
    brightness="very_bright",
    rhythm="electronic",
)
```

---

## Streaming

### Stream vibes with crossfade

```python
import latentscore as ls
from latentscore import AudioStream

stream: AudioStream = ls.stream(
    "morning coffee",
    "afternoon focus",
    "evening wind-down",
    duration=60,
    transition=5.0,
)
stream.play()
```

### Save a stream to file

```python
stream: AudioStream = ls.stream(
    "rainy window",
    "thunder rolling",
    duration=30,
    transition=3.0,
)
stream.save("rain_to_thunder.wav")
```

---

## Live Streaming with Generators

For dynamic, interactive applications (games, installations, adaptive UIs), use generators to feed vibes and steer the music in real time.

### Sync generator

```python
from collections.abc import Iterator

import latentscore as ls
import latentscore.dx as dx
from latentscore import LiveStream, MusicConfigUpdate, Streamable
from latentscore.config import Step


def ambient_journey() -> Iterator[Streamable]:
    # Start with a vibe
    yield Streamable(content="warm sunrise ambient", duration=30, transition_duration=3)

    # Steer brighter and faster using relative steps
    yield Streamable(
        content=MusicConfigUpdate(brightness=Step(+2), tempo=Step(+1)),
        duration=30,
        transition_duration=3,
    )

    # Switch to a completely different vibe
    yield Streamable(content="late night neon", duration=30, transition_duration=5)


live: LiveStream = dx.live(ambient_journey(), chunk_seconds=1.0, transition_seconds=3.0)
live.play()
```

### Async generator

Async generators let you await external events (sensor data, user input, API calls) between yields:

```python
import asyncio
from collections.abc import AsyncIterator

import latentscore as ls
import latentscore.dx as dx
from latentscore import LiveStream, MusicConfigUpdate, Streamable
from latentscore.config import Step


async def reactive_journey() -> AsyncIterator[Streamable]:
    yield Streamable(content="warm sunrise ambient", duration=30, transition_duration=3)

    await asyncio.sleep(5)  # wait for sensor/user input/API response

    # Steer with a relative step
    yield Streamable(
        content=MusicConfigUpdate(brightness=Step(+2)),
        duration=30,
        transition_duration=3,
    )

    await asyncio.sleep(5)

    # Or steer with absolute config values
    yield Streamable(
        content=MusicConfigUpdate(brightness="very_bright", rhythm="electronic"),
        duration=30,
        transition_duration=3,
    )


live: LiveStream = dx.live(reactive_journey(), chunk_seconds=1.0, transition_seconds=3.0)
live.play()
```

The generator is consumed lazily &mdash; you can yield items based on user input, sensor data, time of day, or anything else.

---

## Async API

For web servers, async apps, or when you need non-blocking audio generation:

### arender

```python
import asyncio

import latentscore as ls
from latentscore import Audio


async def main() -> None:
    audio: Audio = await ls.arender("neon city rain")
    audio.save("neon.wav")


asyncio.run(main())
```

### astream

```python
import asyncio
from collections.abc import AsyncIterator

import numpy as np
from numpy.typing import NDArray

import latentscore as ls
from latentscore import Streamable


async def main() -> None:
    async def vibes() -> AsyncIterator[Streamable]:
        yield Streamable(content="deep ocean ambient", duration=30, transition_duration=3)
        yield Streamable(content="forest rain", duration=30, transition_duration=3)

    async for chunk in ls.astream(vibes(), chunk_seconds=1.0, transition_seconds=3.0):
        samples: NDArray[np.float32] = chunk
        process(samples)  # your processing logic here


asyncio.run(main())
```

---

## Bring Your Own LLM

Want an LLM to interpret your vibes instead of the default embedding lookup? LatentScore supports **any model** through [LiteLLM](https://docs.litellm.ai/docs/providers) &mdash; OpenAI, Anthropic, Google, Mistral, Groq, and [100+ others](https://docs.litellm.ai/docs/providers).

```python
import latentscore as ls

# Use Gemini
ls.render(
    "cyberpunk rain on neon streets",
    model="external:gemini/gemini-3-flash-preview",
).play()

# Use Claude
ls.render(
    "cozy library with rain outside",
    model="external:anthropic/claude-sonnet-4-5-20250929",
).play()

# Use GPT
ls.render(
    "space station ambient",
    model="external:openai/gpt-4o",
).play()
```

LiteLLM reads API keys from environment variables automatically (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.). Or pass them explicitly:

```python
from latentscore import ExternalModelSpec

spec: ExternalModelSpec = ExternalModelSpec(
    model="gemini/gemini-3-flash-preview",
    api_key="your-api-key-here",
)

ls.render("late night neon", model=spec).play()
```

For advanced LiteLLM options (timeouts, retries, custom base URLs):

```python
from latentscore import ExternalModelSpec

spec: ExternalModelSpec = ExternalModelSpec(
    model="openai/gpt-4o",
    api_key="...",
    litellm_kwargs={"timeout": 60, "max_retries": 3},
)
```

> **Note:** LLM-based models are slower than the default `fast` model (network round-trips) and can occasionally fail due to API errors or output validation issues. The `fast` model is recommended for production use.

### Accessing LLM Metadata

When using an external LLM, the model returns rich metadata alongside the music config: a title, reasoning, color palettes, and the full config it chose. Access it via `audio.metadata`:

```python
from latentscore import Audio, GenerateResult, Palette

audio: Audio = ls.render(
    "cyberpunk rain on neon streets",
    model="external:gemini/gemini-3-flash-preview",
)

metadata: GenerateResult | None = audio.metadata
if metadata is not None:
    print(metadata.title)      # e.g. "Neon Rain Drift"
    print(metadata.thinking)   # the LLM's reasoning
    print(metadata.config)     # the MusicConfig it chose
    palettes: tuple[Palette, ...] = metadata.palettes
    for palette in palettes:
        print([c.hex for c in palette.colors])
```

The `metadata` field is `None` when using the default `fast` model (which doesn't produce metadata).

---

## Config Reference

Every `MusicConfig` field uses human-readable labels. Here's the full reference:

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

---

## CLI

```bash
latentscore demo                         # render and play a sample
latentscore demo --duration 30           # 30-second demo
latentscore demo --output ambient.wav    # save to file
latentscore doctor                       # check setup and model availability
```

---

## Audio Contract

All audio produced by LatentScore follows this contract:

- **Format:** `float32` mono
- **Sample rate:** `44100` Hz
- **Range:** `[-1.0, 1.0]`
- **Shape:** `(n,)` numpy array

Access the raw array directly:

```python
import numpy as np
from numpy.typing import NDArray

import latentscore as ls
from latentscore import Audio

audio: Audio = ls.render("deep ocean")
samples: NDArray[np.float32] = np.asarray(audio)
```

---

## Local LLM (Expressive Mode)

> **Not recommended.** The default `fast` model is faster, more reliable, and produces higher-quality results. Expressive mode exists for experimentation only.

> **Slow.** Runs a 270M-parameter Gemma 3 LLM locally on CPU. On macOS Apple Silicon, inference uses MLX (~5-15s). On CPU-only Linux/Windows, it uses transformers (**30-120 seconds** per render).

> **Unreliable.** The local LLM can produce invalid configs, hallucinate parameter values, or mode-collapse into repetitive outputs. Our benchmarks showed the fine-tuned local model barely outperforms a random baseline.

> **Blocking.** LLM inference runs in a Python thread. Due to the GIL, this blocks other Python code. Do not use in latency-sensitive applications.

If you still want to try it:

```bash
pip install 'latentscore[expressive]'
latentscore download expressive
```

```python
import latentscore as ls

ls.render("jazz cafe at midnight", model="expressive").play()
```

---

## Research & Training Pipeline

The `data_work/` folder contains the full research pipeline: data preparation, LLM-based config generation, SFT/GRPO training on Modal, CLAP benchmarking, and model export.

See [`data_work/README.md`](data_work/README.md) and [`docs/architecture.md`](docs/architecture.md) for details.

---

## Contributing

See [`CONTRIBUTE.md`](CONTRIBUTE.md) for environment setup and contribution guidelines.

See [`docs/coding-guidelines.md`](docs/coding-guidelines.md) for code style requirements.
