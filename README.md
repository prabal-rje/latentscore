# LatentScore

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guprab/latentscore/blob/main/notebooks/quickstart.ipynb)

**Generate ambient music from text. Locally. No GPU required.**

```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

That's it. One line. You get audio playing on your speakers.

> ⚠️ **Alpha** &mdash; under active development. API may change between versions. [Read more about how it works](https://substack.com/home/post/p-184245090).

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

---

## CLI

```bash
latentscore doctor                       # check setup and model availability
latentscore demo                         # render and play a sample
latentscore demo --duration 30           # 30-second demo
latentscore demo --output ambient.wav    # save to file
```

---

## Quick Start

### Render and play

```python
import latentscore as ls

audio = ls.render("warm sunset over water", duration=10.0)
audio.play()              # plays on your speakers
audio.save("output.wav")  # save to WAV
```

### Different vibes

```python
ls.render("jazz cafe at midnight").play()
ls.render("thunderstorm on a tin roof").play()
ls.render("lo-fi study beats").play()
```

---

## Controlling the Sound

### MusicConfig (full control)

Build a config directly with human-readable labels:

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

### MusicConfigUpdate (tweak a vibe)

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

`Step(+1)` moves one level up the scale, `Step(-1)` moves one down. Saturates at boundaries.

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

---

## Streaming

Chain vibes together with smooth crossfade transitions:

```python
import latentscore as ls

stream = ls.stream(
    "morning coffee",
    "afternoon focus",
    "evening wind-down",
    duration=60,       # 60 seconds per vibe
    transition=5.0,    # 5-second crossfade
)
stream.play()

# Or collect and save
stream.collect().save("session.wav")
```

---

## Live Streaming

For dynamic, interactive use (games, installations, adaptive UIs), use a generator to feed vibes and steer the music in real time:

```python
import asyncio
from collections.abc import AsyncIterator

import latentscore as ls
from latentscore.config import Step


async def my_set() -> AsyncIterator[str | ls.MusicConfigUpdate]:
    yield "warm jazz cafe at midnight"
    await asyncio.sleep(8)

    # Absolute override: switch to bright electronic
    yield ls.MusicConfigUpdate(tempo="fast", brightness="very_bright", rhythm="electronic")
    await asyncio.sleep(8)

    # Relative nudge: dial brightness back down, add more echo
    yield ls.MusicConfigUpdate(brightness=Step(-2), echo=Step(+1))


session = ls.live(my_set(), transition_seconds=2.0)
session.play(seconds=30)
```

Sync generators work too &mdash; use `Iterator` instead of `AsyncIterator` and `time.sleep` instead of `await asyncio.sleep`.

---

## Async API

For web servers and async apps:

```python
import asyncio
import latentscore as ls


async def main() -> None:
    audio = await ls.arender("neon city rain")
    audio.save("neon.wav")


asyncio.run(main())
```

---

## Bring Your Own LLM

Use any LLM through [LiteLLM](https://docs.litellm.ai/docs/providers) &mdash; OpenAI, Anthropic, Google, Mistral, Groq, and [100+ others](https://docs.litellm.ai/docs/providers). LiteLLM is included with latentscore.

```python
import latentscore as ls

# Gemini (free tier available)
ls.render("cyberpunk rain on neon streets", model="external:gemini/gemini-3-flash-preview").play()

# Claude
ls.render("cozy library with rain outside", model="external:anthropic/claude-sonnet-4-5-20250929").play()

# GPT
ls.render("space station ambient", model="external:openai/gpt-4o").play()
```

API keys are read from environment variables automatically (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

### LLM Metadata

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

> **Note:** LLM models are slower than the default `fast` model (network round-trips) and can occasionally produce invalid configs. The `fast` model is recommended for production use.

---

## How It Works

You give LatentScore a **vibe** (a short text description) and it generates ambient music that matches.

Under the hood, the default `fast` model uses **embedding-based retrieval**: your vibe text gets embedded with a sentence transformer, then matched against a curated library of 10,000+ music configurations using cosine similarity. The best-matching config drives a real-time audio synthesizer.

This approach is **instant** (sub-second), **100% reliable** (no LLM hallucinations), and produces the highest-quality results. Our [CLAP benchmarks](https://huggingface.co/datasets/guprab/latentscore-clap-benchmark) showed that embedding retrieval outperforms Claude Opus 4.5 and Gemini 3 Flash at mapping vibes to music configurations.

---

## Audio Contract

All audio produced by LatentScore follows this contract:

- **Format:** `float32` mono
- **Sample rate:** `44100` Hz
- **Range:** `[-1.0, 1.0]`
- **Shape:** `(n,)` numpy array

```python
import numpy as np
import latentscore as ls

audio = ls.render("deep ocean")
samples = np.asarray(audio)  # NDArray[np.float32]
```

---

## Additional Info

### Config Reference

Every `MusicConfig` field uses human-readable labels. Full reference:

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

### Local LLM (Expressive Mode)

> **Not recommended.** The default `fast` model is faster, more reliable, and produces higher-quality results. Expressive mode exists for experimentation only.

Runs a 270M-parameter Gemma 3 LLM locally. On macOS Apple Silicon, inference uses MLX (~5&ndash;15s). On CPU-only Linux/Windows, it uses transformers (30&ndash;120s per render). The local model can produce invalid configs and our benchmarks showed it barely outperforms a random baseline.

```bash
pip install 'latentscore[expressive]'
latentscore download expressive
```

```python
ls.render("jazz cafe at midnight", model="expressive").play()
```

### Research & Training Pipeline

The `data_work/` folder contains the full research pipeline: data preparation, LLM-based config generation, SFT/GRPO training on Modal, CLAP benchmarking, and model export.

See [`data_work/README.md`](data_work/README.md) and [`docs/architecture.md`](docs/architecture.md) for details.

### Contributing

See [`CONTRIBUTE.md`](CONTRIBUTE.md) for environment setup and contribution guidelines.

See [`docs/coding-guidelines.md`](docs/coding-guidelines.md) for code style requirements.
