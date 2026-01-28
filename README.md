# LatentScore

> ⚠️ **Alpha**: This library is under active development. API may change between versions.

Generate ambient music from text descriptions. Locally. No GPU required.

Read more about how it works [here](https://substack.com/home/post/p-184245090).
```python
import latentscore as ls

ls.render("warm sunset over water").play()
```

## Install

### Conda
```bash
conda create -n latentscore python=3.10
conda activate latentscore
conda install pip

pip install latentscore
```

### Pip
```bash
python -m venv .venv
source .venv/bin/activate

pip install latentscore
```

> Requires Python 3.10. If you don't have it: `brew install python@3.10` (macOS) or `pyenv install 3.10`

## Usage
```python
import latentscore as ls

# Render and play
audio = ls.render("warm sunrise over water")
audio.play()
audio.save("output.wav")
```

### Streaming
```python
import latentscore as ls

# Stream a single vibe
ls.stream("warm sunset over water", duration=120).play()

# Stream multiple vibes with crossfade
ls.stream(
    "morning coffee",
    "afternoon focus", 
    "evening wind-down",
    duration=60,
    transition=5.0,
).play()
```

### Async Streaming
```python
import latentscore as ls
import asyncio

async def main():
    items = [
        ls.Streamable(content="morning coffee", duration=30),
        ls.Streamable(content="afternoon focus", duration=30),
    ]
    async for chunk in ls.astream(items):
        # Process chunks as they arrive
        print(f"Got {len(chunk)} samples")

asyncio.run(main())
```

### Playlists
```python
import latentscore as ls

playlist = ls.Playlist(tracks=(
    ls.Track(content="morning energy", duration=60),
    ls.Track(content="deep focus", duration=120),
    ls.Track(content="evening calm", duration=60),
))
playlist.play()
```

### Modes

- **fast** (default): Embedding lookup. Instant.
- **expressive**: Local LLM. Slower, more creative. Run `latentscore download expressive` first.
- **external**: Route through Claude, Gemini, etc. Best quality, needs API key.
```python
# Use expressive mode
ls.render("jazz cafe at midnight", model="expressive").play()

# Use external LLM
ls.render(
    "cyberpunk rain",
    model="external:gemini/gemini-3-flash-preview",
    api_key="..."
).play()
```

## CLI
```bash
latentscore demo                  # Generate and play a sample
latentscore download expressive   # Fetch local LLM weights
latentscore doctor                # Check setup
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the data_work pipeline map and environment notes.

---

## Research & Training Pipeline (`data_work/`)

The `data_work/` folder hosts the full research + training pipeline (data prep, SFT/GRPO on Modal, CLAP benchmarking, eval suites, exports).  
If you want anything beyond the core library, start here:

- `data_work/README.md`
- `docs/architecture.md`

---

## Contributing

See `CONTRIBUTE.md` for environment setup and contribution guidelines.

See [`docs/coding-guidelines.md`](docs/coding-guidelines.md) for code style requirements.
