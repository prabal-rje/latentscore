# Latentscore DX

This package wraps the core LatentScore API with a small DX layer for audio-first workflows.

## Quickstart

```python
import latentscore as ls

audio = ls.render("warm sunrise over water")
audio.play()  # uses sounddevice/simpleaudio; install ipython for notebooks
audio.save(".examples/quickstart.wav")
```

## DX objects

- `Audio`: `.save(path)`, `.play()`, `np.asarray(audio)`
- `AudioStream`: iterates chunks, `.save(path)`, `.play()`, `.collect()`
- `Track` + `Playlist`: composition helpers for longer flows

## Streaming

```python
import latentscore as ls

for chunk in ls.stream("dark ambient", "sunrise", duration=60, transition=5):
    speaker.write(chunk)
```

## Composition

```python
import latentscore as ls

playlist = ls.Playlist(
    tracks=[
        ls.Track(content="dark ambient", duration=20, transition=3),
        ls.Track(content=ls.MusicConfig(tempo="fast"), duration=20, transition=3),
        ls.Track(content=ls.MusicConfigUpdate(brightness="dark"), duration=20, transition=3),
    ]
)

playlist.render().save(".examples/playlist.wav")
```

## Models

- `"fast"` (default): local embedding model
- `"expressive"` / `"local"`: local MLX LLM
- `"external:<model-name>"`: LiteLLM adapter shorthand

## Playback notes

Install `ipython` if you want inline notebook playback. CLI playback uses sounddevice/simpleaudio.

## Demo

Run from the repo root:

```bash
python -m latentscore.demo
```

Outputs land in `.examples/` at the project root.

Flags:

```bash
python -m latentscore.demo --model expressive
python -m latentscore.demo --model fast --vibe "late night neon"
python -m latentscore.demo --live
python -m latentscore.demo --external --api-key "$GEMINI_API_KEY"
```

You can also set the API key via env var:

```bash
export GEMINI_API_KEY="..."
python -m latentscore.demo --external
```

## Debugging

LatentScore writes full traces to `~/.cache/latentscore/logs/latentscore.log` by default (override with `LATENTSCORE_LOG_DIR`).

To echo stack traces in the terminal:

```bash
LATENTSCORE_DEBUG=1 python -m latentscore.demo
```

## Doctor

Run a quick environment check before demos or production:

```bash
latentscore doctor
```

It reports cache locations, whether the local models are present, and hints for prefetching.

## More details

See `docs/latentscore-dx.md` for the full tiered DX guide, model options, and audio contract.
