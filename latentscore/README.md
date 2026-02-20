# Latentscore DX

This package wraps the core LatentScore API with a small DX layer for audio‑first workflows.
For installation and the main usage guide, see the root README.

## Quickstart

```python
import latentscore as ls

audio = ls.render("warm sunrise over water")
audio.play()  # uses sounddevice/simpleaudio; install ipython for notebooks
audio.save(".examples/quickstart.wav")
```

## DX surface

- `Audio`: `.save(path)`, `.play()`, `np.asarray(audio)`
- `AudioStream`: iterates chunks, `.save(path)`, `.play()`, `.collect()`
- `Track` + `Playlist`: composition helpers for longer flows
- `stream(...)`: streaming + optional speculative preview

## Models

- `"fast_heavy"` (recommended): LAION-CLAP audio-embedding retrieval (512-dim, matches text against rendered audio, 71% higher quality than `fast`)
- `"fast"` (default): MiniLM text-embedding retrieval (384-dim, marginally faster but significantly lower quality)
- `"expressive"` / `"local"`: local LLM (CUDA where available; otherwise CPU)
- `"external:<model-name>"`: LiteLLM adapter shorthand

## Demo

```bash
python -m latentscore.demo
```

## Doctor

```bash
latentscore doctor
```

## More details

See `docs/latentscore-dx.md` for the full DX guide, model options, and audio contract.
