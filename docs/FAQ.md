# Frequently Asked Questions

If you have a question that isn't here, open an issue or check the
top-level [`README.md`](../README.md) and [`latentscore-dx.md`](latentscore-dx.md).

---

## Why does the first `ls.render(...)` call appear to hang?

The first text-prompt render silently downloads the MiniLM embedding
model (~90&nbsp;MB) and the embedding-map dataset (~few&nbsp;MB) from
Hugging Face. On a fresh machine this takes 30&ndash;60&nbsp;seconds and
looks indistinguishable from a frozen kernel.

Two fixes, both supported:

```python
# Python API: makes the download explicit with a visible progress bar
import latentscore as ls
ls.prefetch("fast")
ls.render("warm sunset over water").play()
```

```bash
# CLI equivalent
latentscore download fast
```

Subsequent calls hit the local cache and complete in &lt;1&nbsp;s. The
`fast_heavy` model has the same shape but downloads ~1.8&nbsp;GB of
LAION-CLAP weights instead of the 90&nbsp;MB MiniLM &mdash; expect
several minutes on first use.

---

## What's the difference between `fast` and `fast_heavy`?

Both are retrieval-based (no LLM, no hallucinations, no API keys).
They differ in what gets embedded:

| | `fast` (default) | `fast_heavy` |
|---|---|---|
| Embedder | MiniLM-L6-v2 (text encoder) | LAION-CLAP audio encoder |
| What gets matched | Your text vs. the library's text descriptions of each config | Your text vs. CLAP audio embeddings of each config's actual sound |
| Download | ~90&nbsp;MB | ~1.8&nbsp;GB |
| Latency | ~2&nbsp;s warm | ~2&nbsp;s warm |
| Install | core | `pip install "latentscore[heavy]"` |

Intuition: `fast` matches your text to **what the library says** about
each config. `fast_heavy` matches your text to **what each config
actually sounds like**. For prompts where audio properties matter more
than vocabulary (e.g. "a sound that feels like rain"), `fast_heavy`
often does better.

---

## I ran `pip install latentscore` and it succeeded, but `import latentscore` fails. What happened?

This is almost always a missing **system library** that the bundled
`soundfile` or `sounddevice` wheels link against at import time. The
PyPI wheels ship binary copies of `libsndfile` and `PortAudio`, but
they rely on the host OS providing the corresponding runtime `.so`
files.

Quick diagnostic:

```bash
latentscore doctor --strict --offline
```

The `audio_write` and `render_core` checks will fail with a clear
hint pointing at `libsndfile` / `portaudio` if that's the issue.

Fix per platform:

- **Linux:** `sudo apt-get install -y libsndfile1 libasound2`
  (Ubuntu/Debian) or equivalent. ALSA headers (`libasound2-dev`)
  are only needed for source compilation.
- **macOS:** the Homebrew Python comes with these by default. If
  it's still failing, `brew install sox` is usually enough.
- **Windows native:** see [Platform support](../README.md#platform-support)
  in the main README. WSL2 is the recommended path.

---

## Can I run this on Windows?

The honest matrix is in the main README's
[Platform support](../README.md#platform-support) section. Short
version:

- **Native Windows** &mdash; core and `[external]` should work
  (all deps ship Windows wheels) but we haven't tested. `[expressive]`
  also installs on Windows: `llama-cpp-python`, `mlx`, and
  `bitsandbytes` are excluded by pyproject markers, but the code
  falls back to the `transformers` backend which is in core. CPU
  inference of the 270M Gemma model is slow (~30&ndash;120&nbsp;s
  per render) but functional.
- **WSL2** &mdash; works identically to Linux. This is the
  recommended path for native Windows users who want speed parity.
- **Docker Desktop** &mdash; the bundled demo (`demo/`) runs fine.

If you successfully run native Windows, please open an issue so we
can mark it tested.

---

## Do I need a GPU?

No. The entire library is CPU-only. The headline `fast` model is
nearest-neighbor lookup over a precomputed 384-dim embedding matrix
&mdash; effectively a dot product. Audio synthesis is pure NumPy.

`[expressive]` (local LLM inference) optionally uses MLX on Apple
Silicon for speedup, and `transformers` will use CUDA if it's
available, but neither is required.

---

## How do I get a longer or more genre-specific output?

```python
import latentscore as ls

# Longer: pass duration in seconds
ls.render("warm jazz cafe", duration=60).save("jazz_cafe.wav")

# More control: build a MusicConfig and tweak the knobs
config = ls.MusicConfig(
    tempo="slow", mode="dorian", root="d",
    bass="drone", pad="ambient_drift", melody="contemplative",
    rhythm="minimal", texture="shimmer", echo="heavy",
    density=3, brightness="dark", space="vast",
)
ls.render(config, duration=30).save("custom.wav")

# Combine a vibe with an override
ls.render(
    "morning coffee shop",
    update=ls.MusicConfigUpdate(brightness="very_bright", rhythm="electronic"),
    duration=20,
).play()
```

The full parameter reference is in [`docs/latentscore-dx.md`](latentscore-dx.md).

---

## Doesn't every AI music tool hallucinate? Why not this one?

The default model isn't an LLM. It's a **retrieval system**: your
text gets embedded with MiniLM (or CLAP for `fast_heavy`), then
the nearest neighbor is picked from a hand-curated library of
~10,000&nbsp;`MusicConfig` records. Each record is a deterministic
recipe for a piece of music &mdash; no generation, no
hallucinations, just selection + a procedural synth.

You can opt in to LLM-based generation via `[external]` (Anthropic,
Gemini, OpenAI, etc. through LiteLLM) or `[expressive]` (local
Gemma 3 270M). Those modes can produce richer/more-varied configs
but inherit the usual LLM failure modes (invalid configs, weird
preferences). We default to retrieval because it's the more reliable
shape and fits the "responsive musical sketching" use case better
than free-form generation does.

---

## `latentscore doctor` failed. What now?

`latentscore doctor --json` prints structured output that pinpoints
which of the 10 checks failed and why. Common failures:

| Check | What it means | Fix |
|---|---|---|
| `python_version` | Python &lt; 3.10 or &ge; 3.13 | Use a 3.10–3.12 venv/conda env |
| `license_present` | Editable install metadata is stale | `pip install --force-reinstall latentscore` |
| `audio_write` | Can't write a WAV via `soundfile` | Install `libsndfile` (see "import latentscore fails" above) |
| `render_core` | Synthesis is broken | File a bug with the doctor `--json` output |
| `render_retrieval` (warn) | Falling back to a heuristic mapper because retrieval failed | `latentscore download fast` to seed the model cache |
| `external_available` (fail) | You ran `--require-external` and `litellm` is missing | `pip install "latentscore[external]"` |
| `heavy_available` (fail) | Same for `laion_clap` | `pip install "latentscore[heavy]"` |
| `expressive_available` (fail) | Same for `outlines` (and friends) | `pip install "latentscore[expressive]"` |

`--require-*` flags promote those checks to required so they can
fail strict mode; default behavior treats them as warnings.

---

## Can I cite this in academic work?

Yes. The repo ships a [CITATION.cff](../CITATION.cff) with the
SIGGRAPH Talks '26 paper details. Most bibliography managers
(Zotero, BibTeX-style tooling) read CFF automatically; the
BibTeX block is also in the main README's
[Citation](../README.md#citation) section. License is Apache 2.0
&mdash; see [LICENSE](../LICENSE).
