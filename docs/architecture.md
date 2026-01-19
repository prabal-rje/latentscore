# Architecture

## Data Work Architecture Map

The `data_work/` folder is a standalone data pipeline outside the core package.

**Pipeline flow**
- `data_work/01_download_base_data.py` → download + sample raw text → `data_work/.outputs/`
- `data_work/02a_extract_vibes.py` → vibe extraction + dedupe → `data_work/.vibes/`
- `data_work/02b_generate_configs.py` → config generation (Best-of-N) → `data_work/.processed/`
- `data_work/03_modal_train.py` → Modal SFT/GRPO training → `data_work/.modal_outputs/`
- `data_work/04_clap_benchmark.py` → CLAP scoring + model comparisons
- `data_work/05_export_models.py` → merge LoRA adapters → `data_work/.exports/`

**Supporting modules**
- `data_work/lib/config_io.py` / `jsonl_io.py` handle config + JSONL IO
- `data_work/lib/llm_client.py` / `llm_cache.py` wrap LiteLLM requests + caching
- `data_work/lib/record_builder.py` assembles dataset rows per vibe level
- `data_work/lib/vibe_schema.py` / `music_schema.py` define data contracts
- `data_work/lib/dedupe.py`, `rewards.py`, `resilience.py`, `music_prompt.py` provide
  embedding dedupe, reward helpers, retry logic, and prompt helpers

**Environment**
- `data_work/environment.yml` and `data_work/requirements.txt` define the data-only deps
- `data_work/setup_python_env.sh` provisions a local venv
- Prefer the `latentscore-data` environment for `data_work/` scripts and tooling

## Schema Conversion Flow

The pipeline uses two schema formats for music configs that require conversion:

```
MusicConfigPromptPayload (LLM output)
├── thinking: str
├── config: MusicConfigPrompt  ← string labels ("sparse", "light", "medium")
└── palettes: list[Palette]

         ↓ .config.to_config()

MusicConfig (runtime)           ← numeric values (0.3, 0.2, 0.5)
```

**Why two schemas?**
- `MusicConfigPrompt` uses human-readable string labels that LLMs understand
- `MusicConfig` uses numeric values that the synthesizer needs

**Conversion happens in:**
- `latentscore/config.py`: `MusicConfigPrompt.to_config()` converts labels to floats
- `data_work/lib/clap_scorer.py`: `score_config()` uses this conversion for CLAP scoring
- `data_work/04_clap_benchmark.py`: `_config_to_audio()` uses this for audio generation

**Label mappings (defined in `latentscore/config.py`):**
- `melody_density`: "very_sparse" → 0.15, "sparse" → 0.30, "medium" → 0.50, etc.
- `syncopation`: "straight" → 0.0, "light" → 0.2, "medium" → 0.5, "heavy" → 0.8
- `swing`: "none" → 0.0, "light" → 0.2, "medium" → 0.5, "heavy" → 0.8
- Similar mappings for: `motif_repeat_prob`, `step_bias`, `chromatic_prob`, `cadence_strength`, `chord_change_bars`
