# Retrieval Map Card

| | |
|---|---|
| **Artifact (default)** | `2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl` |
| **Artifact (heavy)** | `2026-01-26_scored/vibe_and_clap_audio_embeddings_to_config_map.jsonl` |
| **Rows** | 10,558 |
| **Hosted at** | [guprab/latentscore-data](https://huggingface.co/datasets/guprab/latentscore-data) |
| **Embedding dim** | 384 (MiniLM, default) / 512 (LAION-CLAP, heavy) |
| **Splits** | SFT-Train (5,749) / SFT-Val (534) / GRPO (2,672) / TEST (1,603) |
| **License** | Apache 2.0 |

## What this is

A curated index of schema-valid procedural-audio configurations.
Each row maps a natural-language description to a MusicConfig,
title, color palettes, and a text embedding. At runtime, the user's
prompt is embedded and the nearest neighbor is returned.

A second variant ships alongside it with 512-dim LAION-CLAP
audio embeddings instead of MiniLM text embeddings, powering
the optional `fast_heavy` model (`pip install latentscore[heavy]`).

## Row fields

| Field | Type | Notes |
|---|---|---|
| `vibe_original` | string | Natural-language scene description |
| `embedding` | float[384] | MiniLM unit-normalized text embedding |
| `config` | object | Schema-valid MusicConfig (34 fields) |
| `title` | string | Short evocative name |
| `palettes` | list | Three 5-color weighted palettes |
| `split` | string | SFT-Train / SFT-Val / GRPO / TEST |
| `clap_score` | float | CLAP similarity from Best-of-5 selection |

## Construction

Full pipeline in `data_work/`. Four stages:

1. ~3,000 base texts downloaded from Common Pile; ~160
   processed for vibe extraction.
2. A hosted LLM extracts ~14,000 scene descriptions from
   those texts, deduplicated at cosine similarity 0.95
   to ~10,500.
3. Gemini 3 Flash Preview generates N=5 candidate configs
   per vibe. Each is rendered to audio and scored with
   LAION-CLAP. The highest scorer is selected.
4. Each vibe is embedded with all-MiniLM-L6-v2.

## Known limitations

- **English / Western skew.** Source texts are from Common
  Pile collections that lean English-language and Western.
- **CLAP-in-the-loop.** CLAP is used for Best-of-5 selection
  and in the evaluation benchmark. Treat CLAP scores as
  engineering signals, not perceptual claims.
- **Procedural-audio bound.** Configs are constrained to what
  the LatentScore synthesizer can render.
- **Static.** Versioned by date; not continuously updated.
