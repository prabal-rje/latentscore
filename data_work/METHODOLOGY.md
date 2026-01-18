# Methodology

This document tracks the end-to-end methodology for creating the production vibe-to-config model.
It serves as a research log and reproducibility guide.

**Target model:** Gemma3-270M-IT (production)

---

## 1. Data Pipeline

### 1.1 Source Data

**Dataset:** Common Pile (subset)

| Field | Description |
|-------|-------------|
| `created` | Timestamp |
| `metadata` | Source-specific metadata |
| `dataset` | Source dataset name |
| `id_in_dataset` | Unique ID within source |
| `text` | Raw text content |

**Command:**
```bash
python -m data_work.01_download_base_data \
  --seed 42 \
  --sample-size 1000 \
  --output-dir data_work/2026-01-18_outputs
```

**Status:** [x] Complete (2026-01-18)

**Results:**
| Dataset | Rows |
|---------|------|
| common-pile/news_filtered | 1,000 |
| common-pile/pressbooks_filtered | 1,000 |
| common-pile/project_gutenberg_filtered | 1,000 |
| **Total** | **3,000** |

---

### 1.2 Vibe Extraction

**Script:** `02a_extract_vibes.py`
**Model:** `openrouter/openai/gpt-oss-20b` (cheap model - vibe extraction is straightforward)

**Input truncation:**
- Max input tokens: 80,000 (whitespace-tokenized)
- Rationale: OpenRouter context limit is 131k; 80k leaves headroom for prompt/schema overhead
- Truncation preferred over exclusion to avoid selection bias toward shorter texts

**Process:**
1. Truncate text to 80k tokens (if longer)
2. Paginate text into chunks (~1000 tokens per page)
3. Extract structured vibe objects (scene vibes, character vibes, tags)
4. Build vibe rows (one row per vibe level/scope/character)
5. Inject noise into ~15% of vibes (for robustness training)

**Command:**
```bash
python -m data_work.02a_extract_vibes \
  --input-dir data_work/2026-01-18_outputs \
  --output-dir data_work/2026-01-18_vibes \
  --model openrouter/openai/gpt-oss-20b \
  --api-key-env OPENROUTER_API_KEY \
  --seed 42 \
  --error-rate 0.15 \
  --max-input-tokens 80000 \
  --max-concurrency 16 \
  --max-qps 10 \
  --dedupe-threshold 0.95
```

**Status:** [ ] Not started

---

### 1.3 Deduplication (on Vibes)

**Why dedupe VIBES (not raw text):**
- Two different books about "love" should be deduped if their vibes are similar
- Dedupe on vibe content is more semantically meaningful
- Prevents redundant training examples that teach the same thing

**Method:** Embedding-based cosine similarity on `vibe_original` field
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Threshold: 0.95 (vibes with similarity ≥ 0.95 are considered duplicates)
- Greedy removal: iterate through vibes, remove any too similar to already-kept vibes
- Happens AFTER vibe extraction, BEFORE splitting

**Integrated into:** `02a_extract_vibes.py`

---

### 1.4 Data Splits

**Ratios:**
| Split | Ratio | Purpose |
|-------|-------|---------|
| SFT-Train | 55% | Supervised fine-tuning |
| SFT-Val | 5% | SFT validation / early stopping |
| GRPO | 25% | Reinforcement learning (prompts only) |
| TEST | 15% | Final held-out evaluation |

**Split ordering (matters for scientific validity):**

```
1. TEST  (15%) ← random from full pool      → representative evaluation
2. VAL   (5%)  ← random from remaining      → must predict TEST performance
3. GRPO  (25%) ← diversity from remaining   → max learning per example
4. TRAIN (55%) ← leftovers                  → fine for SFT
```

**Why this order:**
- Evaluation sets (TEST, VAL) must be random samples of the true distribution
- They're sampled BEFORE any optimization-driven selection
- If GRPO took diverse examples first, VAL would be skewed toward the "middle"
- GRPO then gets diversity sampling from what remains
- TRAIN gets the rest (fine for supervised learning)

---

### 1.5 Diversity Sampling for GRPO

**Why diversity sampling for GRPO:**
- GRPO learns from reward signals, not from (input, target) pairs
- Diverse prompts → diverse learning signal → better generalization
- Redundant prompts teach similar things, wasting gradient updates
- Goal: maximize learning per FLOP

**Method:** Farthest-point sampling (greedy) on vibe embeddings
1. Start with a random seed point
2. Repeatedly select the point farthest from all already-selected points
3. This maximizes the minimum distance between selected points

**Distance metric:** Cosine distance in embedding space (1 - cosine_similarity)

**Why NOT diversity sample SFT/VAL/TEST:**
- TEST must represent the true distribution for honest metrics
- VAL must predict TEST performance → same sampling strategy
- SFT-Train: debatable, but random is the conservative choice

**Integrated into:** `02a_extract_vibes.py` (splitting happens after vibe extraction and dedupe)

---

### 1.6 Config Generation (Best-of-N)

**Script:** `02b_generate_configs.py`
**Model:** `anthropic/claude-opus-4-5-20251101` (SOTA model - config generation needs intelligence)

**Why separate scripts:**
- Can re-run config generation without re-extracting vibes
- Different models for different tasks (cheap for vibes, SOTA for configs)
- Incremental writes prevent data loss on crash

**Why SOTA model for configs:**
- Config generation requires understanding musical semantics
- Must produce valid JSON matching strict schema (3 palettes × 5 colors)
- Cheap models fail validation frequently (palette errors, missing fields)

**Best-of-N sampling:**
- Generate N=5 config candidates per vibe
- Temperature: 0.8 for diversity
- Score each candidate:
  - `format_valid`: Parses as JSON
  - `schema_valid`: Validates against MusicConfigPromptPayload
  - `palette_valid`: Exactly 3 palettes with 5 colors each
- Select best valid config (priority: schema > palette > format)
- Store all N candidates + scores for analysis

**Command:**
```bash
python -m data_work.02b_generate_configs \
  --input-dir data_work/2026-01-18_vibes \
  --output-dir data_work/2026-01-18_processed \
  --model anthropic/claude-opus-4-5-20251101 \
  --api-key-env ANTHROPIC_API_KEY \
  --num-candidates 5 \
  --temperature 0.8 \
  --seed 42 \
  --max-concurrency 4 \
  --max-qps 2
```

**Status:** [ ] Not started

**Output schema:**
```json
{
  "...all vibe fields from 02a...",
  "config_model": "anthropic/claude-opus-4-5-20251101",
  "config_candidates": [/* N configs */],
  "scores": {
    "format_valid": [1, 1, 1, 0, 1],
    "schema_valid": [1, 1, 0, 0, 1],
    "palette_valid": [1, 1, 0, 0, 1]
  },
  "best_index": 0,
  "config_payload": {/* best config */},
  "config_error": null
}
```

**Stats:**
- Total vibe rows: ___
- Rows per split: SFT-Train=___, SFT-Val=___, GRPO=___, TEST=___
- Schema validity rate: ___%
- Best-of-N improvement: ___% (vs single generation)

---

### 1.7 External Scoring (Optional)

**Script:** `02c_score_configs.py`

**Purpose:** Add external quality scores to configs (e.g., CLAP, LLM-as-judge) for analysis
or alternative Best-of-N selection.

**Scorers available:**
- `clap`: LAION-CLAP audio-text similarity (requires laion_clap)
- `llm_judge`: LLM-as-judge using multimodal models
- Custom: `path/to/script.py:fn_name` (must return `{final_score: float}`)

**ScoreResult protocol:**
All scorers must return results implementing the `ScoreResult` protocol with a mandatory
`final_score` property in [0.0, 1.0]. This is enforced at runtime.

**Command:**
```bash
python -m data_work.02c_score_configs \
  --input-dir data_work/2026-01-18_processed \
  --output-dir data_work/2026-01-18_scored \
  --scorers 'clap,llm_judge' \
  --env-file .env
```

**Status:** [ ] Not started

**Output schema:**
```json
{
  "...all fields from 02b...",
  "scores_external": {
    "clap": {
      "audio_text_similarity": 0.65,
      "audio_bad_similarity": 0.12,
      "excess_badness": 0.0,
      "penalty": 0.0,
      "raw_score": 0.65,
      "final_score": 0.58
    },
    "llm_judge": {
      "vibe_match": 0.8,
      "audio_quality": 0.7,
      "creativity": 0.6,
      "final_score": 0.74
    }
  }
}
```

---

## 2. Training

### 2.1 SFT (Supervised Fine-Tuning)

**Base model:** `unsloth/gemma-3-270m-it`
**Training data:** `data_work/2026-01-18_processed/SFT-Train.jsonl`
**Validation data:** `data_work/2026-01-18_processed/SFT-Val.jsonl`

**Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | |
| Batch size | 16 | |
| Learning rate | 2e-4 | |
| LoRA rank | 16 | |
| LoRA alpha | 16 | |
| Max seq length | 2048 | |
| Warmup ratio | 0.06 | |

**Command:**
```bash
python -m data_work.03_modal_train sft \
  --data data_work/2026-01-18_processed/SFT-Train.jsonl \
  --output prod-sft-gemma3-270m \
  --base-model gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --max-seq-length 2048 \
  --download-dir data_work/.modal_outputs
```

**Status:** [ ] Not started

**Training metrics:**
- Final train loss: ___
- Final val loss: ___
- Training time: ___

---

### 2.2 GRPO (Group Relative Policy Optimization)

**Base model:** `unsloth/gemma-3-270m-it`
**Init adapter:** SFT output from 2.1
**Training data:** `data_work/2026-01-18_processed/GRPO.jsonl` (prompts only)

**Why GRPO after SFT:**
- SFT teaches the format (vibe → valid config)
- GRPO optimizes for quality (reward signal from config evaluation)
- Starting from SFT adapter gives GRPO a better starting point

**Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | |
| Batch size | 16 | |
| Learning rate | 5e-5 | Lower than SFT |
| Beta | 0.04 | KL penalty coefficient |
| Num generations | 4 | Samples per prompt |
| Temperature | 0.8 | |
| Top-p | 0.95 | |
| Top-k | 64 | |
| Max completion length | 2048 | |

**Reward weights:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Format | 0.2 | Valid JSON structure |
| Schema | 0.3 | Matches MusicConfig schema |
| Audio | 0.5 | CLAP similarity score |

**Command:**
```bash
python -m data_work.03_modal_train grpo \
  --data data_work/2026-01-18_processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir prod-sft-gemma3-270m \
  --output prod-grpo-gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --max-seq-length 2048 \
  --max-completion-length 2048 \
  --num-generations 4 \
  --beta 0.04 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --download-dir data_work/.modal_outputs
```

**Status:** [ ] Not started

**Training metrics:**
- Final reward mean: ___
- Final reward std: ___
- Training time: ___

---

## 3. Evaluation

### 3.1 CLAP Benchmark

**Eval set:** `data_work/2026-01-18_processed/TEST.jsonl`

**Command:**
```bash
python -m data_work.04_clap_benchmark \
  --input data_work/2026-01-18_processed/TEST.jsonl \
  --local-model data_work/.modal_outputs/prod-grpo-gemma3-270m:production \
  --baseline random \
  --baseline rule_based \
  --limit 100
```

**Status:** [ ] Not started

**Results:**
| Source | CLAP Score (mean) | CLAP Score (std) |
|--------|-------------------|------------------|
| Production model | ___ | ___ |
| Random baseline | ___ | ___ |
| Rule-based baseline | ___ | ___ |

---

### 3.2 Eval Suite

**Command:**
```bash
python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --local-model data_work/.modal_outputs/prod-grpo-gemma3-270m \
  --baseline random \
  --baseline rule_based \
  --output-dir data_work/.experiments/eval_results/production
```

**Status:** [ ] Not started

**Results:**
| Metric | Production | Random | Rule-based |
|--------|------------|--------|------------|
| Schema validity | ___ | ___ | ___ |
| CLAP score | ___ | ___ | ___ |
| Format score | ___ | ___ | ___ |

---

## 4. Export & Deployment

### 4.1 Model Export

**Command:**
```bash
python -m data_work.05_export_models \
  --adapter-dir data_work/.modal_outputs/prod-grpo-gemma3-270m \
  --output-dir data_work/.exports/prod-gemma3-270m-merged
```

**Status:** [ ] Not started

---

## 5. Run Log

Track actual runs here as they complete.

| Date | Step | Command | Output | Notes |
|------|------|---------|--------|-------|
| | | | | |

---

## 6. Decisions & Rationale

Document key decisions made during the process.

| Decision | Rationale | Date |
|----------|-----------|------|
| Dedupe on vibes (not raw text) | Two different books about "love" should dedupe if vibes are similar. Semantically meaningful deduplication. | 2026-01-18 |
| Diversity sampling for GRPO only | Eval sets must be representative; GRPO benefits from coverage | 2026-01-18 |
| Split order: TEST → VAL → GRPO → TRAIN | Eval sets sampled before optimization-driven selection | 2026-01-18 |
| 55/5/25/15 split ratios | Independent GRPO data (not reusing SFT prompts) | 2026-01-18 |
| Truncate texts to 80k tokens | Avoids API context limit errors (131k limit minus prompt/schema overhead). Truncation preferred over exclusion to avoid selection bias toward shorter texts. 80k tokens ≈ 60k words, sufficient to establish vibe. | 2026-01-18 |
| Dedupe threshold 0.95 | More aggressive than default 0.97; catches subtle near-duplicates while remaining safe | 2026-01-18 |
| Separate vibe extraction (02a) and config generation (02b) | Allows re-running config generation without re-extracting vibes. Different models for different tasks. Incremental writes prevent data loss. | 2026-01-18 |
| Cheap model (gpt-oss-20b) for vibe extraction | Vibe extraction is straightforward - doesn't need SOTA intelligence. Cost effective. | 2026-01-18 |
| SOTA model (Claude Opus) for config generation | Config generation requires understanding musical semantics. Must produce valid JSON matching strict schema. Cheap models fail validation frequently. | 2026-01-18 |
| Best-of-N config generation (N=5) | Multiple candidates increase probability of valid config. Store all N + scores for analysis. Temperature 0.8 for diversity. | 2026-01-18 |
| ScoreResult protocol for all scorers | Enforces mandatory `final_score` property in [0,1] across all scoring systems (CLAP, LLM, rewards, custom). Runtime validation prevents silent failures. | 2026-01-18 |
| Separate scoring script (02c) | Decouples scoring from config generation. Can re-run scoring with different methods. Supports multiple scorers in one pass. | 2026-01-18 |

