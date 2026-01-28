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
**Model:** `openrouter/openai/gpt-oss-120b` (cheap model with better quality than 20b variant)

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
  --output-dir data_work/2026-01-26_vibes \
  --model openrouter/openai/gpt-oss-120b \
  --api-key-env OPENROUTER_API_KEY \
  --env-file .env \
  --seed 42 \
  --error-rate 0.15 \
  --max-input-tokens 80000 \
  --max-concurrency 16 \
  --max-qps 10 \
  --dedupe-threshold 0.95 \
  --limit 160 \
  --max-vibes 14000
```

**Note:** `--limit 160` processes only 160 of the 3000 texts. `--max-vibes 14000` ensures early termination once enough vibes are extracted, avoiding hanging API calls.

**Status:** [x] Complete (2026-01-26)

**Results:**
- Pre-dedupe vibes: 14,035
- Post-dedupe vibes: 10,558 (24.8% removed)
- Split distribution:
  | Split | Rows |
  |-------|------|
  | SFT-Train | 5,749 |
  | SFT-Val | 534 |
  | GRPO | 2,672 |
  | TEST | 1,603 |
  | **Total** | **10,558** |

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
**Model:** `gemini/gemini-3-flash-preview` (performs as well as Opus at ~200x lower cost; see model comparison 2026-01-26)

**Why separate scripts:**
- Can re-run config generation without re-extracting vibes
- Different models for different tasks (cheap for vibes, SOTA for configs)
- Incremental writes prevent data loss on crash

**Why SOTA model for configs:**
- Config generation requires understanding musical semantics
- Must produce valid JSON matching strict schema (3 palettes × 5 colors)
- Cheap models fail validation frequently (palette errors, missing fields)

**Best-of-N sampling (two-stage selection):**
- Generate N=5 config candidates per vibe
- Temperature: 0.8 for diversity
- Score each candidate for validity:
  - `format_valid`: Parses as JSON
  - `schema_valid`: Validates against MusicConfigPromptPayload
  - `palette_valid`: Exactly 3 palettes with 5 colors each
- **Stage 1 (02b):** Validation-only selection - picks first schema-valid candidate (tentative)
- **Stage 2 (02c):** Quality-based selection - scores ALL valid candidates with CLAP,
  re-selects `best_index` based on highest CLAP score
- Store all N candidates + validation scores + CLAP scores for analysis

**Prompt contract + caching (2026-01-18):**
- System prompt contains all instructions + schema (see `common/prompts.py`)
- User message contains only `<vibe>...</vibe>`; batch calls use `<vibe_input index=...>`
- LiteLLM prompt caching is enabled on the system message by default
- Optional batching: `--batch-size` + `--batch-wait-ms` (batch suffix stays in system prompt)
- Disable caching with `--no-prompt-caching` in `02b_generate_configs` if needed

**Inference note (2026-01-18):**
- MLX-based local inference uses the same system/user role contract and logs a warning
  because MLX is a temporary inference path.

**Command:**
```bash
python -m data_work.02b_generate_configs \
  --input-dir data_work/2026-01-26_vibes \
  --output-dir data_work/2026-01-26_processed \
  --model gemini/gemini-3-flash-preview \
  --api-key-env GEMINI_API_KEY \
  --env-file .env \
  --num-candidates 5 \
  --temperature 0.8 \
  --seed 42 \
  --max-concurrency 500 \
  --max-qps 300.0
```

**Status:** [x] Complete (2026-01-26)

**Output schema:**
```json
{
  "...all vibe fields from 02a...",
  "config_model": "gemini/gemini-3-flash-preview",
  "config_candidates": [/* N configs (thinking, title, config, palettes) */],
  "validation_scores": {
    "format_valid": [1, 1, 1, 0, 1],
    "schema_valid": [1, 1, 0, 0, 1],
    "palette_valid": [1, 1, 0, 0, 1]
  },
  "best_index": 0,
  "config_payload": {/* best config (thinking, title, config, palettes) */},
  "config_error": null
}
```

**Stats (2026-01-26 run):**
- Total vibe rows: 10,558
- Rows per split: SFT-Train=5,749, SFT-Val=534, GRPO=2,672, TEST=1,603
- Schema validity rate: ~99% (N=5 candidates ensures at least one valid)
- Best-of-5 improvement: +28% (vs single generation baseline)

---

### 1.7 External Scoring and Quality-Based Selection

**Script:** `02c_score_configs.py`

**Purpose:** Perform quality-based Best-of-N selection by scoring ALL valid candidates
with CLAP (or other scorers) and selecting the highest-scoring candidate.

**Why quality-based selection:**
- 02b does validation-only selection (first schema-valid candidate) - doesn't consider quality
- 02c scores ALL N candidates and picks the one with highest CLAP score
- This ensures the selected config is not just valid, but the best quality among valid options
- Per-candidate scores are stored in `candidate_scores` for analysis

**Scorers available:**
- `clap`: LAION-CLAP audio-text similarity (requires laion_clap) - **primary scorer for selection**
- `llm_judge`: LLM-as-judge using multimodal models
- Custom: `path/to/script.py:fn_name` (must return `{final_score: float}`)

**ScoreResult protocol:**
All scorers must return results implementing the `ScoreResult` protocol with a mandatory
`final_score` property (higher = better, unbounded). Only relative ordering matters for GRPO/Best-of-N.

**Command:**
```bash
python -m data_work.02c_score_configs \
  --input-dir data_work/2026-01-26_processed \
  --output-dir data_work/2026-01-26_scored \
  --scorers 'clap' \
  --env-file .env
```

**Status:** [x] Complete (2026-01-26)

**CLAP Score Results (2026-01-26 run):**
| Split | Rows | Mean CLAP | Std |
|-------|------|-----------|-----|
| GRPO | 2,672 | 0.181 | 0.081 |
| SFT-Train | 5,749 | 0.204 | 0.083 |
| SFT-Val | 534 | 0.198 | 0.082 |
| TEST | 1,603 | 0.195 | 0.082 |

**Note:** GRPO scores are lower than SFT-Train as expected - diversity sampling selects harder/more diverse vibes which have lower average CLAP alignment.

**Output schema:**
```json
{
  "...all fields from 02b...",
  "best_index": 2,  // RE-SELECTED based on CLAP score (may differ from 02b)
  "config_payload": {/* RE-SELECTED winner */},
  "candidate_scores": {
    "clap": [0.52, 0.61, 0.68, null, 0.55]  // Per-candidate CLAP scores (null = invalid/error)
  },
  "scores_external": {
    "clap": {
      "audio_text_similarity": 0.68,
      "audio_bad_similarity": 0.12,
      "excess_badness": 0.0,
      "penalty": 0.0,
      "raw_score": 0.68,
      "final_score": 0.68
    },
    "llm_judge": {
      "vibe_match": 0.8,
      "audio_quality": 0.7,
      "final_score": 0.75
    }
  }
}
```

Notes:
- `best_index` and `config_payload` are updated to reflect the CLAP-based winner
- `candidate_scores` shows per-candidate CLAP scores for all N candidates
- `scores_external` contains detailed breakdown for the selected winner
- LLM judge `final_score` uses the harmonic mean of `vibe_match` and `audio_quality` (equal weights)

---

## 2. Training

### 2.0 Prerequisites

**Modal Secret for W&B:**

Training runs on Modal and requires a W&B API key for experiment tracking. Create a Modal secret once:

```bash
conda run -n latentscore-data modal secret create wandb WANDB_API_KEY=<your_key>
```

Get your API key from https://wandb.ai/authorize

### 2.1 SFT (Supervised Fine-Tuning)

**Base model:** `unsloth/gemma-3-270m-it`
**Training data:** `data_work/2026-01-26_scored/SFT-Train.jsonl`
**Validation data:** `data_work/2026-01-26_scored/SFT-Val.jsonl`

**Hyperparameters (prod run 2026-01-27):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | |
| Batch size | 16 | Effective batch = 16 (grad_accum=1) |
| Gradient accumulation | 1 | |
| Learning rate | 1e-4 | Conservative for 270M + LoRA |
| LR scheduler | cosine | |
| Weight decay | 0.01 | Regularization |
| LoRA rank | 16 | |
| LoRA alpha | 16 | Matches rank |
| LoRA dropout | 0.0 | Optimized |
| Max seq length | 4096 | Fits long JSON outputs |
| Optimizer | adamw_8bit | Memory efficient |
| Warmup ratio | 0.06 | |

**Debugging:** Add `--debug-prompts` (optionally `--debug-prompts-max`) to print a few
formatted system/user prompt samples for verification.

**Command (prod run 2026-01-27):**
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/2026-01-26_scored/SFT-Train.jsonl \
  --val-data data_work/2026-01-26_scored/SFT-Val.jsonl \
  --output prod-sft-gemma3-270m-v4 \
  --overwrite \
  --base-model gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr 1e-4 \
  --lr-scheduler cosine \
  --warmup-ratio 0.06 \
  --weight-decay 0.01 \
  --lora-r 16 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --lora-bias none \
  --max-seq-length 4096 \
  --optim adamw_8bit \
  --seed 42 \
  --download-dir data_work/.modal_outputs
```

**Status:** [x] Complete (2026-01-27)
**Artifacts:**
- Modal output: `/outputs/prod-sft-gemma3-270m-v4`
- Local download: `data_work/.modal_outputs/prod-sft-gemma3-270m-v4`

**Training metrics:**
- Final train loss: ___
- Final val loss: ___
- Training time: ___

---

### 2.2 GRPO (Group Relative Policy Optimization)

**Base model:** `unsloth/gemma-3-270m-it`
**Init adapter:** SFT output from 2.1
**Training data:** `data_work/2026-01-26_scored/GRPO.jsonl` (prompts only)

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

**Status:** [ ] Skipped for now (compute constraints)

---

### 2.3 Modal SFT inference (batched, constrained)

**Adapter:** `prod-sft-gemma3-270m-v5`  
**Input:** `data_work/2026-01-26_scored/SFT-Val.jsonl`  
**Output:** `data_work/.modal_outputs/sftval-100-v5-infer-batch` (JSONL file)

**Command (2026-01-27):**
```bash
conda run -n latentscore-data python -m data_work.07_modal_infer_eval \
  --adapter prod-sft-gemma3-270m-v5 \
  --input data_work/2026-01-26_scored/SFT-Val.jsonl \
  --limit 100 \
  --prompt-field vibe_noisy \
  --score-vibe-field vibe_original \
  --do-sample \
  --max-retries 3 \
  --batch-size 16 \
  --log-first-n 10 \
  --log-every 10 \
  --output sftval-100-v5-infer-batch
```

**Notes:**
- Uses Outlines structured decoding via `outlines.from_transformers(...).batch(...)`.
- Batched inference enabled; progress logging is in-place.
- Downloaded output is a JSONL file (not a directory).

---

### 2.4 Audio render (30s) from inference results

**Script:** `data_work/09_render_audio_from_results.py` (new)

**Command (2026-01-27):**
```bash
conda run -n latentscore-data python -m data_work.09_render_audio_from_results \
  --input data_work/.modal_outputs/sftval-100-v5-infer-batch \
  --output-dir data_work/.audio/sftval-100-v5-30s \
  --limit 100 \
  --duration 30
```

**Artifacts:**
- `data_work/.audio/sftval-100-v5-30s` (100 WAV files, 30s each)

---

### 2.5 Embedding map export

**Script:** `data_work/10_export_embedding_map.py` (new)

**Input:** `data_work/2026-01-26_scored/_progress.jsonl`  
**Output:** `data_work/2026-01-26_scored/_progress_embeddings.jsonl`

**Command (2026-01-28):**
```bash
conda run -n latentscore-data python -m data_work.10_export_embedding_map \
  --input data_work/2026-01-26_scored/_progress.jsonl \
  --output data_work/2026-01-26_scored/_progress_embeddings.jsonl \
  --batch-size 64
```

**Notes:**
- Embeds `vibe_original` using `sentence-transformers/all-MiniLM-L6-v2`.
- Output rows include: `vibe_original`, `embedding`, `title`, `config`, `palettes`,
  plus dataset metadata (`dataset`, `id_in_dataset`, `split`).

**Reward weights:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Format | 0.2 | Valid JSON structure |
| Schema | 0.3 | Matches MusicConfig schema |
| Audio | 0.5 | CLAP similarity score |

**Command:**
```bash
AUDIO_REWARD_DURATION=30 \
python -m data_work.03_modal_train --advanced grpo \
  --data data_work/2026-01-26_scored/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir prod-sft-gemma3-270m-hq \
  --output prod-grpo-gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.06 \
  --weight-decay 0.01 \
  --lora-r 16 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --lora-bias none \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --num-generations 4 \
  --beta 0.04 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --audio-reward data_work.audio_rewards:score_clap \
  --download-dir data_work/.modal_outputs \
  --seed 42 \
  --wandb-project latentscore-alpha \
  --wandb-entity rjeinc
```

**Status:** [ ] Skipped for now (compute constraints)

**Training metrics:**
- Final reward mean: ___
- Final reward std: ___
- Training time: ___

---

## 3. Evaluation

### 3.1 CLAP Benchmark

**Eval set:** `data_work/2026-01-26_scored/TEST.jsonl`

**Command:**
```bash
python -m data_work.04_clap_benchmark \
  --input data_work/2026-01-26_scored/TEST.jsonl \
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
| 2026-01-18 | 01 | `01_download_base_data --sample-size 1000` | `data_work/2026-01-18_outputs` | 3,000 texts from Common Pile |
| 2026-01-26 | 02a | `02a_extract_vibes --limit 160 --max-vibes 14000` | `data_work/2026-01-26_vibes` | 160 texts → 14,035 vibes → 10,558 post-dedupe |
| 2026-01-26 | 02b | `02b_generate_configs --num-candidates 5 --max-concurrency 500` | `data_work/2026-01-26_processed` | Gemini Flash, N=5, ~$230 cost |
| 2026-01-26 | 02c | `02c_score_configs --scorers clap` | `data_work/2026-01-26_scored` | ~9 hours, Best-of-5 +28% improvement |

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
| gpt-oss-120b for vibe extraction | Better quality vibes than smaller variants while still being cheap. | 2026-01-26 |
| Gemini 3 Flash for config generation | Model comparison (2026-01-26) showed Flash performs as well as Opus (0.102 vs 0.090 best-of-N avg) at significantly lower cost. | 2026-01-26 |
| Best-of-N config generation (N=5) | Multiple candidates increase probability of valid config. Store all N + scores for analysis. Temperature 0.8 for diversity. | 2026-01-18 |
| ScoreResult protocol for all scorers | Enforces mandatory `final_score` property (higher=better, unbounded) across all scoring systems (CLAP, LLM, rewards, custom). Only relative ordering matters for GRPO/Best-of-N. | 2026-01-18 |
| Separate scoring script (02c) | Decouples scoring from config generation. Can re-run scoring with different methods. Supports multiple scorers in one pass. | 2026-01-18 |
| Config payload uses `thinking` and `title` | Aligns prompt outputs with new naming. | 2026-01-19 |
| Quality-based Best-of-N selection in 02c | 02b does validation-only selection (first valid). 02c scores ALL valid candidates with CLAP and re-selects based on highest quality score. This ensures selected config is not just valid, but best quality among valid options. Per-candidate scores stored in `candidate_scores` for analysis. | 2026-01-25 |
