# data_work EXPERIMENTS

This document lists academic experiments with IRL parameters. Use `--limit`/`--limit-per-split`
for smoke tests. All commands use the conda environment `latentscore-data`.

## Shared setup

IRL datasets used here:
- Training (SFT): `data_work/.processed/SFT-Train.jsonl`
- Training (GRPO): `data_work/.processed/GRPO.jsonl`
- Base input for processing: `data_work/.outputs`
- Data scaling: `data_work/.experiments/data_scaling/size_*.jsonl`

Output directories:
- Modal outputs (Modal volume): `/outputs/<experiment>`
- Eval outputs: `data_work/.experiments/eval_results`

Notes:
- Training commands use `--max-seq-length 4096` (~2k tokens).
- System prompts default to prompt registry `config_v1` unless overridden.
- GRPO `--model` should point at the base HF repo (not the SFT adapter).
- GRPO `--init-adapter-dir` should point at the SFT LoRA adapter in `/outputs/<name>`.
- Run notes reflect prior tiny-run validation; re-verify for IRL.
- Run notes are stale after chat-template + adapter-init alignment; re-run before relying on them.

## Core ablations

### SFT vs SFT + GRPO comparison

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-baseline \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --base-model gemma3-270m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-baseline \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --num-generations 4 \
  --beta 0.04 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --overwrite

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/sft_vs_grpo
```

Note: Gemma3 sampling follows Unsloth model guidance.

Run notes (2026-01-17):
- SFT command: OK (Modal output: `/outputs/exp-sft-baseline`)
- GRPO command: OK (Modal output: `/outputs/exp-grpo-baseline`)
- Eval command: OK (`data_work/.experiments/eval_results/sft_vs_grpo/short_prompts/random`)

---

### Model-specific baseline runs (Gemma3, Qwen3)

Use the base-model sweep outputs if available; otherwise run the SFT commands below first.

Commands:
```bash
# Gemma3-270M-IT (Unsloth sampling guidance)
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-base-gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --base-model gemma3-270m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-base-gemma3-270m \
  --output exp-grpo-base-gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --overwrite

# Qwen3-0.6B (disable thinking mode via chat template)
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-base-qwen3-600m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --base-model qwen3-600m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/Qwen3-0.6B \
  --init-adapter-dir exp-sft-base-qwen3-600m \
  --output exp-grpo-base-qwen3-600m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --overwrite
```

Notes:
- Gemma3 sampling follows Unsloth model docs.
- Qwen3 sampling follows HF card non-thinking recommendations.

---

### Reward weights sweep (format, schema, audio)

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-weights-0_1-0_4-0_5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --format-weight 0.1 \
  --schema-weight 0.4 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-weights-0_2-0_3-0_5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --format-weight 0.2 \
  --schema-weight 0.3 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-weights-0_3-0_2-0_5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --format-weight 0.3 \
  --schema-weight 0.2 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-weights-0_4-0_3-0_3 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --format-weight 0.4 \
  --schema-weight 0.3 \
  --audio-weight 0.3 \
  --overwrite
```

Run notes (2026-01-17):
- weights 0.1/0.4/0.5: OK (`/outputs/exp-grpo-weights-0_1-0_4-0_5`)
- weights 0.2/0.3/0.5: OK (`/outputs/exp-grpo-weights-0_2-0_3-0_5`)
- weights 0.3/0.2/0.5: OK (`/outputs/exp-grpo-weights-0_3-0_2-0_5`)
- weights 0.4/0.3/0.3: OK (`/outputs/exp-grpo-weights-0_4-0_3-0_3`)

---

### Vibe representation ablation (raw text, scene vs character, level subsets)

Commands:
```bash
# base extraction for ablation (run once; reused by filters below)
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/vibe_base \
  --overwrite

# raw_text_direct (no vibe extraction; reuse configs, replace vibe text with raw input text)
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/vibe_raw_text \
  --raw-text-direct \
  --overwrite

# scene_only
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_scene_only \
  --filter-vibe-scope scene \
  --overwrite

# character_only
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_character_only \
  --filter-vibe-scope character \
  --overwrite

# both_scopes
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_both_scopes \
  --overwrite

# xl_only
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_xl_only \
  --filter-vibe-level xl \
  --overwrite

# xs_only
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_xs_only \
  --filter-vibe-level xs \
  --overwrite

# all_levels
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --source-dir data_work/.experiments/vibe_base \
  --output-dir data_work/.experiments/vibe_all_levels \
  --overwrite
```

<!--
Legacy Python fallback (kept for reuse):

```bash
# raw_text_direct (no vibe extraction; reuse configs, replace vibe text with raw input text)
python3 - <<'PY'
import json
from pathlib import Path
import shutil

base_input = Path("data_work/.outputs/base.jsonl")
src = Path("data_work/.experiments/vibe_base/SFT-Train.jsonl")
dst_dir = Path("data_work/.experiments/vibe_raw_text")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(Path("data_work/.experiments/vibe_base/run_config.json"), dst_dir / "run_config.json")

text_map = {}
with base_input.open() as handle:
    for line in handle:
        row = json.loads(line)
        text_map[(row["dataset"], str(row["id_in_dataset"]))] = row["text"]

with src.open() as handle, (dst_dir / "SFT-Train.jsonl").open("w") as out:
    for line in handle:
        row = json.loads(line)
        key = (row["dataset"], str(row["id_in_dataset"]))
        text = text_map.get(key, "")
        row["vibe_original"] = text
        row["vibe_noisy"] = text
        row["vibe_scope"] = "raw_text"
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# scene_only
python3 - <<'PY'
import json
from pathlib import Path
import shutil

src_dir = Path("data_work/.experiments/vibe_base")
dst_dir = Path("data_work/.experiments/vibe_scene_only")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(src_dir / "run_config.json", dst_dir / "run_config.json")

with (src_dir / "SFT-Train.jsonl").open() as handle, (dst_dir / "SFT-Train.jsonl").open("w") as out:
    for line in handle:
        row = json.loads(line)
        if row.get("vibe_scope") == "scene":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# character_only
python3 - <<'PY'
import json
from pathlib import Path
import shutil

src_dir = Path("data_work/.experiments/vibe_base")
dst_dir = Path("data_work/.experiments/vibe_character_only")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(src_dir / "run_config.json", dst_dir / "run_config.json")

with (src_dir / "SFT-Train.jsonl").open() as handle, (dst_dir / "SFT-Train.jsonl").open("w") as out:
    for line in handle:
        row = json.loads(line)
        if row.get("vibe_scope") == "character":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# both_scopes
cp -a data_work/.experiments/vibe_base data_work/.experiments/vibe_both_scopes

# xl_only
python3 - <<'PY'
import json
from pathlib import Path
import shutil

src_dir = Path("data_work/.experiments/vibe_base")
dst_dir = Path("data_work/.experiments/vibe_xl_only")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(src_dir / "run_config.json", dst_dir / "run_config.json")

with (src_dir / "SFT-Train.jsonl").open() as handle, (dst_dir / "SFT-Train.jsonl").open("w") as out:
    for line in handle:
        row = json.loads(line)
        if row.get("vibe_level") == "xl":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# xs_only
python3 - <<'PY'
import json
from pathlib import Path
import shutil

src_dir = Path("data_work/.experiments/vibe_base")
dst_dir = Path("data_work/.experiments/vibe_xs_only")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(src_dir / "run_config.json", dst_dir / "run_config.json")

with (src_dir / "SFT-Train.jsonl").open() as handle, (dst_dir / "SFT-Train.jsonl").open("w") as out:
    for line in handle:
        row = json.loads(line)
        if row.get("vibe_level") == "xs":
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# all_levels
cp -a data_work/.experiments/vibe_base data_work/.experiments/vibe_all_levels
```
-->

Run notes (2026-01-17):
- vibe_base: OK (`data_work/.experiments/vibe_base`)
- raw_text_direct: OK (`data_work/.experiments/vibe_raw_text`)
- scene_only: OK (`data_work/.experiments/vibe_scene_only`)
- character_only: OK (`data_work/.experiments/vibe_character_only`)
- both_scopes: OK (`data_work/.experiments/vibe_both_scopes`)
- xl_only: OK (`data_work/.experiments/vibe_xl_only`)
- xs_only: OK (`data_work/.experiments/vibe_xs_only`)
- all_levels: OK (`data_work/.experiments/vibe_all_levels`)

---

### Noise injection sweep (error_rate)

Commands:
```bash
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/noise_0_0 \
  --error-rate 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/noise_0_05 \
  --error-rate 0.05 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/noise_0_15 \
  --error-rate 0.15 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/noise_0_30 \
  --error-rate 0.30 \
  --overwrite
```

Run notes (2026-01-17):
- error_rate 0.0: OK (`data_work/.experiments/noise_0_0`)
- error_rate 0.05: OK (`data_work/.experiments/noise_0_05`)
- error_rate 0.15: OK (`data_work/.experiments/noise_0_15`)
- error_rate 0.30: OK (`data_work/.experiments/noise_0_30`)

---

### Human eval + metric correlation

Commands:
```bash
conda run -n latentscore-data python -m data_work.07_human_eval_pack generate \
  --model-a exp-sft-baseline \
  --model-b exp-grpo-baseline \
  --eval-set short_prompts \
  --n-samples 100 \
  --output-dir data_work/.experiments/human_eval/sft_vs_grpo

conda run -n latentscore-data python -m data_work.07_human_eval_pack analyze \
  --input-dir data_work/.experiments/human_eval/sft_vs_grpo
```

Run notes (2026-01-17):
- generate: NOT IMPLEMENTED (`data_work.07_human_eval_pack` stub)
- analyze: NOT IMPLEMENTED (`data_work.07_human_eval_pack` stub)

---

## Training hyperparameter ablations

### GRPO beta sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-beta-0_01 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --beta 0.01 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-beta-0_02 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --beta 0.02 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-beta-0_04 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --beta 0.04 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-beta-0_08 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --beta 0.08 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-beta-0_16 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --beta 0.16 \
  --overwrite
```

Run notes (2026-01-17):
- beta 0.01: OK (`/outputs/exp-grpo-beta-0_01`)
- beta 0.02: OK (`/outputs/exp-grpo-beta-0_02`)
- beta 0.04: OK (`/outputs/exp-grpo-beta-0_04`)
- beta 0.08: OK (`/outputs/exp-grpo-beta-0_08`)
- beta 0.16: OK (`/outputs/exp-grpo-beta-0_16`)

---

### LoRA rank sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-r4 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-r 4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-r8 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-r 8 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-r16 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-r 16 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-r32 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-r 32 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-r64 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-r 64 \
  --overwrite
```

Run notes (2026-01-17):
- r=4: OK (`/outputs/exp-sft-lora-r4`)
- r=8: OK (`/outputs/exp-sft-lora-r8`)
- r=16: OK (`/outputs/exp-sft-lora-r16`)
- r=32: OK (`/outputs/exp-sft-lora-r32`)
- r=64: OK (`/outputs/exp-sft-lora-r64`)

---

### Learning rate sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lr-1e-5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lr 1e-5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lr-5e-5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lr 5e-5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lr-1e-4 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lr 1e-4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lr-2e-4 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lr 2e-4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lr-5e-4 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lr 5e-4 \
  --overwrite
```

Run notes (2026-01-17):
- lr=1e-5: OK (`/outputs/exp-sft-lr-1e-5`)
- lr=5e-5: OK (`/outputs/exp-sft-lr-5e-5`)
- lr=1e-4: OK (`/outputs/exp-sft-lr-1e-4`)
- lr=2e-4: OK (`/outputs/exp-sft-lr-2e-4`)
- lr=5e-4: OK (`/outputs/exp-sft-lr-5e-4`)

---

### Batch size sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-bs-4 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-bs-8 \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-bs-16 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-bs-32 \
  --epochs 3 \
  --batch-size 32 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite
```

Run notes (2026-01-17):
- batch_size=4: OK (`/outputs/exp-sft-bs-4`)
- batch_size=8: OK (`/outputs/exp-sft-bs-8`)
- batch_size=16: OK (`/outputs/exp-sft-bs-16`)
- batch_size=32: OK (`/outputs/exp-sft-bs-32`)

---

### GRPO num_generations sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-ngen-2 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --num-generations 2 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-ngen-4 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --num-generations 4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-ngen-6 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --num-generations 6 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-ngen-8 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --num-generations 8 \
  --overwrite
```

Run notes (2026-01-17):
- num_generations=2: OK (`/outputs/exp-grpo-ngen-2`)
- num_generations=4: OK (`/outputs/exp-grpo-ngen-4`)
- num_generations=6: OK (`/outputs/exp-grpo-ngen-6`)
- num_generations=8: OK (`/outputs/exp-grpo-ngen-8`)

---

### Dedupe threshold sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/dedupe_0_90 \
  --dedupe-threshold 0.90 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/dedupe_0_95 \
  --dedupe-threshold 0.95 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/dedupe_0_98 \
  --dedupe-threshold 0.98 \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.outputs \
  --output-dir data_work/.experiments/dedupe_1_0 \
  --dedupe-threshold 1.0 \
  --overwrite
```

Run notes (2026-01-17):
- threshold 0.90: OK (`data_work/.experiments/dedupe_0_90/SFT-Train.jsonl`)
- threshold 0.95: OK (`data_work/.experiments/dedupe_0_95/SFT-Train.jsonl`)
- threshold 0.98: OK (`data_work/.experiments/dedupe_0_98/SFT-Train.jsonl`)
- threshold 1.0: OK (`data_work/.experiments/dedupe_1_0/SFT-Train.jsonl`)

---

### Warmup ratio sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-warmup-0_0 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --warmup-ratio 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-warmup-0_06 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --warmup-ratio 0.06 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-warmup-0_1 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --warmup-ratio 0.1 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-warmup-0_2 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --warmup-ratio 0.2 \
  --overwrite
```

Run notes (2026-01-17):
- warmup_ratio 0.0: OK (`/outputs/exp-sft-warmup-0_0`)
- warmup_ratio 0.06: OK (`/outputs/exp-sft-warmup-0_06`)
- warmup_ratio 0.1: OK (`/outputs/exp-sft-warmup-0_1`)
- warmup_ratio 0.2: OK (`/outputs/exp-sft-warmup-0_2`)

---

### LoRA dropout sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-dropout-0_0 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-dropout 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-dropout-0_05 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-dropout 0.05 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-dropout-0_1 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-dropout 0.1 \
  --overwrite
```

Run notes (2026-01-17):
- lora_dropout 0.0: OK (`/outputs/exp-sft-lora-dropout-0_0`)
- lora_dropout 0.05: OK (`/outputs/exp-sft-lora-dropout-0_05`)
- lora_dropout 0.1: OK (`/outputs/exp-sft-lora-dropout-0_1`)

---

### Weight decay sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-weight-decay-0_0 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --weight-decay 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-weight-decay-0_01 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --weight-decay 0.01 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-weight-decay-0_1 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --weight-decay 0.1 \
  --overwrite
```

Run notes (2026-01-17):
- weight_decay 0.0: OK (`/outputs/exp-sft-weight-decay-0_0`)
- weight_decay 0.01: OK (`/outputs/exp-sft-weight-decay-0_01`)
- weight_decay 0.1: OK (`/outputs/exp-sft-weight-decay-0_1`)

---

### Max sequence length sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-max-seq-256 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 256 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-max-seq-512 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 512 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-max-seq-4096 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite
```

Run notes (2026-01-17):
- max_seq_length 256: FAILED (sequence length too small; Unsloth assertion)
- max_seq_length 512: FAILED (sequence length too small; Unsloth assertion)
- max_seq_length 4096: PENDING (rerun after prompt update)

---

### LoRA alpha sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-alpha-8 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-alpha 8 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-alpha-16 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-alpha 16 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-lora-alpha-32 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --lora-alpha 32 \
  --overwrite
```

Run notes (2026-01-17):
- lora_alpha 8: OK (`/outputs/exp-sft-lora-alpha-8`)
- lora_alpha 16: OK (`/outputs/exp-sft-lora-alpha-16`)
- lora_alpha 32: OK (`/outputs/exp-sft-lora-alpha-32`)

---

### Base model sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-base-gemma3-270m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --base-model gemma3-270m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-base-qwen3-600m \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --base-model qwen3-600m \
  --overwrite
```

Run notes (2026-01-17):
- base_model gemma3-270m: OK (`/outputs/exp-sft-base-gemma3-270m`)
- base_model qwen3-600m: OK (smoke run; output `/outputs/exp-sft-base-qwen3-600m-smoke`)

---

## Data ablations

### Data scaling study

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_1.jsonl \
  --output exp-sft-scale-1 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_5.jsonl \
  --output exp-sft-scale-5 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_20.jsonl \
  --output exp-sft-scale-20 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_100.jsonl \
  --output exp-sft-scale-100 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite
```

Run notes (2026-01-17):
- size 1: OK (`/outputs/exp-sft-scale-1`)
- size 5: OK (`/outputs/exp-sft-scale-5`)
- size 20: OK (`/outputs/exp-sft-scale-20`)
- size 100: OK (`/outputs/exp-sft-scale-100`)

---

## Quality and analysis experiments

### Controllability tests

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/tempo_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/ctrl_tempo

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/mode_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/ctrl_mode

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/brightness_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/ctrl_brightness

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/rhythm_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/ctrl_rhythm

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/texture_prompts \
  --baseline random \
  --output-dir data_work/.experiments/eval_results/ctrl_texture
```

Run notes (2026-01-17):
- tempo: OK (`data_work/.experiments/eval_results/ctrl_tempo`)
- mode: OK (`data_work/.experiments/eval_results/ctrl_mode`)
- brightness: OK (`data_work/.experiments/eval_results/ctrl_brightness`)
- rhythm: OK (`data_work/.experiments/eval_results/ctrl_rhythm/controllability_rhythm_prompts/random`)
- texture: OK (`data_work/.experiments/eval_results/ctrl_texture/controllability_texture_prompts/random`)

---

### Synth sensitivity analysis

Commands:
```bash
conda run -n latentscore-data python -m data_work.08_synth_sensitivity \
  --input data_work/.experiments/mini_configs.jsonl \
  --limit 1 \
  --output-dir data_work/.experiments/synth_sensitivity
```

Run notes (2026-01-17):
- synth sensitivity: OK (`data_work/.experiments/synth_sensitivity/synth_sensitivity.jsonl`, smoke input: `data_work/.experiments/mini_configs.jsonl`)

---

### Field usage analysis

Commands:
```bash
conda run -n latentscore-data python -m data_work.09_field_usage \
  --input data_work/.experiments/mini_configs.jsonl \
  --limit 1 \
  --output-dir data_work/.experiments/field_usage
```

Run notes (2026-01-17):
- field usage: OK (`data_work/.experiments/field_usage/field_usage.jsonl`, smoke input: `data_work/.experiments/mini_configs.jsonl`)

---

## Differentiation experiments

### Teacher dependence study

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model gemini/gemini-3-flash-preview \
  --api-key-env GEMINI_API_KEY \
  --output-dir data_work/.experiments/eval_results/teacher_gemini_flash

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model gemini/gemini-3-pro-preview \
  --api-key-env GEMINI_API_KEY \
  --output-dir data_work/.experiments/eval_results/teacher_gemini_pro

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model anthropic/claude-opus-4-5-20251101 \
  --api-key-env ANTHROPIC_API_KEY \
  --output-dir data_work/.experiments/eval_results/teacher_claude_opus

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline rule_based \
  --output-dir data_work/.experiments/eval_results/teacher_rule_based
```

Run notes (2026-01-17):
- Gemini Flash: OK (100% JSON validity; `--limit 1`; `data_work/.experiments/eval_results/teacher_gemini_flash/short_prompts/gemini/gemini-3-flash-preview`; litellm async cleanup warnings)
- Gemini Pro: OK (100% JSON validity; `--limit 1`; `data_work/.experiments/eval_results/teacher_gemini_pro/short_prompts/gemini/gemini-3-pro-preview`; litellm async cleanup warnings)
- Claude Opus: OK (100% JSON validity; `--limit 1`; `data_work/.experiments/eval_results/teacher_claude_opus/short_prompts/anthropic/claude-opus-4-5-20251101`; litellm async cleanup warnings)
- rule_based: OK (`data_work/.experiments/eval_results/teacher_rule_based/short_prompts/rule_based`)

---

## Scoring method experiments

### CLAP vs LLM scorer correlation

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --include-clap \
  --llm-scorer gemini/gemini-3-flash-preview \
  --api-key-env GEMINI_API_KEY \
  --output-dir data_work/.experiments/eval_results/clap_vs_llm
```

Run notes (2026-01-17):
- clap vs llm: OK (explicit `--api-key` from `.env`; `--limit 1`; output `data_work/.experiments/eval_results/clap_vs_llm/short_prompts/random`)

---

### LLM scorer model comparison

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer gemini/gemini-3-flash-preview \
  --api-key-env GEMINI_API_KEY \
  --output-dir data_work/.experiments/eval_results/llm_gemini_flash

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer gemini/gemini-3-pro-preview \
  --api-key-env GEMINI_API_KEY \
  --output-dir data_work/.experiments/eval_results/llm_gemini_pro

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer anthropic/claude-opus-4-5-20251101 \
  --api-key-env ANTHROPIC_API_KEY \
  --output-dir data_work/.experiments/eval_results/llm_claude_opus
```

Run notes (2026-01-17):
- gemini flash: OK (explicit `--api-key` from `.env`; `--limit 1`; output `data_work/.experiments/eval_results/llm_gemini_flash/short_prompts/random`; litellm async cleanup warnings)
- gemini pro: OK (explicit `--api-key` from `.env`; `--limit 1`; output `data_work/.experiments/eval_results/llm_gemini_pro/short_prompts/random`; litellm async cleanup warnings)
- claude opus: OK (explicit `--api-key` from `.env`; `--limit 1`; output `data_work/.experiments/eval_results/llm_claude_opus/short_prompts/random`; litellm async cleanup warnings)

---

### LLM scorer as GRPO reward

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.processed/GRPO.jsonl \
  --model unsloth/gemma-3-270m-it \
  --init-adapter-dir exp-sft-baseline \
  --output exp-grpo-llm-reward \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --max-completion-length 2048 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --audio-reward data_work.lib.llm_scorer:score_config_with_llm \
  --overwrite
```

Run notes (2026-01-17):
- grpo llm reward: OK (smoke run; output `/outputs/exp-grpo-llm-reward-smoke`)

---

## Prompt ablations

### System prompt variants

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-prompt-default \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.processed/SFT-Train.jsonl \
  --output exp-sft-prompt-detailed \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 4096 \
  --system-prompt "You are an expert sound designer. Return only JSON matching the schema." \
  --overwrite
```

Run notes (2026-01-17):
- default prompt: OK (smoke run; output `/outputs/exp-sft-prompt-default-smoke`)
- detailed prompt: OK (smoke run; output `/outputs/exp-sft-prompt-detailed-smoke`)
