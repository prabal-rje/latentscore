# data_work EXPERIMENTS

This document lists academic experiments, with explicit commands and run notes for trivial-data runs.
All commands use the conda environment `latentscore-data`.

## Shared setup

Minimal datasets used here:
- Training: `data_work/.experiments/mini_train.jsonl`
- Base input for processing: `data_work/.experiments/base_input/base.jsonl`
- Data scaling: `data_work/.experiments/data_scaling/size_*.jsonl`

Output directories:
- Modal outputs (Modal volume): `/outputs/<experiment>`
- Eval outputs: `data_work/.experiments/eval_results`

Notes:
- Training defaults to `--max-seq-length 1024` to avoid Unsloth length assertions.
- GRPO `--model` should point at `/outputs/<name>` (bare names resolve to `/outputs/<name>`).

## Core ablations

### SFT vs SFT + GRPO comparison

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-baseline \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-baseline \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --num-generations 2 \
  --beta 0.04 \
  --overwrite

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/sft_vs_grpo
```

Run notes (2026-01-17):
- SFT command: OK (Modal output: `/outputs/exp-sft-baseline`)
- GRPO command: OK (Modal output: `/outputs/exp-grpo-baseline`)
- Eval command: OK (`data_work/.experiments/eval_results/sft_vs_grpo/short_prompts/random`)

---

### Reward weights sweep (format, schema, audio)

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-weights-0_1-0_4-0_5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --format-weight 0.1 \
  --schema-weight 0.4 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-weights-0_2-0_3-0_5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --format-weight 0.2 \
  --schema-weight 0.3 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-weights-0_3-0_2-0_5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --format-weight 0.3 \
  --schema-weight 0.2 \
  --audio-weight 0.5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-weights-0_4-0_3-0_3 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/vibe_base \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

# raw_text_direct (no vibe extraction; reuse configs, replace vibe text with raw input text)
python3 - <<'PY'
import json
from pathlib import Path
import shutil

base_input = Path("data_work/.experiments/base_input/base.jsonl")
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
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/noise_0_0 \
  --error-rate 0.0 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/noise_0_05 \
  --error-rate 0.05 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/noise_0_15 \
  --error-rate 0.15 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/noise_0_30 \
  --error-rate 0.30 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
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
  --n-samples 1 \
  --output-dir data_work/.experiments/human_eval/sft_vs_grpo

conda run -n latentscore-data python -m data_work.07_human_eval_pack analyze \
  --input-dir data_work/.experiments/human_eval/sft_vs_grpo
```

Run notes (2026-01-17):
- generate: FAILED (missing `data_work.07_human_eval_pack`)
- analyze: FAILED (missing `data_work.07_human_eval_pack`)

---

## Training hyperparameter ablations

### GRPO beta sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-beta-0_01 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --beta 0.01 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-beta-0_02 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --beta 0.02 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-beta-0_04 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --beta 0.04 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-beta-0_08 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --beta 0.08 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-beta-0_16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-r4 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-r 4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-r8 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-r 8 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-r16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-r 16 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-r32 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-r 32 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-r64 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lr-1e-5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lr 1e-5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lr-5e-5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lr 5e-5 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lr-1e-4 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lr 1e-4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lr-2e-4 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lr 2e-4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lr-5e-4 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-bs-4 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-bs-8 \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-bs-16 \
  --epochs 1 \
  --batch-size 16 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-bs-32 \
  --epochs 1 \
  --batch-size 32 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-ngen-2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --num-generations 2 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-ngen-4 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --num-generations 4 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-ngen-6 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --num-generations 6 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-ngen-8 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/dedupe_0_90 \
  --dedupe-threshold 0.90 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/dedupe_0_95 \
  --dedupe-threshold 0.95 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/dedupe_0_98 \
  --dedupe-threshold 0.98 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
  --overwrite

conda run -n latentscore-data python -m data_work.02_process_base_data \
  --input-dir data_work/.experiments/base_input \
  --output-dir data_work/.experiments/dedupe_1_0 \
  --dedupe-threshold 1.0 \
  --limit-per-split 1 \
  --only-splits SFT-Train \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-warmup-0_0 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --warmup-ratio 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-warmup-0_06 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --warmup-ratio 0.06 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-warmup-0_1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --warmup-ratio 0.1 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-warmup-0_2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-dropout-0_0 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-dropout 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-dropout-0_05 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-dropout 0.05 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-dropout-0_1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-weight-decay-0_0 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --weight-decay 0.0 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-weight-decay-0_01 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --weight-decay 0.01 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-weight-decay-0_1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-max-seq-256 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 256 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-max-seq-512 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 512 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-max-seq-1024 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite
```

Run notes (2026-01-17):
- max_seq_length 256: FAILED (sequence length too small; Unsloth assertion)
- max_seq_length 512: FAILED (sequence length too small; Unsloth assertion)
- max_seq_length 1024: OK (`/outputs/exp-sft-max-seq-1024`)

---

### LoRA alpha sweep

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-alpha-8 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-alpha 8 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-alpha-16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --lora-alpha 16 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-lora-alpha-32 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-base-smollm2-135m \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --base-model smollm2-135m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-base-gemma3-270m \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --base-model gemma3-270m \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-base-qwen3-600m \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --base-model qwen3-600m \
  --overwrite
```

Run notes (2026-01-17):
- base_model smollm2-135m: OK (`/outputs/exp-sft-base-smollm2-135m`)
- base_model gemma3-270m: OK (`/outputs/exp-sft-base-gemma3-270m`)
- base_model qwen3-600m: FAILED (HF repo missing: `unsloth/Qwen3-600M`)

---

## Data ablations

### Data scaling study

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_1.jsonl \
  --output exp-sft-scale-1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_5.jsonl \
  --output exp-sft-scale-5 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_20.jsonl \
  --output exp-sft-scale-20 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/data_scaling/size_100.jsonl \
  --output exp-sft-scale-100 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
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
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/ctrl_tempo

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/mode_prompts \
  --baseline random \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/ctrl_mode

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/brightness_prompts \
  --baseline random \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/ctrl_brightness

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/rhythm_prompts \
  --baseline random \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/ctrl_rhythm

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set controllability/texture_prompts \
  --baseline random \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/ctrl_texture
```

Run notes (2026-01-17):
- tempo: OK (`data_work/.experiments/eval_results/ctrl_tempo`)
- mode: OK (`data_work/.experiments/eval_results/ctrl_mode`)
- brightness: OK (`data_work/.experiments/eval_results/ctrl_brightness`)
- rhythm: FAILED (missing `data_work/eval_sets/controllability/rhythm_prompts.jsonl`)
- texture: FAILED (missing `data_work/eval_sets/controllability/texture_prompts.jsonl`)

---

### Synth sensitivity analysis

Commands:
```bash
conda run -n latentscore-data python -m data_work.09_synth_sensitivity \
  --input data_work/.experiments/mini_train.jsonl \
  --limit 1 \
  --output-dir data_work/.experiments/synth_sensitivity
```

Run notes (2026-01-17):
- synth sensitivity: FAILED (missing `data_work.09_synth_sensitivity`)

---

### Field usage analysis

Commands:
```bash
conda run -n latentscore-data python -m data_work.10_field_usage \
  --input data_work/.experiments/mini_train.jsonl \
  --limit 1 \
  --output-dir data_work/.experiments/field_usage
```

Run notes (2026-01-17):
- field usage: PENDING

---

## Differentiation experiments

### Story coherence / multi-page continuity

Commands:
```bash
conda run -n latentscore-data python -m data_work.11_story_coherence \
  --input data_work/.experiments/mini_train.jsonl \
  --limit 1 \
  --output-dir data_work/.experiments/story_coherence
```

Run notes (2026-01-17):
- story coherence: PENDING

---

### Teacher dependence study

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model openai/gpt-4o-mini:gpt4 \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/teacher_gpt4

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model openai/gpt-3.5-turbo:gpt35 \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/teacher_gpt35

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --litellm-model mistral/mistral-7b-instruct:mixtral7b \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/teacher_mistral

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline rule_based \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/teacher_rule_based
```

Run notes (2026-01-17):
- GPT-4: OK (0% JSON validity; `data_work/.experiments/eval_results/teacher_gpt4/short_prompts/gpt4`)
- GPT-3.5: OK (0% JSON validity; `data_work/.experiments/eval_results/teacher_gpt35/short_prompts/gpt35`)
- Mistral-7B: OK (0% JSON validity; `data_work/.experiments/eval_results/teacher_mistral/short_prompts/mixtral7b`)
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
  --llm-scorer mistral/voxtral-small-latest \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/clap_vs_llm
```

Run notes (2026-01-17):
- clap vs llm: PENDING

---

### LLM scorer model comparison

Commands:
```bash
conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer gemini/gemini-3-flash-preview \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/llm_gemini_flash

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer gemini/gemini-3-pro-preview \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/llm_gemini_pro

conda run -n latentscore-data python -m data_work.06_eval_suite \
  --eval-set short_prompts \
  --baseline random \
  --llm-scorer mistral/voxtral-small-latest \
  --limit 1 \
  --output-dir data_work/.experiments/eval_results/llm_voxtral
```

Run notes (2026-01-17):
- gemini flash: PENDING
- gemini pro: PENDING
- voxtral: PENDING

---

### LLM scorer as GRPO reward

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train --advanced grpo \
  --data data_work/.experiments/mini_train.jsonl \
  --model /outputs/exp-sft-baseline \
  --output exp-grpo-llm-reward \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --audio-reward data_work.lib.llm_scorer:score_config_with_llm \
  --overwrite
```

Run notes (2026-01-17):
- grpo llm reward: PENDING

---

## Prompt ablations

### System prompt variants

Commands:
```bash
conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-prompt-default \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --overwrite

conda run -n latentscore-data python -m data_work.03_modal_train sft \
  --data data_work/.experiments/mini_train.jsonl \
  --output exp-sft-prompt-detailed \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 1024 \
  --system-prompt "You are an expert sound designer. Return only JSON matching the schema." \
  --overwrite
```

Run notes (2026-01-17):
- default prompt: PENDING
- detailed prompt: PENDING
