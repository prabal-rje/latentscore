#!/usr/bin/env python3
"""Export SFT/GRPO datasets to Fireworks-style JSONL chat format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _fireworks_line(system_prompt: str, user_prompt: str, assistant: str | None) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def _split_rows(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * (1.0 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    return train_rows, val_rows


def _export(
    *,
    source: Path,
    out_train: Path,
    out_val: Path,
    system_prompt: str,
    prompt_field: str,
    response_field: str,
    val_ratio: float,
    seed: int,
    include_assistant: bool,
) -> None:
    from data_work.lib.llm_client import wrap_vibe_for_chat

    rows = []
    for row in _load_jsonl(source):
        prompt = row.get(prompt_field)
        response = row.get(response_field)
        if prompt in (None, ""):
            continue
        if include_assistant and response in (None, ""):
            continue
        user_prompt = wrap_vibe_for_chat(str(prompt))
        assistant = None
        if include_assistant:
            assistant = json.dumps(response, ensure_ascii=False)
        rows.append(_fireworks_line(system_prompt, user_prompt, assistant))

    train_rows, val_rows = _split_rows(rows, val_ratio=val_ratio, seed=seed)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, dataset in ((out_train, train_rows), (out_val, val_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for item in dataset:
                handle.write(json.dumps(item, ensure_ascii=False))
                handle.write("\n")
    print(f"Wrote {len(train_rows):,} train rows to {out_train}")
    print(f"Wrote {len(val_rows):,} val rows to {out_val}")


def main() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default="data_work/2026-01-26_scored/SFT-Train.jsonl")
    parser.add_argument("--grpo", default="data_work/2026-01-26_scored/GRPO.jsonl")
    parser.add_argument("--out-dir", default="data_work/fireworks")
    parser.add_argument("--prompt-field", default="vibe_noisy")
    parser.add_argument("--response-field", default="config_payload")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omit-assistant", action="store_true")
    args = parser.parse_args()

    from common.prompt_registry import render_config_prompt

    system_prompt = render_config_prompt()

    out_dir = Path(args.out_dir)
    include_assistant = not args.omit_assistant

    _export(
        source=Path(args.sft),
        out_train=out_dir / "sft_train.jsonl",
        out_val=out_dir / "sft_val.jsonl",
        system_prompt=system_prompt,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_assistant=include_assistant,
    )

    _export(
        source=Path(args.grpo),
        out_train=out_dir / "grpo_train.jsonl",
        out_val=out_dir / "grpo_val.jsonl",
        system_prompt=system_prompt,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_assistant=include_assistant,
    )


if __name__ == "__main__":
    main()
