#!/usr/bin/env python
"""Generate audio files from config JSONs.

Usage:
    PYTHONPATH=src python -m data_work.generate_audio_from_configs \
        --jsonl scored.jsonl \
        --output-dir ./audio \
        --duration 30 \
        --all-candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure latentscore is importable
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

from latentscore.audio import write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble


def config_dict_to_music_config(config_dict: dict) -> MusicConfig:
    """Convert config dict to MusicConfig, handling string labels."""
    try:
        # Try MusicConfigPrompt first (string labels like "sparse", "light")
        prompt_config = MusicConfigPrompt.model_validate(config_dict)
        return prompt_config.to_config()
    except Exception:
        # Fallback: maybe it's already in MusicConfig format (numeric values)
        return MusicConfig.model_validate(config_dict)


def generate_single(
    config_dict: dict,
    output_path: Path,
    duration: float,
    vibe: str | None = None,
) -> None:
    """Generate a single audio file from config dict."""
    if output_path.exists():
        print(f"  Skipping (exists): {output_path.name}")
        return

    print(f"  Generating: {output_path.name}")
    try:
        music_config = config_dict_to_music_config(config_dict)
        internal = music_config.to_internal()
        audio = assemble(internal, duration=duration)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(output_path, audio)
    except Exception as e:
        print(f"    ERROR: {e}")


def generate_from_jsonl(
    jsonl_path: Path,
    output_dir: Path,
    duration: float,
    all_candidates: bool = False,
    limit: int = 0,
) -> None:
    """Generate audio from scored JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f]

    if limit > 0:
        rows = rows[:limit]

    total = 0
    generated = 0
    for row_idx, row in enumerate(rows):
        vibe = row.get("vibe_noisy", row.get("vibe_original", "ambient"))
        safe_vibe = "".join(c if c.isalnum() else "_" for c in vibe[:25])
        candidates = row.get("config_candidates", [])
        best_idx = row.get("best_index", 0)

        if all_candidates:
            for cand_idx, cand in enumerate(candidates):
                if cand is None:
                    continue
                config = cand.get("config", {})
                if not config:
                    continue

                is_best = "_BEST" if cand_idx == best_idx else ""
                filename = f"row{row_idx:02d}_cand{cand_idx}_{safe_vibe}{is_best}.wav"
                output_path = output_dir / filename
                total += 1
                print(f"[{total}] Row {row_idx}, Cand {cand_idx}{' (BEST)' if is_best else ''}")
                before = output_path.exists()
                generate_single(config, output_path, duration, vibe)
                if output_path.exists() and not before:
                    generated += 1
        else:
            if 0 <= best_idx < len(candidates) and candidates[best_idx]:
                config = candidates[best_idx].get("config", {})
                if config:
                    filename = f"row{row_idx:02d}_{safe_vibe}_BEST.wav"
                    output_path = output_dir / filename
                    total += 1
                    print(f"[{total}] Row {row_idx} (best)")
                    before = output_path.exists()
                    generate_single(config, output_path, duration, vibe)
                    if output_path.exists() and not before:
                        generated += 1

    print(f"\nDone! Generated {generated}/{total} audio files in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio from configs")
    parser.add_argument("--jsonl", type=Path, required=True, help="Scored JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--duration", type=float, default=30.0, help="Audio duration (default: 30)")
    parser.add_argument("--all-candidates", action="store_true", help="Generate all candidates")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 = no limit)")

    args = parser.parse_args()
    generate_from_jsonl(args.jsonl, args.output_dir, args.duration, args.all_candidates, args.limit)


if __name__ == "__main__":
    main()
