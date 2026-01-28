"""Render audio WAVs from inference results.jsonl configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_work.lib.jsonl_io import iter_jsonl
from latentscore.audio import write_wav
from latentscore.config import MusicConfigPrompt, MusicConfigPromptPayload
from latentscore.synth import assemble


def _extract_payload(config_value: Any) -> Mapping[str, Any] | None:
    if config_value is None:
        return None
    if isinstance(config_value, str):
        try:
            parsed = json.loads(config_value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    if isinstance(config_value, Mapping):
        return config_value
    return None


def _extract_config(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "config" in payload and isinstance(payload["config"], Mapping):
        return payload["config"]
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render WAV files from inference results.jsonl configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to results.jsonl with config payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write WAV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of rows to render (0 = all).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Audio duration in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for row in iter_jsonl(input_path):
        if args.limit and count >= args.limit:
            break
        payload = _extract_payload(row.get("config"))
        if payload is None:
            continue
        config_payload = _extract_config(payload)
        if config_payload is None:
            continue

        try:
            if "config" in payload:
                wrapper = MusicConfigPromptPayload.model_validate(payload)
                prompt_config = wrapper.config
                title = wrapper.title
            else:
                prompt_config = MusicConfigPrompt.model_validate(config_payload)
                title = row.get("title") or f"track_{count+1:03d}"
        except Exception:
            continue

        music_config = prompt_config.to_config().to_internal()
        rng = np.random.default_rng(args.seed + count)
        audio = assemble(music_config, duration=args.duration, rng=rng)

        safe_title = "".join(ch for ch in str(title) if ch.isalnum() or ch in ("-", "_")).strip()
        if not safe_title:
            safe_title = f"track_{count+1:03d}"
        output_path = output_dir / f"{count+1:03d}_{safe_title}.wav"
        write_wav(output_path, audio)
        count += 1

    print(f"Wrote {count} wav files to {output_dir}")


if __name__ == "__main__":
    main()
