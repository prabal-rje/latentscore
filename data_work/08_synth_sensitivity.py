"""Synth sensitivity analysis: vary config fields and compare audio features."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from common.music_schema import PROMPT_REGISTER_MAX, PROMPT_REGISTER_MIN
from data_work.lib.audio_features import FEATURE_KEYS, compute_audio_features, feature_distance
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import SAMPLE_RATE, assemble

FLOAT_FIELDS = {
    "tempo",
    "brightness",
    "space",
    "motion",
    "stereo",
    "echo",
    "human",
    "melody_density",
    "syncopation",
    "swing",
    "motif_repeat_prob",
    "step_bias",
    "chromatic_prob",
    "cadence_strength",
}

INT_FIELD_OPTIONS = {
    "density": [2, 3, 4, 5, 6],
    "phrase_len_bars": [2, 4, 8],
    "chord_change_bars": [1, 2, 4],
    "register_min_oct": list(range(PROMPT_REGISTER_MIN, PROMPT_REGISTER_MAX + 1)),
    "register_max_oct": list(range(PROMPT_REGISTER_MIN, PROMPT_REGISTER_MAX + 1)),
}

BOOL_FIELDS = {"depth"}


def default_internal_config():
    return MusicConfig().to_internal()


def extract_config_payload(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = row.get("config_payload") or row.get("config")
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, Mapping):
        return None
    if "config" in payload and isinstance(payload["config"], Mapping):
        return payload["config"]
    return payload


def parse_music_config(payload: Mapping[str, Any]) -> MusicConfig:
    try:
        prompt = MusicConfigPrompt.model_validate(payload)
        return prompt.to_config()
    except Exception:
        return MusicConfig.model_validate(payload)


def _clamp_float(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _step_option(options: Sequence[int], value: int, direction: int) -> int:
    if value not in options:
        options = sorted(options)
        value = min(options, key=lambda opt: abs(opt - value))
    index = options.index(value)
    next_index = max(0, min(len(options) - 1, index + direction))
    return options[next_index]


def build_variant(config, field: str, delta: float, direction: int) -> Any:
    if field in FLOAT_FIELDS:
        new_value = _clamp_float(float(getattr(config, field)) + (delta * direction))
        return config.model_copy(update={field: new_value})
    if field in INT_FIELD_OPTIONS:
        new_value = _step_option(INT_FIELD_OPTIONS[field], int(getattr(config, field)), direction)
        update = {field: new_value}
        updated = config.model_copy(update=update)
        if field == "register_min_oct" and updated.register_min_oct > updated.register_max_oct:
            updated = updated.model_copy(update={"register_max_oct": updated.register_min_oct})
        if field == "register_max_oct" and updated.register_max_oct < updated.register_min_oct:
            updated = updated.model_copy(update={"register_min_oct": updated.register_max_oct})
        return updated
    if field in BOOL_FIELDS:
        return config.model_copy(update={field: not getattr(config, field)})
    raise ValueError(f"Unsupported field: {field}")


def config_proxy_features(config) -> dict[str, float]:
    return {
        "rms": float(config.density) / 6.0,
        "spectral_centroid": float(config.brightness),
        "spectral_bandwidth": float(config.space),
        "zero_crossing_rate": float(config.tempo),
        "onset_strength": float(config.motion),
    }


def compute_features(config, *, duration: float, seed: int, dry_run: bool) -> dict[str, float]:
    if dry_run:
        return config_proxy_features(config)
    rng = np.random.default_rng(seed)
    audio = assemble(config, duration=duration, rng=rng)
    return compute_audio_features(audio, SAMPLE_RATE)


def run_sensitivity(
    *,
    input_path: Path,
    output_dir: Path,
    fields: Sequence[str],
    delta: float,
    limit: int,
    duration: float,
    seed: int,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "synth_sensitivity.jsonl"
    summary_path = output_dir / "summary.json"

    per_field_deltas: dict[str, list[dict[str, float]]] = {field: [] for field in fields}
    per_field_distances: dict[str, list[float]] = {field: [] for field in fields}

    processed = 0
    skipped = 0

    with (
        input_path.open("r", encoding="utf-8") as handle,
        results_path.open("w", encoding="utf-8") as out,
    ):
        for line in handle:
            if limit and processed >= limit:
                break
            row = json.loads(line)
            payload = extract_config_payload(row)
            if payload is None:
                skipped += 1
                continue
            config = parse_music_config(payload)
            internal = config.to_internal()
            base_features = compute_features(
                internal, duration=duration, seed=seed + processed, dry_run=dry_run
            )
            record_id = row.get("id") or row.get("id_in_dataset") or str(processed)
            for field in fields:
                variant = build_variant(internal, field, delta, direction=1)
                variant_features = compute_features(
                    variant, duration=duration, seed=seed + processed, dry_run=dry_run
                )
                delta_features = {
                    key: float(variant_features[key] - base_features[key]) for key in FEATURE_KEYS
                }
                distance = feature_distance(base_features, variant_features)
                entry = {
                    "record_id": record_id,
                    "field": field,
                    "base_value": float(getattr(internal, field)),
                    "variant_value": float(getattr(variant, field)),
                    "delta": float(getattr(variant, field) - getattr(internal, field)),
                    "features_source": "proxy" if dry_run else "audio",
                    "base_features": base_features,
                    "variant_features": variant_features,
                    "delta_features": delta_features,
                    "distance": distance,
                }
                out.write(json.dumps(entry, sort_keys=True) + "\n")
                per_field_deltas[field].append(delta_features)
                per_field_distances[field].append(distance)
            processed += 1

    per_field_summary: dict[str, dict[str, Any]] = {}
    for field in fields:
        deltas = per_field_deltas[field]
        distances = per_field_distances[field]
        mean_abs_delta = {
            key: (statistics.mean(abs(item[key]) for item in deltas) if deltas else 0.0)
            for key in FEATURE_KEYS
        }
        per_field_summary[field] = {
            "count": len(deltas),
            "mean_distance": statistics.mean(distances) if distances else 0.0,
            "mean_abs_delta": mean_abs_delta,
        }

    summary = {
        "input": str(input_path),
        "processed": processed,
        "skipped": skipped,
        "fields": list(fields),
        "features_source": "proxy" if dry_run else "audio",
        "per_field": per_field_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _parse_fields(values: Iterable[str]) -> list[str]:
    fields: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                fields.append(item)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vary config fields and compare audio feature deltas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL with configs")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--fields", action="append", default=[], help="Fields to vary")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta for float fields")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process")
    parser.add_argument("--duration", type=float, default=0.5, help="Audio duration seconds")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--dry-run", action="store_true", help="Skip audio and use proxy features")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    fields = _parse_fields(args.fields)
    if not fields:
        fields = sorted(FLOAT_FIELDS | INT_FIELD_OPTIONS.keys() | BOOL_FIELDS)
    run_sensitivity(
        input_path=args.input,
        output_dir=args.output_dir,
        fields=fields,
        delta=args.delta,
        limit=args.limit,
        duration=args.duration,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
