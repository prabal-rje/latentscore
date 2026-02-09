"""Generate and analyze human eval packs (forced-choice A/B alignment).

This script focuses on stimulus generation: given a prompt set and two model
specs, it renders deterministic audio clips and writes a blinded trial manifest.

Typical usage:
1) Sample a prompt subset: `python -m data_work.11_sample_test_prompts ...`
2) Generate a pack:        `python -m data_work.07_human_eval_pack generate ...`
3) Run a survey tool using `stimuli.csv` + audio files
4) Analyze with:           `python -m data_work.07_human_eval_pack analyze ...`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, Mapping, Sequence, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from data_work.lib.jsonl_io import iter_jsonl
from data_work.lib.llm_client import (
    LocalHFClient,
    litellm_structured_completion,
    load_env_file,
    normalize_model_and_base,
    resolve_api_key_for_models,
    wrap_vibe_for_chat,
)
from data_work.lib.music_prompt import build_music_prompt
from data_work.lib.music_schema import MusicConfigPromptPayload
from latentscore.audio import ensure_audio_contract, write_wav
from latentscore.config import MusicConfig, MusicConfigPrompt
from latentscore.synth import assemble

LOGGER = logging.getLogger("data_work.human_eval_pack")

DEFAULT_SYSTEM_PROMPT = build_music_prompt()
DEFAULT_VIBE_FIELD = "vibe_original"

AudioArray = NDArray[np.floating[Any]]
T = TypeVar("T")


@dataclass
class ModelSpec:
    kind: str  # "baseline" | "local" | "litellm"
    value: str
    label: str
    api_base: str | None = None


class Trial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    prompt: str
    clip_1: str
    clip_2: str


class TrialKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    prompt: str
    seed: int
    clip_1_model: str
    clip_2_model: str
    # Keep configs for auditability / debugging.
    clip_1_config: dict[str, Any] | None = None
    clip_2_config: dict[str, Any] | None = None


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return (base_seed + value) % (2**32)


def _parse_model_spec(raw: str) -> ModelSpec:
    # Supported:
    # - baseline:<name>[:label]
    # - local:<hf_path>[:label]
    # - litellm:<model>[:label]
    kind_value, _, label = raw.partition(":")
    if not _:
        raise SystemExit(
            f"Invalid model spec {raw!r}. Expected kind:value (e.g., baseline:retrieval)."
        )
    kind = kind_value.strip().lower()
    remainder = label  # everything after first ':'

    value, _, label2 = remainder.partition(":")
    value = value.strip()
    if label2:
        label_final = label2.strip()
    else:
        if kind == "local":
            label_final = Path(value).name
        else:
            label_final = value

    if kind not in {"baseline", "local", "litellm"}:
        raise SystemExit(f"Unknown model kind: {kind!r} (expected baseline/local/litellm)")
    if not value:
        raise SystemExit(f"Missing model value in spec: {raw!r}")
    return ModelSpec(kind=kind, value=value, label=label_final)


def _parse_config_to_internal(config_payload: Mapping[str, Any]) -> Any:
    # Prompt schema first, then numeric.
    try:
        prompt = MusicConfigPrompt.model_validate(config_payload)
        return prompt.to_config().to_internal()
    except Exception:
        return MusicConfig.model_validate(config_payload).to_internal()


def _render_audio(internal_config: Any, *, duration: float, seed: int) -> AudioArray:
    rng = np.random.default_rng(seed)
    return assemble(internal_config, duration=duration, rng=rng)


def _rms(audio: AudioArray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _rms_normalize(audio: AudioArray, *, target_rms_db: float) -> AudioArray:
    """Normalize to a target RMS level (dBFS-ish).

    This keeps A/B comparisons fairer than only peak-normalizing.
    """
    normalized = ensure_audio_contract(audio, check_peak=False)
    current = _rms(normalized)
    if current <= 1e-8:
        return ensure_audio_contract(normalized, check_peak=True)
    target = float(10 ** (target_rms_db / 20.0))
    scaled = normalized * (target / current)
    return ensure_audio_contract(scaled, check_peak=True)


def _load_prompts_from_input(
    path: Path,
    *,
    vibe_field: str,
    limit: int,
) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    count = 0
    for row in iter_jsonl(path):
        count += 1
        if limit and count > limit:
            break
        vibe = row.get(vibe_field)
        if not isinstance(vibe, str) or not vibe.strip():
            continue
        record_id = str(row.get("id") or row.get("id_in_dataset") or count)
        prompts.append((record_id, vibe.strip()))
    return prompts


def _select_prompts(
    prompts: list[tuple[str, str]],
    *,
    n_samples: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    ascii_only: bool,
) -> list[tuple[str, str]]:
    filtered: list[tuple[str, str]] = []
    for record_id, vibe in prompts:
        if len(vibe) < min_chars or len(vibe) > max_chars:
            continue
        if ascii_only and not vibe.isascii():
            continue
        filtered.append((record_id, vibe))
    if not filtered:
        raise SystemExit("No prompts left after filtering. Relax --min-chars/--max-chars/--ascii-only.")

    rng = random.Random(seed)
    rng.shuffle(filtered)
    if n_samples <= 0 or n_samples >= len(filtered):
        return filtered
    return filtered[:n_samples]


def _generate_payload(
    spec: ModelSpec,
    *,
    vibe: str,
    system_prompt: str,
    api_key: str | None,
    api_base: str | None,
    model_kwargs: Mapping[str, Any],
    local_clients: Mapping[str, LocalHFClient],
    baseline_clients: Mapping[str, Any],
) -> MusicConfigPromptPayload:
    match spec.kind:
        case "baseline":
            baseline = baseline_clients[spec.label]
            return baseline.generate(vibe)
        case "local":
            client = local_clients[spec.label]
            prompt = client.format_chat_prompt(
                system_prompt=system_prompt,
                user_prompt=wrap_vibe_for_chat(vibe),
            )
            return client.generate_structured(prompt, MusicConfigPromptPayload)
        case "litellm":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": wrap_vibe_for_chat(vibe)},
            ]
            return asyncio_run(
                litellm_structured_completion(
                    model=spec.value,
                    messages=messages,
                    response_model=MusicConfigPromptPayload,
                    api_key=api_key,
                    api_base=api_base,
                    model_kwargs=model_kwargs,
                )
            )
        case _:
            raise SystemExit(f"Unsupported model kind: {spec.kind}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human eval pack generator/analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a human eval pack")
    generate.add_argument("--input", type=Path, required=True, help="Input JSONL with prompts.")
    generate.add_argument("--vibe-field", type=str, default=DEFAULT_VIBE_FIELD)
    generate.add_argument("--n-samples", type=int, default=36, help="Prompts to include (0=all).")
    generate.add_argument("--seed", type=int, default=42, help="Sampling + A/B RNG seed.")
    generate.add_argument("--min-chars", type=int, default=12)
    generate.add_argument("--max-chars", type=int, default=160)
    generate.add_argument("--ascii-only", action="store_true")
    generate.add_argument("--limit", type=int, default=0, help="Limit input rows (0=all).")

    generate.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="Model A spec: baseline:<name>[:label] | local:<path>[:label] | litellm:<model>[:label]",
    )
    generate.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Model B spec: baseline:<name>[:label] | local:<path>[:label] | litellm:<model>[:label]",
    )

    generate.add_argument("--duration", type=float, default=12.0, help="Clip duration seconds.")
    generate.add_argument("--target-rms-db", type=float, default=-20.0, help="RMS normalize level.")
    generate.add_argument("--output-dir", type=Path, required=True, help="Output directory.")

    generate.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    generate.add_argument("--env-file", type=Path, default=None)
    generate.add_argument("--api-key", type=str, default=None)
    generate.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    generate.add_argument("--api-base", type=str, default=None)
    generate.add_argument("--model-kwargs", type=str, default="{}", help="JSON dict of LiteLLM kwargs.")

    generate.add_argument("--local-device", type=str, default=None)
    generate.add_argument("--local-max-new-tokens", type=int, default=512)
    generate.add_argument("--local-temperature", type=float, default=0.0)

    analyze = subparsers.add_parser("analyze", help="Analyze forced-choice responses (simple).")
    analyze.add_argument("--key", type=Path, required=True, help="Path to key.jsonl")
    analyze.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="CSV with columns trial_id,choice (choice is 1 or 2).",
    )
    analyze.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")

    return parser


def _generate(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit("--model-kwargs must be valid JSON.") from exc
    if not isinstance(model_kwargs, dict):
        raise SystemExit("--model-kwargs must be a JSON object.")
    model_kwargs = cast(dict[str, Any], model_kwargs)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    prompts = _load_prompts_from_input(
        input_path,
        vibe_field=args.vibe_field,
        limit=args.limit,
    )
    selected = _select_prompts(
        prompts,
        n_samples=args.n_samples,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        ascii_only=bool(args.ascii_only),
    )

    # Parse model specs
    model_a = _parse_model_spec(args.model_a)
    model_b = _parse_model_spec(args.model_b)

    # Normalize litellm model specs (api_base etc) and resolve API key.
    litellm_models: list[tuple[str, str | None]] = []
    for spec in (model_a, model_b):
        if spec.kind != "litellm":
            continue
        model, api_base = normalize_model_and_base(spec.value, args.api_base)
        spec.value = model
        spec.api_base = api_base
        litellm_models.append((model, api_base))

    api_key = resolve_api_key_for_models(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        models=litellm_models,
    )

    # Construct clients once.
    local_clients: dict[str, LocalHFClient] = {}
    for spec in (model_a, model_b):
        if spec.kind != "local":
            continue
        if spec.label in local_clients:
            continue
        local_clients[spec.label] = LocalHFClient(
            spec.value,
            device=args.local_device,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
        )

    baseline_clients: dict[str, Any] = {}
    if model_a.kind == "baseline" or model_b.kind == "baseline":
        from data_work.lib.baselines import get_baseline

        for spec in (model_a, model_b):
            if spec.kind != "baseline":
                continue
            if spec.label in baseline_clients:
                continue
            baseline_clients[spec.label] = get_baseline(spec.value)

    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    trials_path = output_dir / "trials.jsonl"
    key_path = output_dir / "key.jsonl"
    csv_path = output_dir / "stimuli.csv"
    pack_path = output_dir / "pack.json"

    rng = random.Random(args.seed)

    trials: list[Trial] = []
    keys: list[TrialKey] = []

    for idx, (record_id, vibe) in enumerate(selected, start=1):
        trial_id = f"t{idx:03d}"
        seed = _stable_seed(args.seed, f"{record_id}:{vibe}")

        # Decide which model is clip_1 vs clip_2 (blinding).
        if rng.random() < 0.5:
            clip1_spec, clip2_spec = model_a, model_b
        else:
            clip1_spec, clip2_spec = model_b, model_a

        clip1_path = audio_dir / f"{trial_id}_1.wav"
        clip2_path = audio_dir / f"{trial_id}_2.wav"

        # Generate both configs + audio deterministically.
        clip1_payload = _generate_payload(
            clip1_spec,
            vibe=vibe,
            system_prompt=args.system_prompt,
            api_key=api_key,
            api_base=clip1_spec.api_base,
            model_kwargs=model_kwargs,
            local_clients=local_clients,
            baseline_clients=baseline_clients,
        )
        clip2_payload = _generate_payload(
            clip2_spec,
            vibe=vibe,
            system_prompt=args.system_prompt,
            api_key=api_key,
            api_base=clip2_spec.api_base,
            model_kwargs=model_kwargs,
            local_clients=local_clients,
            baseline_clients=baseline_clients,
        )

        clip1_internal = _parse_config_to_internal(clip1_payload.config.model_dump())
        clip2_internal = _parse_config_to_internal(clip2_payload.config.model_dump())

        clip1_audio = _render_audio(clip1_internal, duration=args.duration, seed=seed)
        clip2_audio = _render_audio(clip2_internal, duration=args.duration, seed=seed)

        clip1_audio = _rms_normalize(clip1_audio, target_rms_db=args.target_rms_db)
        clip2_audio = _rms_normalize(clip2_audio, target_rms_db=args.target_rms_db)

        write_wav(clip1_path, clip1_audio)
        write_wav(clip2_path, clip2_audio)

        trials.append(
            Trial(
                trial_id=trial_id,
                prompt=vibe,
                clip_1=str(clip1_path.relative_to(output_dir)),
                clip_2=str(clip2_path.relative_to(output_dir)),
            )
        )
        keys.append(
            TrialKey(
                trial_id=trial_id,
                prompt=vibe,
                seed=seed,
                clip_1_model=clip1_spec.label,
                clip_2_model=clip2_spec.label,
                clip_1_config=clip1_payload.config.model_dump(),
                clip_2_config=clip2_payload.config.model_dump(),
            )
        )

        LOGGER.info("Rendered %s (%s vs %s)", trial_id, clip1_spec.label, clip2_spec.label)

    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(trial.model_dump_json() + "\n")

    with key_path.open("w", encoding="utf-8") as handle:
        for row in keys:
            handle.write(row.model_dump_json() + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "prompt", "clip_1", "clip_2"])
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial.model_dump())

    pack_meta = {
        "input": str(input_path),
        "vibe_field": args.vibe_field,
        "n_samples": len(trials),
        "seed": args.seed,
        "duration": args.duration,
        "target_rms_db": args.target_rms_db,
        "model_a": vars(model_a),
        "model_b": vars(model_b),
    }
    pack_path.write_text(json.dumps(pack_meta, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("Wrote trials: %s", trials_path)
    LOGGER.info("Wrote key: %s", key_path)
    LOGGER.info("Wrote stimuli CSV: %s", csv_path)
    LOGGER.info("Wrote pack meta: %s", pack_path)


def _analyze(args: argparse.Namespace) -> None:
    key_path = args.key.expanduser().resolve()
    responses_path = args.responses.expanduser().resolve()
    if not key_path.exists():
        raise SystemExit(f"Key not found: {key_path}")
    if not responses_path.exists():
        raise SystemExit(f"Responses not found: {responses_path}")

    key_by_trial: dict[str, TrialKey] = {}
    with key_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = TrialKey.model_validate_json(line)
            key_by_trial[row.trial_id] = row

    # Count wins by model label.
    wins: dict[str, int] = {}
    total = 0
    with responses_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trial_id = (row.get("trial_id") or "").strip()
            choice_raw = (row.get("choice") or "").strip()
            if not trial_id or trial_id not in key_by_trial:
                continue
            if choice_raw not in {"1", "2"}:
                continue
            key = key_by_trial[trial_id]
            chosen = key.clip_1_model if choice_raw == "1" else key.clip_2_model
            wins[chosen] = wins.get(chosen, 0) + 1
            total += 1

    summary: dict[str, Any] = {
        "n_trials": len(key_by_trial),
        "n_responses_used": total,
        "wins": wins,
        "win_rates": {k: (v / total if total else 0.0) for k, v in wins.items()},
    }

    # Add a quick pairwise delta if exactly two models show up.
    if len(wins) == 2 and total:
        labels = sorted(wins.keys())
        a, b = labels[0], labels[1]
        summary["pairwise"] = {
            "a": a,
            "b": b,
            "a_minus_b": (wins[a] - wins[b]) / total,
        }

    if args.output:
        out_path = args.output.expanduser().resolve()
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote analysis: %s", out_path)
    else:
        print(json.dumps(summary, indent=2))


def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    import asyncio

    return asyncio.run(coro)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        _generate(args)
    elif args.command == "analyze":
        _analyze(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
