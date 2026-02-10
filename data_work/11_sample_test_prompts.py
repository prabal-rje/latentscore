"""Sample a compact prompt set from the synthetic TEST split.

This is intended for:
- quick CLAP benchmarking across tiers
- performance benchmarking (latency/RTF)
- human-study stimulus selection

Source of truth is the embedding map produced by `10_export_embedding_map`, which is
also the artifact used by the "fast" (retrieval) tier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBED_MAP_REPO = "guprab/latentscore-data"
DEFAULT_EMBED_MAP_FILE = "2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl"


def _resolve_embed_map_path() -> Path:
    explicit = os.environ.get("LATENTSCORE_EMBED_MAP", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    repo = os.environ.get("LATENTSCORE_EMBED_MAP_REPO", DEFAULT_EMBED_MAP_REPO)
    filename = os.environ.get("LATENTSCORE_EMBED_MAP_FILE", DEFAULT_EMBED_MAP_FILE)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - environment mismatch
        raise SystemExit(
            "huggingface_hub is required (or set LATENTSCORE_EMBED_MAP to a local file)."
        ) from exc

    base_dir = Path(os.environ.get("LATENTSCORE_MODEL_DIR", "")).expanduser()
    if not base_dir:
        base_dir = Path.home() / ".cache" / "latentscore" / "models"
    cache_dir = base_dir / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _is_ascii(text: str) -> bool:
    return text.isascii()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample prompts from the synthetic TEST split (embedding map).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_work/.experiments/nime_2026_test_subset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of prompts to sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling RNG seed.")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=12,
        help="Minimum prompt length (characters).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=160,
        help="Maximum prompt length (characters).",
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Require ASCII-only prompts (helps with human-study readability).",
    )
    parser.add_argument(
        "--embed-map",
        type=Path,
        default=None,
        help="Optional path to embedding map JSONL. If omitted, downloads via HF Hub.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    embed_map_path = args.embed_map.expanduser().resolve() if args.embed_map else _resolve_embed_map_path()
    if not embed_map_path.exists():
        raise SystemExit(f"Embedding map not found: {embed_map_path}")

    candidates: list[dict[str, Any]] = []
    for row in _iter_rows(embed_map_path):
        split = row.get("split")
        if not isinstance(split, str) or split.upper() != "TEST":
            continue
        vibe = row.get("vibe_original") or row.get("vibe")
        if not isinstance(vibe, str):
            continue
        vibe = vibe.strip()
        if len(vibe) < args.min_chars or len(vibe) > args.max_chars:
            continue
        if args.ascii_only and not _is_ascii(vibe):
            continue
        candidates.append(row)

    if not candidates:
        raise SystemExit("No candidates found. Try relaxing filters (--min-chars/--max-chars/--ascii-only).")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: max(0, min(args.n_samples, len(candidates)))]

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(selected, start=1):
            vibe = (row.get("vibe_original") or row.get("vibe") or "").strip()
            record = {
                "id": f"test_subset_{idx:04d}",
                "dataset": row.get("dataset"),
                "id_in_dataset": row.get("id_in_dataset"),
                "split": "TEST",
                "vibe_original": vibe,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "embed_map_path": str(embed_map_path),
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "seed": args.seed,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "ascii_only": bool(args.ascii_only),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("Embedding map: %s", embed_map_path)
    LOGGER.info("Candidates: %d", len(candidates))
    LOGGER.info("Selected: %d", len(selected))
    LOGGER.info("Wrote: %s", output_path)
    LOGGER.info("Meta: %s", meta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

