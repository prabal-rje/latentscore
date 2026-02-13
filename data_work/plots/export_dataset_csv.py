"""Export the embedding-map JSONL to a flat CSV for the supplementary website.

Flattens nested config (34 fields), palettes (3×5 hex colors), and
embedding (384-dim vector) into individual columns.

Run:
    python -m data_work.plots.export_dataset_csv
    python -m data_work.plots.export_dataset_csv --input path/to/map.jsonl --output out.csv
    python -m data_work.plots.export_dataset_csv --limit 100  # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data_work/.experiments/vibe_and_embeddings_to_config_map.jsonl"
)
DEFAULT_OUTPUT = Path("data_work/plots/dataset_export.csv")

CONFIG_FIELDS = [
    "tempo", "root", "mode", "brightness", "space", "density",
    "bass", "pad", "melody", "rhythm", "texture", "accent",
    "motion", "attack", "stereo", "depth", "echo", "human", "grain",
    "melody_engine", "phrase_len_bars", "melody_density", "syncopation",
    "swing", "motif_repeat_prob", "step_bias", "chromatic_prob",
    "cadence_strength", "register_min_oct", "register_max_oct",
    "tension_curve", "harmony_style", "chord_change_bars", "chord_extensions",
]

PALETTE_COUNT = 3
COLORS_PER_PALETTE = 5

# Every palette in the dataset uses this fixed weight sequence.
# Column position → weight is deterministic, so the CSV is losslessly reversible.
PALETTE_WEIGHTS: tuple[str, ...] = ("xxl", "xl", "lg", "md", "sm")


def _palette_columns() -> list[str]:
    """Generate column names for palette_{i}_color_{j}."""
    return [
        f"palette_{p}_color_{c}"
        for p in range(1, PALETTE_COUNT + 1)
        for c in range(1, COLORS_PER_PALETTE + 1)
    ]


def _embedding_columns(dim: int = 384) -> list[str]:
    return [f"emb_{i}" for i in range(dim)]


def _build_header() -> list[str]:
    return (
        ["dataset", "id_in_dataset", "split", "vibe_original", "title"]
        + [f"config.{f}" for f in CONFIG_FIELDS]
        + _palette_columns()
        + _embedding_columns()
    )


def _flatten_row(row: dict[str, Any]) -> list[str]:
    cfg = row.get("config") or {}
    palettes = row.get("palettes") or []
    embedding = row.get("embedding") or []

    flat: list[str] = [
        str(row.get("dataset", "")),
        str(row.get("id_in_dataset", "")),
        str(row.get("split", "")),
        row.get("vibe_original", ""),
        row.get("title", ""),
    ]

    # Config fields
    for field in CONFIG_FIELDS:
        val = cfg.get(field, "")
        flat.append(str(val).lower() if isinstance(val, bool) else str(val))

    # Palette hex colors (3 palettes × 5 colors)
    for p_idx in range(PALETTE_COUNT):
        palette = palettes[p_idx] if p_idx < len(palettes) else {}
        colors = palette.get("colors", [])
        for c_idx in range(COLORS_PER_PALETTE):
            hex_val = colors[c_idx]["hex"] if c_idx < len(colors) else ""
            flat.append(hex_val)

    # Embedding vector
    for i in range(384):
        flat.append(f"{embedding[i]:.6f}" if i < len(embedding) else "")

    return flat


# ── Type-aware config field coercion ──

_BOOL_FIELDS = {"depth"}
_INT_FIELDS = {"density", "phrase_len_bars", "register_min_oct", "register_max_oct"}


def _coerce_config_value(field: str, raw: str) -> str | bool | int:
    """Coerce a flat CSV string back to its typed value."""
    if field in _BOOL_FIELDS:
        return raw.lower() == "true"
    if field in _INT_FIELDS:
        return int(raw)
    return raw


# ── CSV → JSONL (reverse) ──


def _unflatten_row(flat: dict[str, str]) -> dict[str, Any]:
    """Reconstruct the original JSONL structure from a flat CSV row."""
    config: dict[str, str | bool | int] = {}
    for field in CONFIG_FIELDS:
        config[field] = _coerce_config_value(field, flat[f"config.{field}"])

    palettes: list[dict[str, list[dict[str, str]]]] = []
    for p in range(1, PALETTE_COUNT + 1):
        colors: list[dict[str, str]] = []
        for c in range(1, COLORS_PER_PALETTE + 1):
            colors.append({
                "hex": flat[f"palette_{p}_color_{c}"],
                "weight": PALETTE_WEIGHTS[c - 1],
            })
        palettes.append({"colors": colors})

    embedding = [float(flat[f"emb_{i}"]) for i in range(384)]

    return {
        "dataset": flat["dataset"],
        "id_in_dataset": flat["id_in_dataset"],
        "split": flat["split"],
        "vibe_original": flat["vibe_original"],
        "embedding": embedding,
        "title": flat["title"],
        "config": config,
        "palettes": palettes,
    }


def csv_to_jsonl(csv_path: Path, jsonl_path: Path, limit: int = 0) -> int:
    """Convert a flat CSV back to nested JSONL. Returns rows written."""
    written = 0
    with (
        csv_path.open("r", encoding="utf-8") as fin,
        jsonl_path.open("w", encoding="utf-8") as fout,
    ):
        for row in csv.DictReader(fin):
            fout.write(json.dumps(_unflatten_row(row), ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    return written


# ── CLI ──


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Forward: JSONL → CSV
    fwd = sub.add_parser("to-csv", help="JSONL → CSV (default)")
    fwd.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    fwd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    fwd.add_argument("--limit", type=int, default=0)

    # Reverse: CSV → JSONL
    rev = sub.add_parser("to-jsonl", help="CSV → JSONL (reverse)")
    rev.add_argument("--input", type=Path, required=True)
    rev.add_argument("--output", type=Path, required=True)
    rev.add_argument("--limit", type=int, default=0)

    args = parser.parse_args(argv)

    # Default to to-csv when no subcommand given
    if not args.command:
        args.command = "to-csv"
        if not hasattr(args, "input") or args.input is None:
            args.input = DEFAULT_INPUT
        if not hasattr(args, "output") or args.output is None:
            args.output = DEFAULT_OUTPUT
        if not hasattr(args, "limit"):
            args.limit = 0

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    match args.command:
        case "to-csv":
            header = _build_header()
            written = 0
            with (
                input_path.open("r", encoding="utf-8") as fin,
                output_path.open("w", encoding="utf-8", newline="") as fout,
            ):
                writer = csv.writer(fout)
                writer.writerow(header)
                for line in fin:
                    row = json.loads(line)
                    writer.writerow(_flatten_row(row))
                    written += 1
                    if args.limit and written >= args.limit:
                        break
            print(f"Wrote {written} rows → {output_path}")
            cols = len(header)
            print(f"Columns: {cols} ({len(CONFIG_FIELDS)} config + {PALETTE_COUNT * COLORS_PER_PALETTE} palette + 384 embedding + 5 meta)")

        case "to-jsonl":
            written = csv_to_jsonl(input_path, output_path, limit=args.limit)
            print(f"Wrote {written} rows → {output_path}")

        case _:
            parser.print_help()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
