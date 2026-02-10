"""Export vibe embeddings + config payloads for fast lookup."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from data_work.lib.jsonl_io import iter_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data_work/2026-01-26_scored/_progress.jsonl")
DEFAULT_OUTPUT = Path("data_work/2026-01-26_scored/vibe_and_embeddings_to_config_map.jsonl")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


def _load_embedder(model_name: str | None) -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    if model_name:
        return SentenceTransformer(model_name)
    return SentenceTransformer(DEFAULT_MODEL)


def _iter_rows(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    count = 0
    for row in iter_jsonl(path):
        yield row
        count += 1
        if limit and count >= limit:
            break


def _batch(iterable: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in iterable:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _coerce_float_list(vec: np.ndarray) -> list[float]:
    return [float(x) for x in vec.astype(np.float32).tolist()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export vibe embeddings + config payloads for fast lookup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit rows (0 = all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )

    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output file exists: {output_path} (use --overwrite)")

    embedder = _load_embedder(args.model)
    LOGGER.info("Embedding model loaded: %s", args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for batch in _batch(_iter_rows(input_path, args.limit), args.batch_size):
            vibes: list[str] = []
            rows: list[dict[str, Any]] = []
            for row in batch:
                vibe = row.get("vibe_original")
                payload = row.get("config_payload")
                if not vibe or not isinstance(payload, dict):
                    continue
                vibes.append(vibe)
                rows.append(row)
            total += len(batch)
            if not rows:
                continue

            embeddings = embedder.encode(vibes, normalize_embeddings=True)
            vectors = np.asarray(embeddings, dtype=np.float32)

            for row, vec in zip(rows, vectors, strict=False):
                payload = row.get("config_payload") or {}
                record = {
                    "dataset": row.get("dataset"),
                    "id_in_dataset": row.get("id_in_dataset"),
                    "split": row.get("split"),
                    "vibe_original": row.get("vibe_original"),
                    "embedding": _coerce_float_list(vec),
                    "title": payload.get("title"),
                    "config": payload.get("config"),
                    "palettes": payload.get("palettes"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    LOGGER.info("Processed rows: %s", total)
    LOGGER.info("Wrote rows: %s", written)
    LOGGER.info("Output: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
