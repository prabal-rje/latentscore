"""Process base data into train/test/eval splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, ValidationError

REQUIRED_FIELDS = ("created", "metadata", "dataset", "id_in_dataset", "text")


class BaseRecord(BaseModel):
    created: str
    metadata: dict[str, Any]
    dataset: str
    id_in_dataset: Any = Field(alias="id_in_dataset")
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate base dataset files before processing into train/test/eval splits. "
            "Processing is not yet implemented."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing base JSONL files created by download_base_data.py.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def validate_schema(paths: Iterable[Path]) -> None:
    for path in paths:
        for row in iter_jsonl(path):
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing required fields {missing} in {path}.")
            try:
                BaseRecord.model_validate(row)
            except ValidationError as exc:
                raise ValueError(f"Invalid record in {path}: {exc}") from exc


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    validate_schema(jsonl_paths)
    raise NotImplementedError("Train/test/eval splitting is not implemented yet.")


if __name__ == "__main__":
    main()
