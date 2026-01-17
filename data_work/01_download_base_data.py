"""Download and sample Common Pile datasets for data work."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, get_dataset_config_names, get_dataset_infos, load_dataset
from pydantic import BaseModel

DATASETS: tuple[str, ...] = (
    "common-pile/news_filtered",
    "common-pile/pressbooks_filtered",
    "common-pile/project_gutenberg_filtered",
)
DEFAULT_SAMPLE_SIZE = 1_000
TEXT_FIELD = "text"
DATASET_FIELD = "dataset"
ID_FIELD = "id_in_dataset"
CREATED_FIELD = "created"
METADATA_FIELD = "metadata"


class DatasetPlan(BaseModel):
    name: str
    config: str | None
    display_name: str
    size_bytes: int | None

    @property
    def output_stem(self) -> str:
        if self.config:
            return f"{self.name.replace('/', '__')}__{self.config}"
        return self.name.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Common Pile datasets, sample 1,000 texts each, and write JSONL files "
            "with standardized fields for downstream processing."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic dataset table selection and sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Relative or absolute directory where sampled JSONL files will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of texts to sample per dataset (must be >= 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset files in the output directory.",
    )
    return parser.parse_args()


def format_bytes(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def choose_project_gutenberg_config(seed: int) -> str:
    configs = get_dataset_config_names("common-pile/project_gutenberg_filtered")
    if not configs:
        raise ValueError("No configs found for common-pile/project_gutenberg_filtered")
    rng = random.Random(seed)
    return rng.choice(configs)


def collect_plan(seed: int) -> list[DatasetPlan]:
    gutenberg_config = choose_project_gutenberg_config(seed)
    plans: list[DatasetPlan] = []
    for dataset_name in DATASETS:
        config = gutenberg_config if dataset_name.endswith("project_gutenberg_filtered") else None
        infos = get_dataset_infos(dataset_name)
        info = infos.get(config) if config else infos.get("default") or next(iter(infos.values()))
        size_bytes = info.download_size if info else None
        display_name = dataset_name if config is None else f"{dataset_name}/{config}"
        plans.append(
            DatasetPlan(
                name=dataset_name,
                config=config,
                display_name=display_name,
                size_bytes=size_bytes,
            )
        )
    return plans


def confirm_download(plans: Iterable[DatasetPlan]) -> None:
    print("This script will download the following datasets from Common Pile:")
    for plan in plans:
        size_label = format_bytes(plan.size_bytes)
        print(f"- {plan.display_name} (approx download size: {size_label})")
    response = input("Proceed with download? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        raise SystemExit("Download cancelled by user.")


def validate_output_dir(output_dir: Path, plans: Iterable[DatasetPlan], overwrite: bool) -> None:
    existing = [plan for plan in plans if (output_dir / f"{plan.output_stem}.jsonl").exists()]
    if existing and not overwrite:
        names = ", ".join(plan.output_stem for plan in existing)
        raise SystemExit(f"Output files already exist ({names}). Use --overwrite to replace them.")


def sample_dataset(
    dataset: Dataset,
    seed: int,
    dataset_label: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    if TEXT_FIELD not in dataset.column_names:
        raise ValueError(f"Expected '{TEXT_FIELD}' column in {dataset_label}")
    if len(dataset) < sample_size:
        raise ValueError(f"Dataset {dataset_label} has only {len(dataset)} rows.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), sample_size)
    selected = dataset.select(indices)

    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for idx, row in zip(indices, selected):
        text_value = row.get(TEXT_FIELD)
        match text_value:
            case str():
                pass
            case _:
                raise ValueError(f"Row text is not a string in {dataset_label} (index {idx}).")
        row_metadata = {key: value for key, value in row.items() if key != TEXT_FIELD}
        row_id = row_metadata.get("id", idx)
        rows.append(
            {
                CREATED_FIELD: created,
                METADATA_FIELD: row_metadata,
                DATASET_FIELD: dataset_label,
                ID_FIELD: row_id,
                TEXT_FIELD: text_value,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1.")

    plans = collect_plan(args.seed)
    validate_output_dir(output_dir, plans, args.overwrite)
    confirm_download(plans)

    for plan in plans:
        dataset_label = plan.display_name
        dataset = load_dataset(plan.name, plan.config, split="train")
        rows = sample_dataset(dataset, args.seed, dataset_label, args.sample_size)
        output_path = output_dir / f"{plan.output_stem}.jsonl"
        write_jsonl(output_path, rows)
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
