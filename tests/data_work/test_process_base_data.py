import importlib
import json
import sys
import types
from pathlib import Path

from data_work.lib.jsonl_io import iter_jsonl


def _import_process_base_data():
    if "aiosqlite" not in sys.modules:
        sys.modules["aiosqlite"] = types.ModuleType("aiosqlite")
    return importlib.import_module("data_work.02_process_base_data")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parse_model_kwargs_accepts_mapping() -> None:
    module = _import_process_base_data()
    parse = getattr(module, "_parse_model_kwargs")
    assert parse({"foo": "bar"}, label="--model-kwargs") == {"foo": "bar"}


def test_transform_filters_scope_and_level(tmp_path: Path) -> None:
    module = _import_process_base_data()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    rows = [
        {
            "dataset": "demo",
            "id_in_dataset": "1",
            "split": "SFT-Train",
            "vibe_scope": "scene",
            "vibe_level": "xl",
            "vibe_original": "scene xl",
            "vibe_noisy": "scene xl",
            "tags_original": [],
            "tags_noisy": [],
        },
        {
            "dataset": "demo",
            "id_in_dataset": "1",
            "split": "SFT-Train",
            "vibe_scope": "scene",
            "vibe_level": "xs",
            "vibe_original": "scene xs",
            "vibe_noisy": "scene xs",
            "tags_original": [],
            "tags_noisy": [],
        },
        {
            "dataset": "demo",
            "id_in_dataset": "1",
            "split": "SFT-Train",
            "vibe_scope": "character",
            "vibe_level": "xl",
            "vibe_original": "char xl",
            "vibe_noisy": "char xl",
            "tags_original": [],
            "tags_noisy": [],
        },
    ]
    _write_jsonl(source_dir / "SFT-Train.jsonl", rows)

    transform = getattr(module, "transform_processed_splits")
    transform(
        source_dir=source_dir,
        output_dir=output_dir,
        input_dir=None,
        only_splits=["SFT-Train"],
        filter_vibe_scopes=["scene"],
        filter_vibe_levels=["xl"],
        raw_text_direct=False,
        overwrite=True,
    )

    output_rows = list(iter_jsonl(output_dir / "SFT-Train.jsonl"))
    assert len(output_rows) == 1
    assert output_rows[0]["vibe_scope"] == "scene"
    assert output_rows[0]["vibe_level"] == "xl"


def test_transform_raw_text_direct(tmp_path: Path) -> None:
    module = _import_process_base_data()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "input"
    base_rows = [
        {
            "created": "2026-01-01",
            "metadata": {},
            "dataset": "demo",
            "id_in_dataset": "99",
            "text": "raw text payload",
        }
    ]
    _write_jsonl(input_dir / "base.jsonl", base_rows)

    source_rows = [
        {
            "dataset": "demo",
            "id_in_dataset": "99",
            "split": "SFT-Train",
            "vibe_scope": "scene",
            "vibe_level": "sm",
            "vibe_original": "old",
            "vibe_noisy": "old",
            "tags_original": ["x"],
            "tags_noisy": ["x"],
        }
    ]
    _write_jsonl(source_dir / "SFT-Train.jsonl", source_rows)

    transform = getattr(module, "transform_processed_splits")
    transform(
        source_dir=source_dir,
        output_dir=output_dir,
        input_dir=input_dir,
        only_splits=["SFT-Train"],
        filter_vibe_scopes=None,
        filter_vibe_levels=None,
        raw_text_direct=True,
        overwrite=True,
    )

    output_rows = list(iter_jsonl(output_dir / "SFT-Train.jsonl"))
    assert len(output_rows) == 1
    assert output_rows[0]["vibe_scope"] == "raw_text"
    assert output_rows[0]["vibe_original"] == "raw text payload"
    assert output_rows[0]["vibe_noisy"] == "raw text payload"
