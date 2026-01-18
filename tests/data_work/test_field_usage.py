import importlib
import json
from pathlib import Path

MODULE = importlib.import_module("data_work.09_field_usage")


def test_ablate_field_uses_default() -> None:
    base = MODULE.default_internal_config().model_copy(update={"density": 6})
    ablated = MODULE.ablate_field(base, "density")
    assert ablated.density == MODULE.default_internal_config().density


def test_run_field_usage_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"config_payload": {"tempo": "medium"}}) + "\n", encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    MODULE.run_field_usage(
        input_path=input_path,
        output_dir=output_dir,
        fields=["tempo"],
        limit=1,
        duration=0.1,
        seed=1,
        dry_run=True,
    )
    assert (output_dir / "field_usage.jsonl").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "per_field" in summary
