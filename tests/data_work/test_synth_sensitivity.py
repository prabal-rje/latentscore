import importlib
import json
from pathlib import Path

MODULE = importlib.import_module("data_work.08_synth_sensitivity")


def test_build_variant_clamps_float() -> None:
    base = MODULE.default_internal_config()
    variant = MODULE.build_variant(base, "tempo", 10.0, direction=1)
    assert 0.0 <= variant.tempo <= 1.0


def test_extract_config_payload_prefers_config_key() -> None:
    row = {"config_payload": {"config": {"tempo": "medium"}}}
    payload = MODULE.extract_config_payload(row)
    assert payload is not None
    assert payload["tempo"] == "medium"


def test_run_sensitivity_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"config_payload": {"tempo": "medium"}}) + "\n", encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    MODULE.run_sensitivity(
        input_path=input_path,
        output_dir=output_dir,
        fields=["tempo"],
        delta=0.2,
        limit=1,
        duration=0.1,
        seed=1,
        dry_run=True,
    )
    assert (output_dir / "synth_sensitivity.jsonl").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "per_field" in summary
