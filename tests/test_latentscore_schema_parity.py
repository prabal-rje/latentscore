from __future__ import annotations

from latentscore import config as cfg
from latentscore import synth_ANOTHER as another


def _field_names(model: object) -> set[str]:
    if hasattr(model, "model_fields"):
        return set(getattr(model, "model_fields").keys())
    if hasattr(model, "__fields__"):
        return set(getattr(model, "__fields__").keys())
    if hasattr(model, "__annotations__"):
        return set(getattr(model, "__annotations__").keys())
    raise AssertionError(f"Unsupported model type: {model!r}")


def test_music_config_matches_another_schema() -> None:
    missing = _field_names(another.MusicConfig) - _field_names(cfg.MusicConfig)
    assert not missing, f"Missing fields: {sorted(missing)}"
