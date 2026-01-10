from __future__ import annotations


def _field_names(model: object) -> set[str]:
    if hasattr(model, "model_fields"):
        return set(getattr(model, "model_fields").keys())
    if hasattr(model, "__fields__"):
        return set(getattr(model, "__fields__").keys())
    if hasattr(model, "__annotations__"):
        return set(getattr(model, "__annotations__").keys())
    raise AssertionError(f"Unsupported model type: {model!r}")
