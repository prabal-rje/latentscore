"""Configuration IO and hashing for data_work pipelines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_BANNED_KEYS = {"api_key", "api_key_env"}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(config[key]) for key in sorted(config)}


def build_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        normalize_config(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    banned = _BANNED_KEYS.intersection(data)
    if banned:
        keys = ", ".join(sorted(banned))
        raise ValueError(f"Config file cannot contain API key fields: {keys}")
    return data


def write_config_file(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalize_config(config), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
