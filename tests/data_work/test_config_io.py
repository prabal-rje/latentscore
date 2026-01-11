from pathlib import Path

import pytest

from data_work.lib.config_io import build_config_hash, load_config_file, write_config_file


def test_build_config_hash_stable() -> None:
    config = {"model": "openrouter/openai/gpt-oss-20b", "max_input_tokens": 100000}
    assert build_config_hash(config) == build_config_hash(config)


def test_load_config_file_rejects_api_key(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"api_key": "secret"}', encoding="utf-8")
    with pytest.raises(ValueError, match="API key"):
        load_config_file(path)


def test_write_config_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {"model": "openrouter/openai/gpt-oss-20b", "seed": 7}
    write_config_file(path, payload)
    data = load_config_file(path)
    assert data["model"] == "openrouter/openai/gpt-oss-20b"
    assert data["seed"] == 7
