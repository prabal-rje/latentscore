"""Integration tests for training configuration system."""

import argparse
import json
import tempfile
from pathlib import Path

import pytest


def test_ablation_preset_parsing():
    """Test ablation preset parsing works correctly."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    # Test lora_rank preset
    config = modal_train._apply_ablation_preset("lora_rank:r32")
    assert config.lora.r == 32

    # Test grpo_beta preset
    config = modal_train._apply_ablation_preset("grpo_beta:beta0.02")
    assert config.grpo.beta == 0.02

    # Test learning_rate preset
    config = modal_train._apply_ablation_preset("learning_rate:lr0.0001")
    assert config.optimizer.learning_rate == 1e-4

    # Test batch_size preset
    config = modal_train._apply_ablation_preset("batch_size:bs8")
    assert config.batch.batch_size == 8


def test_ablation_preset_invalid_format():
    """Test ablation preset rejects invalid format."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    with pytest.raises(SystemExit):
        modal_train._apply_ablation_preset("invalid_no_colon")

    with pytest.raises(SystemExit):
        modal_train._apply_ablation_preset("unknown_category:r32")

    with pytest.raises(SystemExit):
        modal_train._apply_ablation_preset("lora_rank:unknown_name")


def test_reward_config_from_args():
    """Test reward config building from CLI args."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    # With all weights specified
    args = argparse.Namespace(format_weight=0.3, schema_weight=0.4, audio_weight=0.3)
    reward_config = modal_train._build_reward_config_from_args(args)
    assert reward_config is not None
    assert reward_config.weights.format_weight == 0.3
    assert reward_config.weights.schema_weight == 0.4
    assert reward_config.weights.audio_weight == 0.3

    # With no weights specified - should return None
    args = argparse.Namespace(format_weight=None, schema_weight=None, audio_weight=None)
    reward_config = modal_train._build_reward_config_from_args(args)
    assert reward_config is None


def test_config_file_loading():
    """Test config file loading works."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    # Create temp config file
    config_data = {"lora": {"r": 64, "alpha": 64}, "epochs": 5, "seed": 123}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = modal_train._load_config_file(config_path)
        assert config.lora.r == 64
        assert config.lora.alpha == 64
        assert config.epochs == 5
        assert config.seed == 123
    finally:
        config_path.unlink()


def test_config_file_not_found():
    """Test config file loading fails gracefully for missing files."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    with pytest.raises(SystemExit):
        modal_train._load_config_file(Path("/nonexistent/config.json"))


def test_config_file_none():
    """Test config file loading returns None when path is None."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    result = modal_train._load_config_file(None)
    assert result is None
