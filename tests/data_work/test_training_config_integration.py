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
    args = argparse.Namespace(
        format_weight=0.3,
        schema_weight=0.4,
        audio_weight=0.3,
        title_similarity_weight=0.15,
        title_length_penalty_weight=0.05,
    )
    reward_config = modal_train._build_reward_config_from_args(args)
    assert reward_config is not None
    assert reward_config.weights.format_weight == 0.3
    assert reward_config.weights.schema_weight == 0.4
    assert reward_config.weights.audio_weight == 0.3
    assert reward_config.weights.title_similarity_weight == 0.15
    assert reward_config.weights.title_length_penalty_weight == 0.05

    # With no weights specified - should return None
    args = argparse.Namespace(
        format_weight=None,
        schema_weight=None,
        audio_weight=None,
        title_similarity_weight=None,
        title_length_penalty_weight=None,
    )
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


def test_default_max_seq_length_is_4096():
    """Default max sequence length should be large enough for training samples."""
    from common.training_config import TrainingConfig

    config = TrainingConfig()
    assert config.data.max_seq_length == 4096


def test_default_grpo_max_completion_length_is_3072() -> None:
    """Default GRPO completion length should be ~3k tokens."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")
    args = modal_train.parse_args(
        [
            "grpo",
            "--data",
            "data_work/.tmp_grpo_one.jsonl",
            "--output",
            "exp-grpo-default-length",
            "--model",
            "unsloth/gemma-3-270m-it",
            "--epochs",
            "1",
        ]
    )
    assert args.max_completion_length == 3072


def test_resolve_model_path_defaults_to_outputs():
    """Bare model names should resolve to the Modal outputs volume."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    assert modal_train._resolve_model_path("exp-sft-baseline") == "/outputs/exp-sft-baseline"
    assert (
        modal_train._resolve_model_path("/outputs/exp-sft-baseline") == "/outputs/exp-sft-baseline"
    )
    assert (
        modal_train._resolve_model_path("outputs/exp-sft-baseline") == "/outputs/exp-sft-baseline"
    )
    assert modal_train._resolve_model_path("org/model") == "org/model"


def test_base_model_registry_contains_only_gemma_qwen():
    """TrainingConfig should only advertise Gemma3 and Qwen3 aliases."""
    from common.training_config import BASE_MODELS

    assert set(BASE_MODELS.keys()) == {"gemma3-270m", "qwen3-600m"}


def test_default_base_model_is_gemma3():
    """Default training config should use Gemma3 as the base model."""
    from common.training_config import TrainingConfig

    config = TrainingConfig()
    assert config.base_model == "gemma3-270m"


def test_modal_base_models_only_gemma_qwen():
    """Modal training aliases should match Gemma3/Qwen3 only."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")
    assert set(modal_train.BASE_MODELS.keys()) == {"gemma3-270m", "qwen3-600m"}


def test_base_model_alias_qwen3_600m():
    """Qwen3-600m alias should point to the correct HF repo."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    assert modal_train.BASE_MODELS["qwen3-600m"] == "unsloth/Qwen3-0.6B"


def test_training_config_resolves_qwen3_600m():
    """TrainingConfig should resolve qwen3-600m to the correct HF repo."""
    from common.training_config import TrainingConfig

    config = TrainingConfig(base_model="qwen3-600m")
    assert config.resolve_base_model() == "unsloth/Qwen3-0.6B"


class _DummyTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [text.split() for text in texts]}


def test_estimate_max_token_length():
    """Estimator should return the longest tokenized length."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")
    tokenizer = _DummyTokenizer()
    texts = ["short", "two words here", "a b c d e"]

    assert modal_train._estimate_max_token_length(texts, tokenizer, batch_size=2) == 5


def test_validate_max_seq_length_raises_for_short_limit():
    """Guard should raise when configured max is too small."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    with pytest.raises(SystemExit):
        modal_train._validate_max_seq_length(max_seq_length=3, observed_max_length=4)


def test_validate_max_seq_length_allows_equal_or_longer():
    """Guard should allow max lengths that meet or exceed observed lengths."""
    import importlib

    modal_train = importlib.import_module("data_work.03_modal_train")

    modal_train._validate_max_seq_length(max_seq_length=4, observed_max_length=4)
    modal_train._validate_max_seq_length(max_seq_length=5, observed_max_length=4)
