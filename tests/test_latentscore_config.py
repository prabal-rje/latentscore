from __future__ import annotations

import pytest
from pydantic import ValidationError

from latentscore.config import (
    MusicConfig,
    MusicConfigUpdate,
    cadence_to_float,
    chord_change_to_bars,
    chromatic_to_float,
    melody_density_to_float,
    step_bias_to_float,
    swing_to_float,
    syncopation_to_float,
    tempo_to_float,
)
from latentscore.errors import InvalidConfigError


def test_user_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        MusicConfigUpdate.model_validate({"tempo": "medium", "nope": 123})


def test_model_config_preserves_extras() -> None:
    config = MusicConfig.model_validate({"tempo": "medium", "custom": "value"})
    assert config.extras["custom"] == "value"


def test_external_config_accepts_literal_strings() -> None:
    config = MusicConfig(tempo="slow", brightness="bright")
    assert config.tempo == "slow"
    assert config.brightness == "bright"


def test_external_config_converts_to_internal() -> None:
    config = MusicConfig(tempo="fast", brightness="dark")
    internal = config.to_internal()
    assert internal.tempo == pytest.approx(0.7)
    assert internal.brightness == pytest.approx(0.3)


def test_config_rejects_unknown_root() -> None:
    with pytest.raises(ValidationError):
        MusicConfig(root="h")


def test_label_mappings_match_expected_values() -> None:
    assert tempo_to_float("slow") == 0.3
    assert melody_density_to_float("busy") == 0.7
    assert syncopation_to_float("heavy") == 0.8
    assert swing_to_float("light") == 0.2
    assert step_bias_to_float("balanced") == 0.7
    assert chromatic_to_float("medium") == 0.12
    assert cadence_to_float("strong") == 0.9
    assert chord_change_to_bars("very_slow") == 4


def test_label_mappings_reject_unknown() -> None:
    with pytest.raises(InvalidConfigError):
        tempo_to_float("warp")  # type: ignore[arg-type]
