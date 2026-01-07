from __future__ import annotations

import pytest
from pydantic import ValidationError

from latentscore.config import MusicConfig, MusicConfigUpdate


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
