from __future__ import annotations

import pytest
from pydantic import ValidationError

from latentscore.config import (
    MusicConfig,
    MusicConfigUpdate,
    Step,
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


# -----------------------------------------------------------------------------
# Step tests
# -----------------------------------------------------------------------------


class TestStep:
    """Tests for Step-based relative updates in MusicConfigUpdate."""

    def test_step_in_update_accepted_for_steppable_field(self) -> None:
        update = MusicConfigUpdate(brightness=Step(+1))
        assert isinstance(update.brightness, Step)
        assert update.brightness.delta == 1

    def test_step_rejected_for_non_steppable_field(self) -> None:
        with pytest.raises(ValidationError):
            MusicConfigUpdate(bass=Step(+1))  # type: ignore[arg-type]

    def test_apply_to_with_absolute_value(self) -> None:
        base = MusicConfig(tempo="slow")
        update = MusicConfigUpdate(tempo="fast")
        result = update.apply_to(base)
        assert result.tempo == "fast"

    def test_apply_to_with_step_up(self) -> None:
        base = MusicConfig(brightness="medium")
        update = MusicConfigUpdate(brightness=Step(+1))
        result = update.apply_to(base)
        assert result.brightness == "bright"

    def test_apply_to_with_step_down(self) -> None:
        base = MusicConfig(brightness="medium")
        update = MusicConfigUpdate(brightness=Step(-1))
        result = update.apply_to(base)
        assert result.brightness == "dark"

    def test_apply_to_with_mixed_absolute_and_step(self) -> None:
        base = MusicConfig(brightness="medium", tempo="slow", density=4)
        update = MusicConfigUpdate(
            brightness=Step(+1),
            tempo="fast",
            density=Step(+2),
        )
        result = update.apply_to(base)
        assert result.brightness == "bright"
        assert result.tempo == "fast"
        assert result.density == 6

    def test_apply_to_saturates_at_max(self) -> None:
        base = MusicConfig(brightness="medium")
        update = MusicConfigUpdate(brightness=Step(+10))
        result = update.apply_to(base)
        assert result.brightness == "very_bright"

    def test_apply_to_saturates_at_min(self) -> None:
        base = MusicConfig(tempo="slow")
        update = MusicConfigUpdate(tempo=Step(-10))
        result = update.apply_to(base)
        assert result.tempo == "very_slow"

    def test_apply_to_preserves_unmodified_fields(self) -> None:
        base = MusicConfig(brightness="medium", tempo="slow", bass="drone")
        update = MusicConfigUpdate(brightness=Step(+1))
        result = update.apply_to(base)
        assert result.tempo == "slow"
        assert result.bass == "drone"

    def test_to_internal_rejects_unresolved_step(self) -> None:
        update = MusicConfigUpdate(brightness=Step(+1))
        with pytest.raises(ValueError, match="unresolved Step"):
            update.to_internal()

    def test_step_repr(self) -> None:
        assert repr(Step(+1)) == "Step(+1)"
        assert repr(Step(-2)) == "Step(-2)"
        assert repr(Step(0)) == "Step(+0)"
