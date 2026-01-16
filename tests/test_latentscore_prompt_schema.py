from __future__ import annotations

import json
from typing import Any, Literal, get_origin

from pydantic import BaseModel

from data_work.lib import music_schema as data_schema
from latentscore import config as ls_config
from latentscore.config import MusicConfig, MusicConfigPrompt, MusicConfigPromptPayload


def _field_names(model: type[BaseModel]) -> set[str]:
    return set(model.model_fields)


def _has_defaults(schema: Any) -> bool:
    if isinstance(schema, dict):
        if "default" in schema:
            return True
        return any(_has_defaults(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_has_defaults(value) for value in schema)
    return False


def _is_literal(annotation: object) -> bool:
    return get_origin(annotation) is Literal


def _schema_signature(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), sort_keys=True)


def _assert_field_desc_match(left: type[BaseModel], right: type[BaseModel]) -> None:
    assert set(left.model_fields) == set(right.model_fields)
    for name in left.model_fields:
        assert left.model_fields[name].description == right.model_fields[name].description


def test_prompt_schema_matches_music_config_fields() -> None:
    excluded = {"schema_version", "seed"}
    assert _field_names(MusicConfigPrompt) == _field_names(MusicConfig) - excluded


def test_prompt_schema_excludes_seed_and_version() -> None:
    assert "schema_version" not in _field_names(MusicConfigPrompt)
    assert "seed" not in _field_names(MusicConfigPrompt)


def test_prompt_schema_has_no_defaults() -> None:
    schema = MusicConfigPromptPayload.model_json_schema()
    assert not _has_defaults(schema)


def test_prompt_schema_uses_literal_labels_for_scalar_fields() -> None:
    literal_fields = {
        "phrase_len_bars",
        "melody_density",
        "syncopation",
        "swing",
        "motif_repeat_prob",
        "step_bias",
        "chromatic_prob",
        "cadence_strength",
        "chord_change_bars",
    }
    for field_name in literal_fields:
        annotation = MusicConfigPrompt.model_fields[field_name].annotation
        assert _is_literal(annotation)


def test_prompt_payload_schema_matches_data_work() -> None:
    """Ensure latentscore and data_work schemas are in sync."""
    _assert_field_desc_match(ls_config.MusicConfigPrompt, data_schema.MusicConfigPrompt)
    _assert_field_desc_match(ls_config.PaletteColor, data_schema.PaletteColor)
    _assert_field_desc_match(ls_config.Palette, data_schema.Palette)
    _assert_field_desc_match(
        ls_config.MusicConfigPromptPayload, data_schema.MusicConfigPromptPayload
    )
    assert _schema_signature(ls_config.MusicConfigPromptPayload) == _schema_signature(
        data_schema.MusicConfigPromptPayload
    )


def test_prompt_payload_includes_palettes() -> None:
    """Ensure MusicConfigPromptPayload has palettes field."""
    assert "palettes" in MusicConfigPromptPayload.model_fields
    field = MusicConfigPromptPayload.model_fields["palettes"]
    assert field.is_required()
