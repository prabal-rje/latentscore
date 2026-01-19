import pytest
from pydantic import ValidationError

from data_work.lib.pipeline_models import BaseRecord, ConfigGenerationRow, VibeRow


def test_pipeline_models_validate_and_forbid_extra() -> None:
    row = VibeRow(
        dataset="demo",
        id_in_dataset="1",
        split="SFT-Train",
        vibe_index=0,
        text_page=0,
        vibe_scope="scene",
        character_name=None,
        vibe_level="sm",
        vibe_original="calm dusk",
        vibe_noisy="calm dusk",
        tags_original=["calm"],
        tags_noisy=["calm"],
        vibe_model="openrouter/openai/gpt-oss-20b",
    )
    assert row.text_page == 0

    with pytest.raises(ValidationError):
        VibeRow.model_validate({**row.model_dump(), "extra": "nope"})

    config = ConfigGenerationRow.model_validate(
        {
            **row.model_dump(),
            "config_model": "anthropic/claude-opus-4-5-20251101",
            "config_candidates": [None],
            "scores": {"format_valid": [1], "schema_valid": [0], "palette_valid": [0]},
            "best_index": -1,
            "config_payload": None,
            "config_error": "error",
        }
    )
    assert config.config_candidates == [None]

    base = BaseRecord(
        created="2026-01-01",
        metadata={"source": "demo"},
        dataset="demo",
        id_in_dataset="1",
        text="hello",
    )
    assert isinstance(base.metadata, dict)
