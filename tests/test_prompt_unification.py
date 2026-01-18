from common import build_config_generation_prompt, build_vibe_extraction_prompt
from common.prompt_registry import render_config_prompt, render_prompt
from data_work.lib.vibe_schema import schema_signature as vibe_schema_signature


def test_prompt_registry_matches_config_prompt() -> None:
    assert render_config_prompt() == build_config_generation_prompt()


def test_prompt_registry_matches_vibe_prompt() -> None:
    schema = vibe_schema_signature()
    assert render_prompt("vibe_v1", schema=schema) == build_vibe_extraction_prompt(schema)
