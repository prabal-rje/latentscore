import importlib

from data_work.lib.pipeline_models import ConfigGenerationRow

score_row = importlib.import_module("data_work.02c_score_configs").score_row


def test_score_row_returns_scored_row() -> None:
    row = ConfigGenerationRow(
        dataset="demo",
        id_in_dataset="1",
        split="SFT-Train",
        vibe_index=0,
        text_page=0,
        vibe_scope="scene",
        character_name=None,
        vibe_level="xl",
        vibe_original="calm",
        vibe_noisy="calm",
        tags_original=[],
        tags_noisy=[],
        vibe_model="m",
        config_model="m",
        config_candidates=[{"thinking": "", "title": "Calm Drift", "config": {}, "palettes": []}],
        scores={"format_valid": [1], "schema_valid": [1], "palette_valid": [1]},
        best_index=0,
        config_payload={"thinking": "", "title": "Calm Drift", "config": {}, "palettes": []},
        config_error=None,
    )

    result = score_row(row, {"dummy": lambda vibe, config: {"final_score": 1.0}})
    assert result.scores_external["dummy"]["final_score"] == 1.0
