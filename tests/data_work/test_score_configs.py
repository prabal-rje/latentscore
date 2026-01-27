import importlib

from data_work.lib.pipeline_models import ConfigGenerationRow

score_row = importlib.import_module("data_work.02c_score_configs").score_row


def test_score_row_returns_scored_row() -> None:
    """Test that score_row scores all candidates and selects best."""
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
        validation_scores={"format_valid": [1], "schema_valid": [1], "palette_valid": [1]},
        best_index=0,
        config_payload={"thinking": "", "title": "Calm Drift", "config": {}, "palettes": []},
        config_error=None,
    )

    result = score_row(row, {"dummy": lambda vibe, config: {"final_score": 1.0}})

    # Check scores_external for the winner
    assert result.scores_external["dummy"]["final_score"] == 1.0

    # Check candidate_scores has per-candidate scores
    assert "dummy" in result.candidate_scores
    assert result.candidate_scores["dummy"] == [1.0]

    # Check best_index is correctly selected (only 1 candidate)
    assert result.best_index == 0


def test_score_row_selects_best_by_score() -> None:
    """Test that score_row selects the candidate with highest score."""
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
        config_candidates=[
            {"thinking": "a", "title": "Low", "config": {}, "palettes": []},
            {"thinking": "b", "title": "High", "config": {}, "palettes": []},
            {"thinking": "c", "title": "Mid", "config": {}, "palettes": []},
        ],
        validation_scores={
            "format_valid": [1, 1, 1],
            "schema_valid": [1, 1, 1],
            "palette_valid": [1, 1, 1],
        },
        best_index=0,  # 02b picked first valid
        config_payload={"thinking": "a", "title": "Low", "config": {}, "palettes": []},
        config_error=None,
    )

    # Scorer returns different scores for each candidate (based on title)
    def score_by_title(vibe: str, config: dict) -> dict:
        title = config.get("title", "")
        if title == "High":
            return {"final_score": 0.9}
        elif title == "Mid":
            return {"final_score": 0.6}
        else:
            return {"final_score": 0.3}

    result = score_row(row, {"scorer": score_by_title})

    # Should select candidate 1 (title="High") with highest score
    assert result.best_index == 1
    assert result.config_payload is not None
    assert result.config_payload["title"] == "High"

    # Check per-candidate scores
    assert result.candidate_scores["scorer"] == [0.3, 0.9, 0.6]

    # Check detailed scores for winner
    assert result.scores_external["scorer"]["final_score"] == 0.9


def test_score_row_skips_invalid_candidates() -> None:
    """Test that score_row skips candidates that failed validation in 02b."""
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
        config_candidates=[
            {"thinking": "a", "title": "Invalid", "config": {}, "palettes": []},
            {"thinking": "b", "title": "Valid", "config": {}, "palettes": []},
        ],
        validation_scores={
            "format_valid": [1, 1],
            "schema_valid": [0, 1],  # First candidate failed schema validation
            "palette_valid": [1, 1],
        },
        best_index=1,
        config_payload={"thinking": "b", "title": "Valid", "config": {}, "palettes": []},
        config_error=None,
    )

    result = score_row(row, {"scorer": lambda vibe, config: {"final_score": 0.5}})

    # First candidate should have None score (skipped due to schema_valid=0)
    assert result.candidate_scores["scorer"] == [None, 0.5]

    # Should select candidate 1 (the only valid one)
    assert result.best_index == 1
    assert result.config_payload is not None
    assert result.config_payload["title"] == "Valid"
