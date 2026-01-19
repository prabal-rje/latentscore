from data_work.lib.eval_schema import EvalResult, compute_field_distributions


def test_field_distributions_exclude_title() -> None:
    result = EvalResult(
        prompt_id="p1",
        source_label="test",
        config={"thinking": "ok", "title": "Soft Dawn", "tempo": "slow", "palettes": []},
        json_valid=True,
        schema_valid=True,
    )
    distributions = compute_field_distributions([result])
    assert "title" not in distributions
