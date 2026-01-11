import pytest

from data_work.lib.dedupe import dedupe_rows


def test_dedupe_rows_removes_similar() -> None:
    rows = [
        {"vibe_original": "calm night"},
        {"vibe_original": "calm night"},
        {"vibe_original": "storm"},
    ]

    def embed_fn(texts: list[str]) -> list[list[float]]:
        mapping = {
            "calm night": [1.0, 0.0],
            "storm": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]

    kept, removed = dedupe_rows(rows, threshold=0.99, model_name="dummy", embed_fn=embed_fn)
    assert len(kept) == 2
    assert removed == 1


def test_dedupe_threshold_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        dedupe_rows(
            [{"vibe_original": "x"}], threshold=1.5, model_name="dummy", embed_fn=lambda x: [[1.0]]
        )
