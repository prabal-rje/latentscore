import pytest

from data_work.lib.dedupe import dedupe_rows
from data_work.lib.pipeline_models import VibeRow


def test_dedupe_rows_removes_similar() -> None:
    rows = [
        VibeRow(
            dataset="demo",
            id_in_dataset="1",
            split="SFT-Train",
            vibe_index=0,
            text_page=0,
            vibe_scope="scene",
            character_name=None,
            vibe_level="xl",
            vibe_original="calm night",
            vibe_noisy="calm night",
            tags_original=[],
            tags_noisy=[],
            vibe_model="m",
        ),
        VibeRow(
            dataset="demo",
            id_in_dataset="1",
            split="SFT-Train",
            vibe_index=1,
            text_page=0,
            vibe_scope="scene",
            character_name=None,
            vibe_level="xl",
            vibe_original="calm night",
            vibe_noisy="calm night",
            tags_original=[],
            tags_noisy=[],
            vibe_model="m",
        ),
        VibeRow(
            dataset="demo",
            id_in_dataset="1",
            split="SFT-Train",
            vibe_index=2,
            text_page=0,
            vibe_scope="scene",
            character_name=None,
            vibe_level="xl",
            vibe_original="storm",
            vibe_noisy="storm",
            tags_original=[],
            tags_noisy=[],
            vibe_model="m",
        ),
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
            [
                VibeRow(
                    dataset="demo",
                    id_in_dataset="1",
                    split="SFT-Train",
                    vibe_index=0,
                    text_page=0,
                    vibe_scope="scene",
                    character_name=None,
                    vibe_level="xl",
                    vibe_original="x",
                    vibe_noisy="x",
                    tags_original=[],
                    tags_noisy=[],
                    vibe_model="m",
                )
            ],
            threshold=1.5,
            model_name="dummy",
            embed_fn=lambda x: [[1.0]],
        )


def test_dedupe_rows_accepts_viberow() -> None:
    rows = [
        VibeRow(
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
            vibe_model="model",
        )
    ]
    kept, removed = dedupe_rows(
        rows, threshold=0.99, model_name="dummy", embed_fn=lambda x: [[1.0]]
    )
    assert len(kept) == 1
    assert removed == 0
