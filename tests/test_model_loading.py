from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from latentscore.config import MusicConfig
from latentscore.models import FastEmbeddingModel, FastHeavyModel, _load_clap_weights

_EMBEDDINGS = [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]


def _write_map(path: Path, column: str) -> None:
    rows = [
        {
            "vibe_original": f"vibe {i}",
            "config": MusicConfig().model_dump(),
            "title": f"Title {i}",
            "palettes": [],
            column: embedding,
        }
        for i, embedding in enumerate(_EMBEDDINGS)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _expected_matrix() -> np.ndarray:
    mat = np.asarray(_EMBEDDINGS, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def test_embed_maps_give_normalized_float32_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_map(tmp_path / "minilm.jsonl", "embedding")
    _write_map(tmp_path / "clap.jsonl", "clap_audio_embedding")
    monkeypatch.setenv("LATENTSCORE_EMBED_MAP", str(tmp_path / "minilm.jsonl"))
    monkeypatch.setenv("LATENTSCORE_CLAP_EMBED_MAP", str(tmp_path / "clap.jsonl"))

    for examples, matrix in (
        FastEmbeddingModel()._embed_map_examples(),
        FastHeavyModel()._clap_embed_map_examples(),
    ):
        assert matrix is not None and matrix.dtype == np.float32
        np.testing.assert_array_equal(matrix, _expected_matrix())
        assert [example.title for example in examples] == ["Title 0", "Title 1", "Title 2"]


class _Wrapper:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(3, 2)


def test_clap_weights_load_without_optimizer_state(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save(
        {
            "epoch": 15,
            "name": "run",
            "optimizer": {
                "state": {0: {"exp_avg": torch.ones(2, 3)}},
                "param_groups": [{"lr": np.float64(1e-4)}],
            },
            "state_dict": {
                "module.weight": source.weight.detach().clone(),
                "module.bias": source.bias.detach().clone(),
                "module.text_branch.embeddings.position_ids": torch.arange(3),
            },
        },
        checkpoint,
    )
    target = _Wrapper()

    _load_clap_weights(target, checkpoint)

    assert torch.equal(target.model.weight, source.weight)
    assert torch.equal(target.model.bias, source.bias)


class _FakeClap:
    def __init__(self) -> None:
        self.inference_mode: list[bool] = []

    def get_text_embedding(self, texts: list[str]) -> Any:
        self.inference_mode.append(torch.is_inference_mode_enabled())
        return np.ones((len(texts), 3), dtype=np.float32)


def test_fast_heavy_text_embedding_runs_without_autograd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_map(tmp_path / "clap.jsonl", "clap_audio_embedding")
    monkeypatch.setenv("LATENTSCORE_CLAP_EMBED_MAP", str(tmp_path / "clap.jsonl"))
    clap = _FakeClap()
    model = FastHeavyModel()
    monkeypatch.setattr(model, "_load_clap", lambda: clap)

    result = model._embed_and_select("rain")

    assert clap.inference_mode == [True]
    assert result.title == "Title 2"
