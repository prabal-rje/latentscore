"""Embedding-based deduplication for vibe rows."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Sequence

import numpy as np

_LOGGER = logging.getLogger("data_work.dedupe")

EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]


@functools.lru_cache(maxsize=2)
def _load_embedder(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("sentence-transformers not installed: %s", exc)
        raise SystemExit(
            "sentence-transformers is required for dedupe. Install via data_work/requirements.txt."
        ) from exc
    return SentenceTransformer(model_name)


def _embed_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    embedder = _load_embedder(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def _resolve_embeddings(
    texts: list[str],
    *,
    model_name: str,
    embed_fn: EmbedFn | None,
) -> np.ndarray:
    match embed_fn:
        case None:
            return _embed_texts(texts, model_name=model_name)
        case _:
            return np.asarray(embed_fn(texts), dtype="float32")


def dedupe_rows(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    model_name: str,
    embed_fn: EmbedFn | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1.")

    texts: list[str] = []
    for row in rows:
        value = row.get("vibe_original", "")
        text = str(value or "")
        texts.append(text)

    embeddings = _resolve_embeddings(texts, model_name=model_name, embed_fn=embed_fn)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.where(norms == 0, 0.0, embeddings / norms)

    kept_rows: list[dict[str, Any]] = []
    kept_embeddings: list[np.ndarray] = []
    removed = 0
    for idx, row in enumerate(rows):
        embedding = normalized[idx]
        if not kept_embeddings:
            kept_rows.append(row)
            kept_embeddings.append(embedding)
            continue
        matrix = np.stack(kept_embeddings, axis=0)
        similarities = matrix @ embedding
        if float(similarities.max(initial=-1.0)) >= threshold:
            removed += 1
            continue
        kept_rows.append(row)
        kept_embeddings.append(embedding)
    return kept_rows, removed
