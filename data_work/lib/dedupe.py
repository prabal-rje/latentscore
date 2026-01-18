"""Embedding-based deduplication for vibe rows."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Protocol, Sequence, TypeVar

import numpy as np

_LOGGER = logging.getLogger("data_work.dedupe")


class HasText(Protocol):
    """Protocol for objects with a text field."""

    @property
    def text(self) -> str: ...


T = TypeVar("T", bound=HasText)

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


def dedupe_records(
    records: Sequence[T],
    *,
    threshold: float,
    model_name: str,
    embed_fn: EmbedFn | None = None,
) -> tuple[list[T], int]:
    """Dedupe typed records (e.g., BaseRecord) based on their text field.

    This should be called BEFORE splitting to prevent train/test leakage.
    """
    if not records:
        return [], 0
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1.")

    texts = [record.text for record in records]
    embeddings = _resolve_embeddings(texts, model_name=model_name, embed_fn=embed_fn)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.where(norms == 0, 0.0, embeddings / norms)

    kept_records: list[T] = []
    kept_embeddings: list[np.ndarray] = []
    removed = 0
    for idx, record in enumerate(records):
        embedding = normalized[idx]
        if not kept_embeddings:
            kept_records.append(record)
            kept_embeddings.append(embedding)
            continue
        matrix = np.stack(kept_embeddings, axis=0)
        similarities = matrix @ embedding
        if float(similarities.max(initial=-1.0)) >= threshold:
            removed += 1
            continue
        kept_records.append(record)
        kept_embeddings.append(embedding)
    return kept_records, removed


def diversity_sample(
    records: Sequence[T],
    n_samples: int,
    *,
    model_name: str,
    seed: int = 42,
    embed_fn: EmbedFn | None = None,
) -> tuple[list[T], list[int]]:
    """Select n_samples that maximize coverage of embedding space.

    Uses farthest-point sampling (greedy algorithm):
    1. Start with a random seed point
    2. Repeatedly select the point farthest from all already-selected points
    3. This maximizes the minimum distance between selected points

    Why this matters for GRPO:
    - GRPO learns from reward signals on generated completions
    - Diverse prompts → diverse learning signal → better generalization
    - Avoids redundant examples that teach similar things

    Returns:
        Tuple of (selected_records, selected_indices)
    """
    if not records:
        return [], []
    if n_samples <= 0:
        return [], []
    if n_samples >= len(records):
        return list(records), list(range(len(records)))

    texts = [record.text for record in records]
    embeddings = _resolve_embeddings(texts, model_name=model_name, embed_fn=embed_fn)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.where(norms == 0, 0.0, embeddings / norms)

    n = len(records)
    rng = np.random.default_rng(seed)

    # Start with random seed point
    selected_indices: list[int] = [int(rng.integers(n))]

    # Track minimum distance from each point to any selected point
    # Initialize with distance to first selected point
    first_embedding = normalized[selected_indices[0]]
    min_distances = 1.0 - (normalized @ first_embedding)  # cosine distance

    _LOGGER.info(
        "Diversity sampling: selecting %d from %d examples (farthest-point algorithm)",
        n_samples,
        n,
    )

    # Greedy farthest-point selection
    for i in range(1, n_samples):
        # Mark already-selected points as ineligible
        min_distances[selected_indices[-1]] = -np.inf

        # Select point with maximum minimum distance (farthest from all selected)
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)

        # Update minimum distances with distance to newly selected point
        new_embedding = normalized[next_idx]
        new_distances = 1.0 - (normalized @ new_embedding)
        min_distances = np.minimum(min_distances, new_distances)

        if (i + 1) % 100 == 0:
            _LOGGER.debug("Diversity sampling progress: %d/%d", i + 1, n_samples)

    selected_records = [records[i] for i in selected_indices]

    _LOGGER.info(
        "Diversity sampling complete: selected %d examples covering embedding space",
        len(selected_records),
    )

    return selected_records, selected_indices


def dedupe_vibe_rows(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    model_name: str,
    text_field: str = "vibe_original",
    embed_fn: EmbedFn | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Dedupe vibe rows based on text field similarity.

    This dedupes on the vibe content (e.g., "love") rather than raw source text,
    which is more semantically meaningful - two different books about "love"
    should be deduped if their vibes are similar.

    Args:
        rows: List of vibe row dicts
        threshold: Cosine similarity threshold (0.95 = very similar)
        model_name: Sentence-transformers model for embeddings
        text_field: Field to use for deduplication (default: vibe_original)
        embed_fn: Optional custom embedding function

    Returns:
        Tuple of (kept_rows, removed_count)
    """
    if not rows:
        return [], 0
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1.")

    texts = [str(row.get(text_field, "") or "") for row in rows]
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

    _LOGGER.info(
        "Vibe dedupe: kept=%d removed=%d (threshold=%.2f, field=%s)",
        len(kept_rows),
        removed,
        threshold,
        text_field,
    )
    return kept_rows, removed


def diversity_sample_rows(
    rows: list[dict[str, Any]],
    n_samples: int,
    *,
    model_name: str,
    seed: int = 42,
    text_field: str = "vibe_original",
    embed_fn: EmbedFn | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select n_samples vibe rows that maximize coverage of embedding space.

    Uses farthest-point sampling (greedy algorithm) on vibe text content.

    Args:
        rows: List of vibe row dicts
        n_samples: Number of samples to select
        model_name: Sentence-transformers model for embeddings
        seed: Random seed for reproducibility
        text_field: Field to use for embeddings (default: vibe_original)
        embed_fn: Optional custom embedding function

    Returns:
        Tuple of (selected_rows, selected_indices)
    """
    if not rows:
        return [], []
    if n_samples <= 0:
        return [], []
    if n_samples >= len(rows):
        return list(rows), list(range(len(rows)))

    texts = [str(row.get(text_field, "") or "") for row in rows]
    embeddings = _resolve_embeddings(texts, model_name=model_name, embed_fn=embed_fn)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.where(norms == 0, 0.0, embeddings / norms)

    n = len(rows)
    rng = np.random.default_rng(seed)

    # Start with random seed point
    selected_indices: list[int] = [int(rng.integers(n))]

    # Track minimum distance from each point to any selected point
    first_embedding = normalized[selected_indices[0]]
    min_distances = 1.0 - (normalized @ first_embedding)  # cosine distance

    _LOGGER.info(
        "Diversity sampling rows: selecting %d from %d (farthest-point algorithm)",
        n_samples,
        n,
    )

    # Greedy farthest-point selection
    for i in range(1, n_samples):
        min_distances[selected_indices[-1]] = -np.inf
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)

        new_embedding = normalized[next_idx]
        new_distances = 1.0 - (normalized @ new_embedding)
        min_distances = np.minimum(min_distances, new_distances)

        if (i + 1) % 100 == 0:
            _LOGGER.debug("Diversity sampling progress: %d/%d", i + 1, n_samples)

    selected_rows = [rows[i] for i in selected_indices]

    _LOGGER.info(
        "Diversity sampling complete: selected %d rows covering embedding space",
        len(selected_rows),
    )

    return selected_rows, selected_indices
