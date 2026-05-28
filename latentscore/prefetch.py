"""Explicit model-asset prefetching with visible progress.

The first call to `ls.render("vibe")` silently downloads the model weights
and the embedding-map JSONL it needs. In a notebook / Colab session this
looks like the kernel is hung for 2-4 minutes. `latentscore.prefetch()`
triggers those same downloads up front so users see progress bars and
aren't surprised. It's a UX nicety — `ls.render(...)` does the same work
on demand.

Usage:

    import latentscore as ls
    ls.prefetch("fast")              # MiniLM + embedding map
    ls.prefetch("fast_heavy")        # LAION-CLAP weights + CLAP embedding map
    ls.prefetch("fast", "fast_heavy")
"""

from __future__ import annotations

from typing import Literal

__all__ = ["prefetch", "PrefetchTarget"]

PrefetchTarget = Literal["fast", "fast_heavy"]

# Module-level idempotency: subsequent calls with the same target are no-ops
# instead of re-instantiating + re-warming the model.
_PREFETCHED: set[str] = set()


def prefetch(*targets: PrefetchTarget) -> None:
    """Download the model assets needed by the given target(s).

    Args:
        *targets: One or more of ``"fast"``, ``"fast_heavy"``. If empty,
            defaults to ``("fast",)``.

    Raises:
        ValueError: if a target is not recognised.
        ModelNotAvailableError: if a required dependency is missing
            (e.g. asking for ``"fast_heavy"`` without ``[heavy]`` installed).
    """
    chosen = targets or ("fast",)
    for target in chosen:
        if target in _PREFETCHED:
            print(f'[latentscore.prefetch] "{target}" already prefetched, skipping.')
            continue
        if target == "fast":
            _prefetch_fast()
        elif target == "fast_heavy":
            _prefetch_fast_heavy()
        else:
            raise ValueError(
                f'Unknown prefetch target {target!r}. Valid targets: "fast", "fast_heavy".'
            )
        _PREFETCHED.add(target)


def _prefetch_fast() -> None:
    print('[latentscore.prefetch] Fetching "fast" model assets...')
    print("  - sentence-transformers/all-MiniLM-L6-v2 (~90 MB)")
    print("  - embedding map JSONL (~100 MB)")

    from .models import FastEmbeddingModel

    FastEmbeddingModel().warmup()
    print('[latentscore.prefetch] "fast" ready.')


def _prefetch_fast_heavy() -> None:
    print('[latentscore.prefetch] Fetching "fast_heavy" model assets...')
    print("  - LAION-CLAP checkpoint (~1.8 GB, one-time)")
    print("  - CLAP embedding map JSONL (~tens of MB)")

    from .models import FastHeavyModel

    FastHeavyModel().warmup()
    print('[latentscore.prefetch] "fast_heavy" ready.')
