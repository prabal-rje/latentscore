"""Debug vibe_noisy differences and noise injection."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from data_work.lib.jsonl_io import iter_jsonl

DEFAULT_INPUT = Path("data_work/.processed_smoke2/SFT-Train.jsonl")
DEFAULT_CHAR_AUG_P = 0.15


def _coerce_augmented(value: Any, fallback: str) -> str:
    match value:
        case list() if value:
            return str(value[0])
        case str():
            return value
        case _:
            return fallback


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    vibe_diff = sum(1 for row in rows if row.get("vibe_noisy") != row.get("vibe_original"))
    tags_diff = sum(1 for row in rows if row.get("tags_noisy") != row.get("tags_original"))
    return {"total": total, "vibe_diff": vibe_diff, "tags_diff": tags_diff}


def _print_examples(rows: list[dict[str, Any]], limit: int) -> None:
    shown = 0
    for row in rows:
        original = row.get("vibe_original")
        noisy = row.get("vibe_noisy")
        if original == noisy:
            continue
        print("Example diff:")
        print(f"  original: {original}")
        print(f"  noisy:    {noisy}")
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("No differing vibe_noisy rows found in sample.")


async def _demo_noise(sample: str, error_rate: float, seed: int) -> None:
    try:
        import nlpaug.util as nlpaug_util  # type: ignore[import]
        from nlpaug.augmenter.char import RandomCharAug  # type: ignore[import]
    except ImportError as exc:
        print(f"nlpaug missing: {exc}")
        return

    nlpaug_util.Randomness.seed(seed)
    augmenter = RandomCharAug(action="substitute", aug_char_p=min(DEFAULT_CHAR_AUG_P, error_rate))
    augmented = augmenter.augment(sample)
    noisy = _coerce_augmented(augmented, sample)
    print("Noise demo:")
    print(f"  original: {sample}")
    print(f"  noisy:    {noisy}")
    print(f"  changed:  {noisy != sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect vibe_noisy differences.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--error-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = _load_rows(args.input, args.limit)
    stats = _summarize(rows)
    total = stats["total"]
    print(f"Rows sampled: {total}")
    if total == 0:
        return
    print(
        f"vibe_noisy differences: {stats['vibe_diff']} / {total} ({stats['vibe_diff'] / total:.1%})"
    )
    print(
        f"tags_noisy differences: {stats['tags_diff']} / {total} ({stats['tags_diff'] / total:.1%})"
    )
    _print_examples(rows, args.show)

    sample_text = rows[0].get("vibe_original") or "A warm, gentle atmosphere."
    asyncio.run(_demo_noise(str(sample_text), args.error_rate, args.seed))


if __name__ == "__main__":
    main()
