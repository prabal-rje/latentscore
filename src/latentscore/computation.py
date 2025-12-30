from __future__ import annotations

from typing import Callable, Iterable

Number = float | int


async def mean_of(
    values: Iterable[Number],
    transform: Callable[[Number], Number] | None = None,
) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0

    mapper: Callable[[Number], Number] = transform or (lambda x: x)
    mapped = list(map(mapper, numbers))
    return sum(mapped) / len(mapped)
