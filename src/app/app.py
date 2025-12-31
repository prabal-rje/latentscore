from __future__ import annotations

from typing import Iterable

from dependency_injector import containers, providers

from .computation import mean_of


class Container(containers.DeclarativeContainer):
    mean = providers.Callable(mean_of)


async def demo_run(values: Iterable[float] | None = None) -> float:
    container = Container()
    sample = values if values is not None else (1.0, 2.0, 3.0)
    return await container.mean(sample)


async def main() -> None:
    result = await demo_run()
    print(f"Sample mean: {result:.2f}")
