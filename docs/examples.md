# Code Samples

## Async-friendly utility
```python
from __future__ import annotations
from typing import Iterable, Callable
import asyncio

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

async def mean_async(values: Iterable[Number]) -> float:
    return await asyncio.to_thread(mean_of, values)
```

## Functional composition
```python
from functools import reduce
from itertools import chain
from typing import Iterable

def flatten_and_sum(seqs: Iterable[Iterable[int]]) -> int:
    flattened = chain.from_iterable(seqs)
    return reduce(lambda acc, x: acc + x, flattened, 0)
```

## Dependency injection with `dependency-injector`
```python
from typing import Protocol
from dependency_injector import containers, providers

class EmailSender(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...

async def notify_user(sender: EmailSender, to: str, subject: str, body: str) -> None:
    if not to:
        return
    await sender.send(to, subject, body)

class ConsoleSender:
    async def send(self, to: str, subject: str, body: str) -> None:
        print(f\"{to=} {subject=} {body=}\")

class Container(containers.DeclarativeContainer):
    email_sender = providers.Singleton(ConsoleSender)
    notify_user = providers.Callable(notify_user, sender=email_sender)
```

## Pytest examples
```python
import pytest

@pytest.mark.asyncio
async def test_notify_user_sends_once():
    fake = FakeSender()
    await notify_user(fake, \"a@example.com\", \"Hi\", \"Body\")
    assert fake.sent == [(\"a@example.com\", \"Hi\", \"Body\")]

@pytest.mark.asyncio
async def test_notify_user_early_exit_on_missing_recipient():
    fake = FakeSender()
    await notify_user(fake, \"\", \"Hi\", \"Body\")
    assert fake.sent == []
```

## Diagnostics log viewer
```bash
python -m latentscore.diagnostics_tui --log-dir "$HOME/Library/Logs/LatentScore"
```
