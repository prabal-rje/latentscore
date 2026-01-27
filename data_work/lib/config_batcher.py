from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

from data_work.lib.music_schema import MusicConfigPromptPayload


@dataclass(slots=True)
class BatchRequest:
    vibe_text: str
    future: asyncio.Future[MusicConfigPromptPayload]


def build_batch_response_keys(batch_size: int) -> list[str]:
    return [f"generated_config_{index}" for index in range(batch_size)]


def build_batch_response_model(batch_size: int) -> type[BaseModel]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    fields: dict[str, tuple[type[MusicConfigPromptPayload], Field]] = {}
    for key in build_batch_response_keys(batch_size):
        fields[key] = (
            MusicConfigPromptPayload,
            Field(..., description="Generated config payload."),
        )
    return create_model(
        f"MusicConfigBatch_{batch_size}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def build_batch_prompt(vibes: Sequence[str], keys: Sequence[str]) -> str:
    header = (
        "Batch keys:\n"
        + ", ".join(keys)
        + "\n\nVibe inputs:"  # keep variable content at end for prompt caching
    )
    lines = [header]
    for index, vibe in enumerate(vibes):
        lines.append(f"<vibe_input index={index}>{vibe}</vibe_input>")
    return "\n".join(lines)


def parse_batch_response(response: BaseModel, batch_size: int) -> list[MusicConfigPromptPayload]:
    data = response.model_dump()
    results: list[MusicConfigPromptPayload] = []
    for index in range(batch_size):
        key = f"generated_config_{index}"
        results.append(MusicConfigPromptPayload.model_validate(data[key]))
    return results


class ConfigBatcher:
    def __init__(
        self,
        *,
        batch_size: int,
        max_wait: float,
        call_batch: Callable[[Sequence[str]], Awaitable[list[MusicConfigPromptPayload]]],
        num_workers: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_wait < 0:
            raise ValueError("max_wait must be >= 0")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._call_batch = call_batch
        self._num_workers = num_workers
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._closed = False
        # Spawn multiple workers for concurrent batch processing
        self._workers = [asyncio.create_task(self._run()) for _ in range(num_workers)]

    async def submit(self, vibe_text: str) -> MusicConfigPromptPayload:
        if self._closed:
            raise RuntimeError("ConfigBatcher is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[MusicConfigPromptPayload] = loop.create_future()
        await self._queue.put(BatchRequest(vibe_text=vibe_text, future=future))
        return await future

    async def aclose(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            seen_vibes: set[str] = {first.vibe_text}
            deferred: list[BatchRequest] = []
            deadline = asyncio.get_running_loop().time() + self._max_wait

            while len(batch) < self._batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    request = await asyncio.wait_for(self._queue.get(), timeout)
                except asyncio.TimeoutError:
                    break

                # Dedupe: if same vibe already in batch, defer to next batch
                if request.vibe_text in seen_vibes:
                    deferred.append(request)
                    continue

                batch.append(request)
                seen_vibes.add(request.vibe_text)

            # Re-queue deferred items for next batch
            for req in deferred:
                await self._queue.put(req)

            vibes = [request.vibe_text for request in batch]
            try:
                results = await self._call_batch(vibes)
                if len(results) != len(batch):
                    raise ValueError("Batch response length mismatch")
                for request, result in zip(batch, results):
                    request.future.set_result(result)
            except Exception as exc:
                for request in batch:
                    request.future.set_exception(exc)
            finally:
                # Only mark processed items as done (not deferred ones)
                for _ in batch:
                    self._queue.task_done()
