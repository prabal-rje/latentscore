"""Async periodic writer for incremental JSONL output.

Provides human-readable progress by writing partial results periodically,
not just at the end of processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class AsyncPeriodicWriter:
    """Async writer that periodically flushes buffered rows to a JSONL file.

    Usage:
        writer = AsyncPeriodicWriter(path, interval_seconds=60)
        await writer.start()

        # Add rows as they complete
        await writer.add_row({"field": "value"})

        # When done, stop and flush remaining
        await writer.stop()

    The writer flushes to disk every `interval_seconds` and on stop().
    This provides human-readable progress during long-running jobs.
    """

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float = 60.0,
        overwrite: bool = False,
    ) -> None:
        """Initialize the periodic writer.

        Args:
            path: Output JSONL file path.
            interval_seconds: How often to flush to disk (default 60s).
            overwrite: If True, truncate file on start. If False, append.
        """
        self._path = path
        self._interval = interval_seconds
        self._overwrite = overwrite
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._total_written = 0

    async def start(self) -> None:
        """Start the periodic flush task."""
        if self._running:
            return

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate file if overwrite mode
        if self._overwrite and self._path.exists():
            self._path.unlink()

        self._running = True
        self._task = asyncio.create_task(self._periodic_flush_loop())
        _LOGGER.debug("Started periodic writer for %s (interval=%.1fs)", self._path, self._interval)

    async def stop(self) -> None:
        """Stop the periodic flush task and flush remaining buffer."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Final flush
        await self._flush()
        _LOGGER.debug("Stopped periodic writer for %s (total written: %d)", self._path, self._total_written)

    async def add_row(self, row: dict[str, Any]) -> None:
        """Add a row to the buffer (will be flushed periodically)."""
        async with self._lock:
            self._buffer.append(row)

    async def add_rows(self, rows: list[dict[str, Any]]) -> None:
        """Add multiple rows to the buffer."""
        async with self._lock:
            self._buffer.extend(rows)

    @property
    def total_written(self) -> int:
        """Return total rows written to disk so far."""
        return self._total_written

    @property
    def buffered_count(self) -> int:
        """Return count of rows in buffer (not yet written)."""
        return len(self._buffer)

    async def _periodic_flush_loop(self) -> None:
        """Background task that flushes buffer periodically."""
        while self._running:
            await asyncio.sleep(self._interval)
            if self._running:  # Check again after sleep
                await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        async with self._lock:
            if not self._buffer:
                return

            rows_to_write = self._buffer
            self._buffer = []

        # Write outside the lock to minimize lock hold time
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for row in rows_to_write:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

            self._total_written += len(rows_to_write)
            _LOGGER.info(
                "Periodic flush: wrote %d rows to %s (total: %d)",
                len(rows_to_write),
                self._path.name,
                self._total_written,
            )
        except Exception as exc:
            # Re-add rows to buffer on failure
            async with self._lock:
                self._buffer = rows_to_write + self._buffer
            _LOGGER.error("Failed to flush to %s: %s", self._path, exc)
            raise


class SyncPeriodicWriter:
    """Synchronous version for non-async contexts.

    Uses a background thread for periodic flushing.
    """

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float = 60.0,
        overwrite: bool = False,
    ) -> None:
        """Initialize the periodic writer.

        Args:
            path: Output JSONL file path.
            interval_seconds: How often to flush to disk (default 60s).
            overwrite: If True, truncate file on start. If False, append.
        """
        import threading

        self._path = path
        self._interval = interval_seconds
        self._overwrite = overwrite
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._total_written = 0

    def start(self) -> None:
        """Start the periodic flush thread."""
        import threading

        if self._thread is not None:
            return

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate file if overwrite mode
        if self._overwrite and self._path.exists():
            self._path.unlink()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._periodic_flush_loop, daemon=True)
        self._thread.start()
        _LOGGER.debug("Started periodic writer for %s (interval=%.1fs)", self._path, self._interval)

    def stop(self) -> None:
        """Stop the periodic flush thread and flush remaining buffer."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None

        # Final flush
        self._flush()
        _LOGGER.debug("Stopped periodic writer for %s (total written: %d)", self._path, self._total_written)

    def add_row(self, row: dict[str, Any]) -> None:
        """Add a row to the buffer (will be flushed periodically)."""
        with self._lock:
            self._buffer.append(row)

    def add_rows(self, rows: list[dict[str, Any]]) -> None:
        """Add multiple rows to the buffer."""
        with self._lock:
            self._buffer.extend(rows)

    @property
    def total_written(self) -> int:
        """Return total rows written to disk so far."""
        return self._total_written

    @property
    def buffered_count(self) -> int:
        """Return count of rows in buffer (not yet written)."""
        return len(self._buffer)

    def _periodic_flush_loop(self) -> None:
        """Background thread that flushes buffer periodically."""
        while not self._stop_event.wait(self._interval):
            self._flush()

    def _flush(self) -> None:
        """Flush buffer to disk."""
        with self._lock:
            if not self._buffer:
                return

            rows_to_write = self._buffer
            self._buffer = []

        # Write outside the lock
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for row in rows_to_write:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

            self._total_written += len(rows_to_write)
            _LOGGER.info(
                "Periodic flush: wrote %d rows to %s (total: %d)",
                len(rows_to_write),
                self._path.name,
                self._total_written,
            )
        except Exception as exc:
            # Re-add rows to buffer on failure
            with self._lock:
                self._buffer = rows_to_write + self._buffer
            _LOGGER.error("Failed to flush to %s: %s", self._path, exc)
            raise
