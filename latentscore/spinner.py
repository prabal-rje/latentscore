from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import IO, Iterator


class Spinner:
    """Minimal ASCII spinner for CLI/TUI feedback."""

    def __init__(
        self,
        message: str,
        *,
        interval: float = 0.1,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._message = message
        self._interval = interval
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty() if enabled is None else enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_len = 0

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, message: str) -> None:
        with self._lock:
            self._message = message

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        self._clear_line()

    def _run(self) -> None:
        frames = "|/-\\"
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            message = self._current_message()
            text = f"{message} {frame}"
            self._render(text)
            index += 1
            time.sleep(self._interval)
        self._clear_line()

    def _current_message(self) -> str:
        with self._lock:
            return self._message

    def _render(self, text: str) -> None:
        self._last_len = max(self._last_len, len(text))
        self._stream.write(f"\r{text}")
        self._stream.flush()

    def _clear_line(self) -> None:
        if self._last_len <= 0:
            return
        self._stream.write("\r" + (" " * self._last_len) + "\r")
        self._stream.flush()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()


@contextmanager
def spinner(
    message: str,
    *,
    interval: float = 0.1,
    stream: IO[str] | None = None,
    enabled: bool | None = None,
) -> Iterator[Spinner]:
    handle = Spinner(
        message,
        interval=interval,
        stream=stream,
        enabled=enabled,
    )
    handle.start()
    try:
        yield handle
    finally:
        handle.stop()
