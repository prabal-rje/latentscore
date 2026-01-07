from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from rich.console import Console as RichConsole
    from rich.status import Status as RichStatus
else:
    RichConsole = Any
    RichStatus = Any

try:
    from rich.console import Console
    from rich.status import Status
except ImportError:  # pragma: no cover - optional dependency
    Console = None
    Status = None


class _AsciiSpinner:
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


class _RichSpinner:
    def __init__(
        self,
        message: str,
        *,
        stream: IO[str] | None = None,
    ) -> None:
        self._message = message
        self._stream = stream or sys.stderr
        self._console: RichConsole | None = Console(file=self._stream) if Console else None
        self._status: RichStatus | None = None

    def start(self) -> None:
        if self._console is None or Status is None:
            return
        if self._status is not None:
            return
        self._status = self._console.status(self._message)
        self._status.start()

    def update(self, message: str) -> None:
        self._message = message
        if self._status is not None:
            self._status.update(message)

    def stop(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None


class Spinner:
    """Spinner helper with Rich status fallback."""

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
        self._backend: _AsciiSpinner | _RichSpinner | None = None
        if not self._enabled:
            return
        if Console and Status:
            self._backend = _RichSpinner(self._message, stream=self._stream)
        else:
            self._backend = _AsciiSpinner(
                self._message,
                interval=self._interval,
                stream=self._stream,
                enabled=self._enabled,
            )

    def start(self) -> None:
        if self._backend is None:
            return
        self._backend.start()

    def update(self, message: str) -> None:
        self._message = message
        if self._backend is None:
            return
        self._backend.update(message)

    def stop(self) -> None:
        if self._backend is None:
            return
        self._backend.stop()

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
