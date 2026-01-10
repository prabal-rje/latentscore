from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from rich.console import Console as RichConsole
    from rich.progress import Progress as RichProgress
    from rich.progress import TaskID as RichTaskID
    from rich.status import Status as RichStatus
else:
    RichConsole = Any
    RichProgress = Any
    RichTaskID = Any
    RichStatus = Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.status import Status
    from rich.text import Text
    from rich.traceback import Traceback
except ImportError:  # pragma: no cover - optional dependency
    Console = None
    Panel = None
    Progress = None
    Status = None
    Text = None
    Traceback = None
    BarColumn = None
    TextColumn = None
    SpinnerColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None


class _AsciiSpinner:
    def __init__(
        self,
        message: str,
        *,
        interval: float = 0.1,
        frames: str = "|/-\\",
        stream: IO[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._message = message
        self._interval = interval
        self._frames = frames
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
        index = 0
        while not self._stop.is_set():
            frame = self._frames[index % len(self._frames)]
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
        spinner: str | None = None,
        stream: IO[str] | None = None,
    ) -> None:
        self._message = message
        self._spinner = spinner or "dots"
        self._stream = stream or sys.stderr
        self._console: RichConsole | None = Console(file=self._stream) if Console else None
        self._status: RichStatus | None = None

    def start(self) -> None:
        if self._console is None or Status is None:
            return
        if self._status is not None:
            return
        self._status = self._console.status(self._message, spinner=self._spinner)
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


class _RichElapsedSpinner:
    def __init__(
        self,
        message: str,
        *,
        spinner: str | None = None,
        stream: IO[str] | None = None,
    ) -> None:
        self._message = message
        self._spinner = spinner or "dots"
        self._stream = stream or sys.stderr
        self._console: RichConsole | None = Console(file=self._stream) if Console else None
        self._progress: RichProgress | None = None
        self._task_id: RichTaskID | None = None

    def start(self) -> None:
        if (
            self._console is None
            or Progress is None
            or SpinnerColumn is None
            or TextColumn is None
            or TimeElapsedColumn is None
        ):
            return
        if self._progress is not None:
            return
        self._progress = Progress(
            SpinnerColumn(spinner_name=self._spinner),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self._message, total=None)

    def update(self, message: str) -> None:
        self._message = message
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=message)

    def stop(self) -> None:
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None
        self._task_id = None


class Spinner:
    """Spinner helper with Rich status fallback."""

    def __init__(
        self,
        message: str,
        *,
        interval: float = 0.1,
        frames: str | None = None,
        spinner: str | None = None,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
        show_elapsed: bool = False,
    ) -> None:
        self._message = message
        self._interval = interval
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty() if enabled is None else enabled
        self._backend: _AsciiSpinner | _RichSpinner | _RichElapsedSpinner | None = None
        if not self._enabled:
            return
        if Console:
            if show_elapsed and Progress and SpinnerColumn and TextColumn and TimeElapsedColumn:
                self._backend = _RichElapsedSpinner(
                    self._message,
                    spinner=spinner,
                    stream=self._stream,
                )
            elif Status:
                self._backend = _RichSpinner(
                    self._message,
                    spinner=spinner,
                    stream=self._stream,
                )
            else:
                self._backend = _AsciiSpinner(
                    self._message,
                    interval=self._interval,
                    frames=frames or "|/-\\",
                    stream=self._stream,
                    enabled=self._enabled,
                )
        else:
            self._backend = _AsciiSpinner(
                self._message,
                interval=self._interval,
                frames=frames or "|/-\\",
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
    frames: str | None = None,
    spinner: str | None = None,
    stream: IO[str] | None = None,
    enabled: bool | None = None,
    show_elapsed: bool = False,
) -> Iterator[Spinner]:
    handle = Spinner(
        message,
        interval=interval,
        frames=frames,
        spinner=spinner,
        stream=stream,
        enabled=enabled,
        show_elapsed=show_elapsed,
    )
    handle.start()
    try:
        yield handle
    finally:
        handle.stop()


class _AsciiProgressBar:
    def __init__(
        self,
        message: str,
        *,
        total: float,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
        width: int = 24,
    ) -> None:
        self._message = message
        self._total = total
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty() if enabled is None else enabled
        self._width = width
        self._last_len = 0

    def start(self) -> None:
        if not self._enabled:
            return
        self.update(0.0)

    def update(self, completed: float) -> None:
        if not self._enabled:
            return
        if self._total <= 0:
            fraction = 1.0
        else:
            fraction = max(0.0, min(completed / self._total, 1.0))
        filled = int(round(self._width * fraction))
        bar = "=" * filled + "-" * (self._width - filled)
        percent = int(round(100 * fraction))
        text = f"{self._message} [{bar}] {percent:>3d}%"
        self._render(text)

    def stop(self) -> None:
        if not self._enabled:
            return
        self._clear_line()

    def _render(self, text: str) -> None:
        self._last_len = max(self._last_len, len(text))
        self._stream.write(f"\r{text}")
        self._stream.flush()

    def _clear_line(self) -> None:
        if self._last_len <= 0:
            return
        self._stream.write("\r" + (" " * self._last_len) + "\r")
        self._stream.flush()


class _RichProgressBar:
    def __init__(
        self,
        message: str,
        *,
        total: float,
        stream: IO[str] | None = None,
    ) -> None:
        self._message = message
        self._total = total
        self._stream = stream or sys.stderr
        self._console: RichConsole | None = Console(file=self._stream) if Console else None
        self._progress: RichProgress | None = None
        self._task_id: RichTaskID | None = None

    def start(self) -> None:
        if (
            self._console is None
            or Progress is None
            or TextColumn is None
            or BarColumn is None
            or TimeElapsedColumn is None
            or TimeRemainingColumn is None
        ):
            return
        if self._progress is not None:
            return
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self._message, total=self._total)

    def update(self, completed: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, completed=min(completed, self._total))

    def stop(self) -> None:
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None
        self._task_id = None


class ProgressBar:
    """Progress bar helper with Rich fallback."""

    def __init__(
        self,
        message: str,
        *,
        total: float,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._message = message
        self._total = total
        self._stream = stream or sys.stderr
        self._enabled = self._stream.isatty() if enabled is None else enabled
        self._backend: _AsciiProgressBar | _RichProgressBar | None = None
        if not self._enabled:
            return
        if Console and Progress:
            self._backend = _RichProgressBar(
                self._message,
                total=self._total,
                stream=self._stream,
            )
        else:
            self._backend = _AsciiProgressBar(
                self._message,
                total=self._total,
                stream=self._stream,
                enabled=self._enabled,
            )

    def start(self) -> None:
        if self._backend is None:
            return
        self._backend.start()

    def update(self, completed: float) -> None:
        if self._backend is None:
            return
        self._backend.update(completed)

    def stop(self) -> None:
        if self._backend is None:
            return
        self._backend.stop()


def render_error(
    context: str,
    exc: BaseException,
    *,
    stream: IO[str] | None = None,
) -> None:
    target = stream or sys.stderr
    debug = os.environ.get("LATENTSCORE_DEBUG")
    log_path = None
    try:
        from .logging_utils import get_log_path

        log_path = get_log_path()
    except Exception:
        log_path = None
    if Console and Panel and Text and Traceback and target.isatty():
        console = Console(file=target)
        headline = Text(f"{type(exc).__name__}", style="bold red")
        summary = Text(str(exc))
        log_line = ""
        if log_path is not None:
            log_line = f"\nLogs: {log_path}"
        body = Text.assemble(
            ("LatentScore error while ", "bold"),
            (context, "bold"),
            (":\n\n", "bold"),
            headline,
            (": ", "bold"),
            summary,
            (log_line, "dim"),
            ("\n\nSet LATENTSCORE_DEBUG=1 for console trace.", "dim"),
        )
        console.print(Panel(body, title="Error", border_style="red"))
        if debug:
            console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))
    else:
        suffix = f" (logs: {log_path})" if log_path is not None else ""
        target.write(f"{context} failed: {type(exc).__name__}: {exc}{suffix}\n")
        if debug:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=target)
