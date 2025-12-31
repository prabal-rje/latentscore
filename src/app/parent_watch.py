from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Mapping

from .branding import APP_ENV_PREFIX

PARENT_FD_ENV = f"{APP_ENV_PREFIX}_PARENT_FD"
PARENT_PID_ENV = f"{APP_ENV_PREFIX}_PARENT_PID"


@dataclass(frozen=True)
class ParentWatchConfig:
    fd: int | None
    pid: int | None


def create_parent_watch_pipe() -> tuple[int, int]:
    return os.pipe()


def parse_parent_watch_env(env: Mapping[str, str] | None = None) -> ParentWatchConfig:
    source = os.environ if env is None else env
    fd = _coerce_fd(source.get(PARENT_FD_ENV))
    pid = _coerce_pid(source.get(PARENT_PID_ENV))
    return ParentWatchConfig(fd=fd, pid=pid)


_started_fd = False
_started_pid = False


def start_parent_watchdog_from_env(
    env: Mapping[str, str] | None = None,
    *,
    interval: float = 1.0,
) -> bool:
    config = parse_parent_watch_env(env)
    return start_parent_watchdog(config, interval=interval)


def start_parent_watchdog(config: ParentWatchConfig, *, interval: float = 1.0) -> bool:
    started = False
    global _started_fd, _started_pid
    if config.fd is not None and not _started_fd:
        thread = threading.Thread(target=_watch_fd, args=(config.fd,), daemon=True)
        thread.start()
        _started_fd = True
        started = True
    if config.pid is not None and not _started_pid:
        thread = threading.Thread(target=_watch_pid, args=(config.pid, interval), daemon=True)
        thread.start()
        _started_pid = True
        started = True
    return started


def _coerce_fd(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        fd = int(value)
    except ValueError:
        return None
    if fd < 0:
        return None
    try:
        os.fstat(fd)
    except OSError:
        return None
    return fd


def _coerce_pid(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        pid = int(value)
    except ValueError:
        return None
    if pid <= 1:
        return None
    return pid


def _watch_fd(fd: int) -> None:
    while True:
        try:
            data = os.read(fd, 1024)
        except OSError:
            break
        if not data:
            break
    _terminate_self()


def _watch_pid(pid: int, interval: float) -> None:
    interval = interval if interval > 0 else 1.0
    while True:
        if not _pid_alive(pid):
            _terminate_self()
            return
        time.sleep(interval)


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_self() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)
    time.sleep(1.0)
    os._exit(0)
