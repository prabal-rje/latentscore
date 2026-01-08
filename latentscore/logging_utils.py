from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

_LOGGER = logging.getLogger("latentscore.logging")
_LOG_DIR_ENV = "LATENTSCORE_LOG_DIR"
_MODEL_DIR_ENV = "LATENTSCORE_MODEL_DIR"
_LOG_FILE = "latentscore.log"


def _default_model_dir() -> Path:
    configured = os.environ.get(_MODEL_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "latentscore" / "models"


def get_log_dir() -> Path:
    configured = os.environ.get(_LOG_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return _default_model_dir().parent / "logs"


def get_log_path() -> Path:
    return get_log_dir() / _LOG_FILE


def log_exception(context: str, exc: BaseException) -> Path | None:
    try:
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = get_log_path()
        timestamp = datetime.now().isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {context} failed: {type(exc).__name__}: {exc}\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=handle)
            handle.write("\n")
        return path
    except Exception as log_exc:
        _LOGGER.warning("Failed to write log file: %s", log_exc, exc_info=True)
        return None
