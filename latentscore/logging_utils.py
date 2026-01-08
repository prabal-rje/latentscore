from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

_LOGGER = logging.getLogger("latentscore.logging")
_LOG_DIR_ENV = "LATENTSCORE_LOG_DIR"
_MODEL_DIR_ENV = "LATENTSCORE_MODEL_DIR"
_LOG_FILE = "latentscore.log"
_logging_configured = False
_CONSOLE_FORMAT = "%(level_prefix)s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LEVEL_PREFIXES = {
    logging.DEBUG: "🐛",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}


def _level_prefix(levelno: int) -> str:
    return _LEVEL_PREFIXES.get(levelno, "")


class _ConsoleEmojiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.level_prefix = _level_prefix(record.levelno)
        return super().format(record)


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


def configure_logging(*, force: bool = False) -> None:
    global _logging_configured
    if _logging_configured and not force:
        return

    logger = logging.getLogger("latentscore")
    logger.setLevel(logging.DEBUG)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    root_has_handlers = bool(logging.getLogger().handlers)
    if force or not root_has_handlers:
        console_handler = logging.StreamHandler(stream=sys.__stderr__)
        console_level = logging.DEBUG if os.environ.get("LATENTSCORE_DEBUG") else logging.INFO
        console_handler.setLevel(console_level)
        console_handler.setFormatter(_ConsoleEmojiFormatter(_CONSOLE_FORMAT))
        logger.addHandler(console_handler)

    try:
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(get_log_path(), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception as exc:
        _LOGGER.warning("Failed to configure file logging: %s", exc, exc_info=True)

    # Allow app/test harness handlers to capture logs.
    logger.propagate = True
    _logging_configured = True


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
