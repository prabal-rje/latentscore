from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

from .branding import APP_ENV_PREFIX, APP_NAME, APP_SLUG

LOG_DIR_ENV = f"{APP_ENV_PREFIX}_LOG_DIR"
LOG_DIR_NAME = APP_NAME
QUIT_SIGNAL_FILENAME = "textual-ui.quit"
DIAGNOSTICS_QUIT_SIGNAL_FILENAME = "diagnostics-ui.quit"


def default_log_dir() -> Path:
    env_dir = os.environ.get(LOG_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / LOG_DIR_NAME
    return Path.home() / f".{APP_SLUG}" / "logs"


def log_path(filename: str, app_support_dir: str | None = None) -> Path:
    base_dir = Path(app_support_dir) if app_support_dir else default_log_dir()
    return base_dir / filename


def setup_file_logger(
    name: str,
    filename: str,
    *,
    level: int = logging.INFO,
    app_support_dir: str | None = None,
) -> Path:
    logger = logging.getLogger(name)
    path = log_path(filename, app_support_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return path
    logger.setLevel(level)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return path
