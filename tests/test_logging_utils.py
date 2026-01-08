from __future__ import annotations

import logging

from latentscore import logging_utils


def _reset_logging(monkeypatch) -> None:
    monkeypatch.setattr(logging_utils, "_logging_configured", False)
    logger = logging.getLogger("latentscore")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def test_console_formatter_prefixes_warning() -> None:
    formatter = logging_utils._ConsoleEmojiFormatter("%(level_prefix)s %(message)s")
    record = logging.LogRecord(
        name="latentscore.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="oops",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "⚠️" in rendered


def test_configure_logging_adds_file_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LATENTSCORE_LOG_DIR", str(tmp_path))
    _reset_logging(monkeypatch)
    logging_utils.configure_logging(force=True)
    logger = logging.getLogger("latentscore")
    assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)


def test_configure_logging_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LATENTSCORE_LOG_DIR", str(tmp_path))
    _reset_logging(monkeypatch)
    logging_utils.configure_logging(force=True)
    logger = logging.getLogger("latentscore")
    handler_ids = [id(handler) for handler in logger.handlers]
    logging_utils.configure_logging()
    assert handler_ids == [id(handler) for handler in logger.handlers]
