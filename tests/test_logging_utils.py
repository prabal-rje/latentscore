import logging

from app.logging_utils import LOG_DIR_ENV, default_log_dir, log_path, setup_file_logger


def test_log_path_uses_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    assert default_log_dir() == tmp_path
    assert log_path("demo.log") == tmp_path / "demo.log"


def test_setup_file_logger_creates_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    logger_name = f"app.test.{tmp_path.name}"
    path = setup_file_logger(logger_name, "app.log")
    assert path == tmp_path / "app.log"
    logger = logging.getLogger(logger_name)
    assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
