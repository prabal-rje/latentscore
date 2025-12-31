from __future__ import annotations

import logging

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Static

from .branding import APP_NAME
from .logging_utils import (
    MINIMIZE_SIGNAL_FILENAME,
    QUIT_SIGNAL_FILENAME,
    log_path,
    setup_file_logger,
)
from .quit_signal import wait_for_signal_clear
from .tui_base import ChromeHeader, CopySupportApp

LOGGER_NAME = f"{__package__ or 'app'}.textual"
QUIT_SIGNAL_WAIT_TIMEOUT = 2.0
QUIT_SIGNAL_WAIT_INTERVAL = 0.05


# class NonTallHeader(ChromeHeader):
#     """Header that keeps the tall flag disabled on click."""
#     pass


class SampleApp(CopySupportApp):
    """Minimal Textual app placeholder for the tracer bullet."""

    TITLE = APP_NAME

    def on_mount(self) -> None:
        log_file = setup_file_logger(LOGGER_NAME, "textual-ui.log")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("%s mounted (log check).", self.__class__.__name__)
        logger.info("%s logs at %s", APP_NAME, log_file)

    def on_ready(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("%s ready (log check).", self.__class__.__name__)

    def on_click(self, event: events.Click) -> None:
        if isinstance(event.control, ChromeHeader):
            logger = logging.getLogger(LOGGER_NAME)
            logger.info("Header clicked (log check).")

    async def action_quit(self) -> None:
        signal_path = log_path(QUIT_SIGNAL_FILENAME)
        signal_path.write_text("quit", encoding="utf-8")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Quit requested; wrote %s", signal_path)
        await wait_for_signal_clear(
            signal_path,
            timeout=QUIT_SIGNAL_WAIT_TIMEOUT,
            interval=QUIT_SIGNAL_WAIT_INTERVAL,
        )
        self.exit()

    def action_minimize(self) -> None:
        signal_path = log_path(MINIMIZE_SIGNAL_FILENAME)
        signal_path.write_text("minimize", encoding="utf-8")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Minimize requested; wrote %s", signal_path)

    def compose(self) -> ComposeResult:
        yield ChromeHeader(show_clock=False)
        with Vertical():
            yield Static(f"{APP_NAME} is running via the menubar app.", id="body")
            yield Static("Use the menubar item to open this UI.", id="hint")
            yield Static(f"Logs: {log_path('textual-ui.log')}", id="logs")
        yield Footer()


APP_SPEC = f"{__name__}:{SampleApp.__name__}"


def run() -> None:
    setup_file_logger(LOGGER_NAME, "textual-ui.log")
    logging.getLogger(LOGGER_NAME).info(
        "Launching %s (log check).",
        SampleApp.__name__,
    )
    SampleApp().run()


if __name__ == "__main__":
    run()
