from __future__ import annotations

import logging
from typing import ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from .logging_utils import QUIT_SIGNAL_FILENAME, log_path, setup_file_logger

LOGGER_NAME = "latentscore.textual"


class NonTallHeader(Header):
    """Header that keeps the tall flag disabled on click."""

    def _on_click(self) -> None:
        """Call through for now; this inexplicably avoids the titlebar glitch."""
        # We don't know why this variant behaves correctly, but it does in practice.
        super()._on_click()

    def watch_tall(self, tall: bool) -> None:
        """Keep the header in short mode even if tall gets toggled."""
        self.set_class(False, "-tall")


class LatentScoreApp(App[None]):
    """Minimal Textual app placeholder for the tracer bullet."""

    TITLE = "LatentScore"
    _inherit_bindings = False
    BINDINGS: ClassVar[list[BindingType]] = []

    def on_mount(self) -> None:
        log_file = setup_file_logger(LOGGER_NAME, "textual-ui.log")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("LatentScoreApp mounted (log check).")
        logger.info("LatentScore logs at %s", log_file)

    def on_ready(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("LatentScoreApp ready (log check).")

    def on_click(self, event: events.Click) -> None:
        if isinstance(event.control, NonTallHeader):
            logger = logging.getLogger(LOGGER_NAME)
            logger.info("Header clicked (log check).")
            event.control.remove_class("-tall")

    async def action_quit(self) -> None:
        signal_path = log_path(QUIT_SIGNAL_FILENAME)
        signal_path.write_text("quit", encoding="utf-8")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Quit requested; wrote %s", signal_path)
        self.exit()

    def compose(self) -> ComposeResult:
        yield NonTallHeader(show_clock=False)
        with Vertical():
            yield Static("LatentScore is running via the menubar app.", id="body")
            yield Static("Use the menubar item to open this UI.", id="hint")
            yield Static(f"Logs: {log_path('textual-ui.log')}", id="logs")
        yield Footer()


def run() -> None:
    setup_file_logger(LOGGER_NAME, "textual-ui.log")
    logging.getLogger(LOGGER_NAME).info("Launching LatentScoreApp (log check).")
    LatentScoreApp().run()


if __name__ == "__main__":
    run()
