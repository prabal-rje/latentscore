from __future__ import annotations

from textual import log
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from .logging_utils import log_path, setup_file_logger

class NonTallHeader(Header):
    """Header that keeps the tall flag disabled on click."""

    def _on_click(self) -> None:
        """Override click handler to prevent tall mode toggle."""
        log.info("clicked tall header 0")
        log.error("clicked tall header 1")
        log.warning("clicked tall header 2")
        print("clicked tall header 3")
        pass

class LatentScoreApp(App[None]):
    """Minimal Textual app placeholder for the tracer bullet."""

    TITLE = "LatentScore"

    def on_mount(self) -> None:
        log_file = setup_file_logger("latentscore.textual", "textual-ui.log")
        log(f"LatentScore logs at {log_file}")

    def compose(self) -> ComposeResult:
        yield NonTallHeader(show_clock=False)
        with Vertical():
            yield Static("LatentScore is running via the menubar app.", id="body")
            yield Static("Use the menubar item to open this UI.", id="hint")
            yield Static(f"Logs: {log_path('textual-ui.log')}", id="logs")
        yield Footer()


def run() -> None:
    LatentScoreApp().run()


if __name__ == "__main__":
    run()
