from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static


class LatentScoreApp(App[None]):
    """Minimal Textual app placeholder for the tracer bullet."""

    TITLE = "LatentScore"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical():
            yield Static("LatentScore is running via the menubar app.", id="body")
            yield Static("Use the menubar item to open this UI.", id="hint")
        yield Footer()


def run() -> None:
    LatentScoreApp().run()


if __name__ == "__main__":
    run()
