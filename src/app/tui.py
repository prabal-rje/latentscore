from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from .loop import install_uvloop_policy


class HelloWorldApp(App[None]):
    """Minimal Textual application that shows a hello world message."""

    CSS = """
    Screen {
        align: center middle;
    }

    #greeting {
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Hello world!", id="greeting")


def run_tui() -> None:
    """Install uvloop and run the hello world Textual app."""
    install_uvloop_policy()
    HelloWorldApp().run()


if __name__ == "__main__":
    run_tui()
