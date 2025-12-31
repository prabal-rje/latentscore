from __future__ import annotations

import importlib
import inspect

from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Input, TextArea
from textual.widgets._header import Header, HeaderClock, HeaderClockSpace, HeaderTitle

from .copy_hints import COPY_HINT_MESSAGE


class CopySupportApp(App[None]):
    """Base Textual app with Ctrl+C copy support and consistent notifications."""

    _inherit_bindings = False
    BINDINGS = [
        Binding("ctrl+c", "copy_selection", "Copy", key_display="Ctrl+C", priority=True),
    ]

    def action_copy_selection(self) -> None:
        selected_text = self._selected_text()
        if not selected_text:
            self.notify("No text selected to copy.", timeout=2.0)
            return
        if self._copy_text_to_clipboard(selected_text):
            self.notify("Copied to clipboard.", timeout=2.0)
        else:
            self.notify("Clipboard copy failed; ensure pyperclip is installed.", timeout=2.0)

    def action_copy_hint(self) -> None:
        self.notify(COPY_HINT_MESSAGE, timeout=2.0)

    def _selected_text(self) -> str | None:
        target = self.focused
        if isinstance(target, (TextArea, Input)):
            if target.selected_text:
                return target.selected_text
        try:
            screen = self.screen
        except ScreenStackError:
            screen = None
        if screen is not None:
            screen_selection = screen.get_selected_text()
            if screen_selection:
                return screen_selection
        return None

    def _copy_text_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        if not self._copy_with_pyperclip(text):
            return False
        try:
            self.copy_to_clipboard(text)
        except Exception:
            pass
        return True

    def _copy_with_pyperclip(self, text: str) -> bool:
        try:
            pyperclip = importlib.import_module("pyperclip")
        except Exception:
            return False
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False


class HeaderControl(Widget):
    app: "CopySupportApp"
    DEFAULT_CSS = """
    HeaderControl {
        dock: left;
        height: 1;
        width: 3;
        padding: 0 0;
        content-align: center middle;
        text-style: bold;
    }
    HeaderControl:hover {
        background: $foreground 10%;
    }
    #tl-close {
        color: #ff5f57;
    }
    #tl-minimize {
        color: #febc2e;
    }
    #tl-menu {
        color: #28c840;
    }
    """

    def __init__(self, icon: str, action: str, *, id: str) -> None:
        super().__init__(id=id)
        self._icon = icon
        self._action = action

    def render(self) -> str:
        return self._icon

    async def on_click(self, event: events.Click) -> None:
        app = self.app
        event.stop()
        if self._action == "quit":
            if not await self._invoke_action(app, "quit"):
                app.exit()
            return
        if self._action == "minimize":
            await self._invoke_action(app, "minimize")
            return
        if self._action == "menu":
            await self.run_action("app.command_palette")

    async def _invoke_action(self, app: "CopySupportApp", name: str) -> bool:
        action = getattr(app, f"action_{name}", None)
        if not callable(action):
            return False
        result = action()
        if inspect.isawaitable(result):
            await result
        return True


class ChromeHeader(Header):
    def compose(self) -> ComposeResult:
        yield HeaderControl("X", "quit", id="tl-close")
        yield HeaderControl("\u2014", "minimize", id="tl-minimize")
        yield HeaderControl("O", "menu", id="tl-menu")
        yield HeaderTitle()
        yield (
            HeaderClock().data_bind(Header.time_format) if self._show_clock else HeaderClockSpace()
        )
