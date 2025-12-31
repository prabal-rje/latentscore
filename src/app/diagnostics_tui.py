from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static, TextArea

from .branding import APP_NAME
from .logging_utils import DIAGNOSTICS_QUIT_SIGNAL_FILENAME, LOG_DIR_ENV, default_log_dir
from .loop import install_uvloop_policy


def _empty_lines() -> list[str]:
    return []


@dataclass
class _LogPane:
    title: str
    path: Path
    widget: TextArea
    position: int = 0
    missing_noted: bool = False
    lines: list[str] = field(default_factory=_empty_lines)
    match_index: int = -1
    highlight_line: int | None = None


class DiagnosticsHeader(Header):
    app: "DiagnosticsApp"

    def _on_click(self) -> None:
        return

    def watch_tall(self, tall: bool) -> None:
        self.set_class(False, "-tall")

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            app = self.app
            assert isinstance(app, DiagnosticsApp)
            app.action_show_help()


class DiagnosticsApp(App[None]):
    """Textual log viewer for diagnostics."""

    _inherit_bindings = False
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("Ctrl+c", "copy_selection", "Copy"),
        Binding("r", "reload", "Reload"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "close_search", "Close Search"),
        Binding("n", "search_next", "Next Match"),
        Binding("N", "search_prev", "Prev Match"),
        Binding("?", "show_help", "Help"),
        Binding("h", "show_help", "Help"),
    ]

    # Minimal CSS for layout only - widget styling done via native SDK
    CSS = """
    Screen {
        layout: vertical;
    }

    #logs {
        height: 1fr;
        border-top: solid $secondary;
    }

    #search-bar {
        display: none;
        height: auto;
        padding: 1;
        background: $surface;
        border-bottom: solid $secondary;
    }

    #search-bar.visible {
        display: block;
    }

    #search-label {
        padding-right: 1;
        padding-top: 1;
        height: 3;
        content-align: center middle;
    }

    #search-input {
        width: 1fr;
        height: auto; /* Let the widget decide its own natural height */
        border: tall $accent;
        background: $surface;
        color: $foreground;
    }

    #help {
        display: none;
        padding: 1;
        background: $panel;
        border: tall $primary;
        dock: bottom;
        layer: overlay;
        width: 60%;
        height: auto;
    }

    #help.visible {
        display: block;
    }

    #size-warning {
        display: none;
        layer: overlay;
        width: 100%;
        height: 100%;
        background: $panel;
        color: $text-muted;
        content-align: center middle;
        border: tall $accent;
    }

    #size-warning.visible {
        display: block;
    }

    .pane {
        width: 1fr;
    }

    .pane-title {
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    TextArea {
        height: 1fr;
    }

    TextArea .text-area--selection {
        background: $accent 40%;
    }
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        max_lines: int = 5000,
        min_viewport_width: int = 110,
        min_viewport_height: int = 28,
    ) -> None:
        super().__init__()
        self._log_dir = log_dir
        self._max_lines = max_lines
        self._min_viewport_width = min_viewport_width
        self._min_viewport_height = min_viewport_height
        self._follow = True
        self._search_query: str | None = None
        self._search_visible = False
        self._active_pane: _LogPane | None = None
        self._panes: list[_LogPane] = []
        self._size_warning = Static(
            "Please re-size window to view properly",
            id="size-warning",
        )
        # Use TextArea for proper text selection and copy support
        self._menubar_log = TextArea(id="menubar-log", read_only=True)
        self._ui_log = TextArea(id="ui-log", read_only=True)
        self._search_input = Input(
            placeholder="Type search query and press Enter", id="search-input"
        )
        help_text = (
            "Diagnostics viewer\n"
            "  /      search (press Enter to jump)\n"
            "  n/N    next/prev match\n"
            "  f      toggle follow\n"
            "  r      reload logs\n"
            "  Ctrl+C  copy selection\n"
            "  q      quit"
        )
        self._help = Static(help_text, id="help")

    def compose(self) -> ComposeResult:
        yield DiagnosticsHeader(show_clock=False)
        yield self._help
        yield self._size_warning
        with Horizontal(id="search-bar"):
            yield self._search_input
        with Horizontal(id="logs"):
            with Vertical(classes="pane"):
                yield Static("Menubar Server", classes="pane-title")
                yield self._menubar_log
            with Vertical(classes="pane"):
                yield Static("Textual UI", classes="pane-title")
                yield self._ui_log
        yield Footer()

    def on_mount(self) -> None:
        self.notify("Diagnostics viewer mounted (runtime check).", timeout=2.0)
        self._panes = [
            _LogPane(
                "Menubar Server",
                self._log_dir / "menubar-server.log",
                self._menubar_log,
            ),
            _LogPane("Textual UI", self._log_dir / "textual-ui.log", self._ui_log),
        ]
        self._active_pane = self._panes[0]
        self._active_pane.widget.focus()
        self._load_initial()
        self._update_size_warning()
        self.set_interval(0.35, self._poll_files)

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        status = "on" if self._follow else "off"
        self.notify(f"Follow {status}", timeout=1.0)
        if self._follow:
            self._scroll_to_end_all()

    def action_reload(self) -> None:
        for pane in self._panes:
            pane.widget.load_text("")
            pane.position = 0
            pane.missing_noted = False
            pane.lines.clear()
            pane.match_index = -1
        self._load_initial()
        self.notify("Reloaded logs", timeout=1.0)

    def _scroll_to_end_all(self) -> None:
        """Scroll all log panes to the end."""
        for pane in self._panes:
            pane.widget.scroll_end(animate=False)

    def action_focus_search(self) -> None:
        self._search_visible = True
        self.query_one("#search-bar").add_class("visible")
        self._search_input.value = self._search_query or ""
        self._search_input.focus()

    def action_close_search(self) -> None:
        if not self._search_visible:
            return
        self._search_visible = False
        self.query_one("#search-bar").remove_class("visible")
        if self._active_pane:
            self._active_pane.widget.focus()

    def action_search_next(self) -> None:
        self._jump_match(1)

    def action_search_prev(self) -> None:
        self._jump_match(-1)

    def action_show_help(self) -> None:
        help_widget = self.query_one("#help")
        if "visible" in help_widget.classes:
            help_widget.remove_class("visible")
        else:
            help_widget.add_class("visible")

    def action_copy_selection(self) -> None:
        target = self.focused
        if isinstance(target, (TextArea, Input)):
            try:
                target.action_copy()
            except SkipAction:
                return
            return
        if self._active_pane is None:
            return
        try:
            self._active_pane.widget.action_copy()
        except SkipAction:
            return

    def on_focus(self, event: events.Focus) -> None:
        control = event.control
        if isinstance(control, TextArea):
            pane = self._pane_for_widget(control)
            if pane is not None:
                self._active_pane = pane

    def on_click(self, event: events.Click) -> None:
        if isinstance(event.control, Header):
            event.control.remove_class("-tall")

    def on_resize(self, event: events.Resize) -> None:
        self._update_size_warning(event.size.width, event.size.height)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._search_input:
            return
        query = event.value.strip()
        if not query:
            self._search_query = None
            if self._active_pane:
                self._active_pane.highlight_line = None
            self.notify("Search cleared", timeout=1.0)
            self.action_close_search()
            return
        self._follow = False
        self._search_query = query
        self._jump_match(1, reset=True)
        self.action_close_search()

    async def action_quit(self) -> None:
        signal_path = self._log_dir / DIAGNOSTICS_QUIT_SIGNAL_FILENAME
        signal_path.write_text("quit", encoding="utf-8")
        self.exit()

    def _load_initial(self) -> None:
        for pane in self._panes:
            if not pane.path.exists():
                if not pane.missing_noted:
                    pane.lines.append(f"{pane.title} log not found yet at {pane.path}")
                    self._update_pane_text(pane)
                    pane.missing_noted = True
                continue
            text = pane.path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            tail = lines[-200:] if len(lines) > 200 else lines
            pane.lines.extend(tail)
            self._update_pane_text(pane)
            pane.position = pane.path.stat().st_size
        if self._follow:
            self._scroll_to_end_all()

    def _poll_files(self) -> None:
        any_updated = False
        for pane in self._panes:
            if not pane.path.exists():
                if not pane.missing_noted:
                    pane.lines.append(f"{pane.title} log not found yet at {pane.path}")
                    self._update_pane_text(pane)
                    pane.missing_noted = True
                continue
            try:
                size = pane.path.stat().st_size
            except OSError:
                continue
            if size < pane.position:
                pane.lines.clear()
                pane.match_index = -1
                pane.lines.append(f"{pane.title} log rotated")
                pane.position = 0
                self._update_pane_text(pane)
                any_updated = True
            with pane.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(pane.position)
                chunk = handle.read()
                pane.position = handle.tell()
            if not chunk:
                continue
            new_lines = chunk.splitlines()
            pane.lines.extend(new_lines)
            # Trim to max_lines
            if len(pane.lines) > self._max_lines:
                pane.lines = pane.lines[-self._max_lines :]
            self._update_pane_text(pane)
            any_updated = True
        if any_updated and self._follow:
            self._scroll_to_end_all()

    def _update_pane_text(self, pane: _LogPane) -> None:
        """Update the TextArea content with all lines."""
        pane.widget.load_text("\n".join(pane.lines))

    def _update_size_warning(self, width: int | None = None, height: int | None = None) -> None:
        size = self.size
        resolved_width = width if width is not None else size.width
        resolved_height = height if height is not None else size.height
        too_small = (
            resolved_width < self._min_viewport_width or resolved_height < self._min_viewport_height
        )
        self._size_warning.set_class(too_small, "visible")

    def _pane_for_widget(self, widget: TextArea) -> _LogPane | None:
        for pane in self._panes:
            if pane.widget is widget:
                return pane
        return None

    def _jump_match(self, direction: int, *, reset: bool = False) -> None:
        if not self._search_query:
            self.notify("No active search", timeout=1.0)
            return
        pane = self._active_pane
        if pane is None:
            return
        matches = self._search_matches(pane, self._search_query)
        if not matches:
            self.notify(f"No matches for {self._search_query!r}", timeout=1.0)
            pane.match_index = -1
            pane.highlight_line = None
            return
        if reset or pane.match_index < 0:
            pane.match_index = 0
        else:
            pane.match_index = (pane.match_index + direction) % len(matches)
        line_index = matches[pane.match_index]
        pane.highlight_line = line_index
        # Move cursor to the matching line and select it
        pane.widget.focus()
        pane.widget.move_cursor((line_index, 0))
        pane.widget.select_line(line_index)
        self.notify(
            f"{pane.title}: match {pane.match_index + 1}/{len(matches)}",
            timeout=1.0,
        )

    def _search_matches(self, pane: _LogPane, query: str) -> list[int]:
        target = query.lower()
        return [idx for idx, line in enumerate(pane.lines) if target in line.lower()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} diagnostics viewer")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir(),
        help=f"Log directory (default: ${LOG_DIR_ENV} or platform default)",
    )
    return parser.parse_args()


def build_app(log_dir: Path | None = None) -> DiagnosticsApp:
    install_uvloop_policy()
    return DiagnosticsApp(log_dir or default_log_dir())


APP_SPEC = f"{__name__}:{build_app.__name__}"


def run() -> None:
    args = _parse_args()
    build_app(args.log_dir).run()


if __name__ == "__main__":
    run()
