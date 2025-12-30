from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from .logging_utils import LOG_DIR_ENV, default_log_dir
from .loop import install_uvloop_policy


@dataclass
class _LogPane:
    title: str
    path: Path
    widget: RichLog
    position: int = 0
    missing_noted: bool = False
    lines: list[str] = field(default_factory=list)
    match_index: int = -1


class DiagnosticsHeader(Header):
    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            app = self.app
            if isinstance(app, DiagnosticsApp):
                app.action_show_help()


class DiagnosticsApp(App[None]):
    """Textual log viewer for LatentScore diagnostics."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("r", "reload", "Reload"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "close_search", "Close Search"),
        Binding("n", "search_next", "Next Match"),
        Binding("N", "search_prev", "Prev Match"),
        Binding("?", "show_help", "Help"),
        Binding("h", "show_help", "Help"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #logs {
        height: 1fr;
    }

    #search-bar {
        display: none;
        height: auto;
    }

    #search-bar.visible {
        display: block;
    }

    #search-label {
        padding: 0 1;
    }

    #help {
        display: none;
        padding: 1;
        border: round $secondary;
        background: $panel;
    }

    #help.visible {
        display: block;
    }

    .pane {
        width: 1fr;
        border: round $secondary;
    }

    .pane-title {
        padding: 0 1;
        background: $boost;
    }

    RichLog {
        height: 1fr;
    }
    """

    def __init__(self, log_dir: Path, *, max_lines: int = 5000) -> None:
        super().__init__()
        self._log_dir = log_dir
        self._max_lines = max_lines
        self._follow = True
        self._search_query: str | None = None
        self._search_visible = False
        self._active_pane: _LogPane | None = None
        self._panes: list[_LogPane] = []
        self._menubar_log = RichLog(
            highlight=False, wrap=False, max_lines=max_lines, id="menubar-log"
        )
        self._ui_log = RichLog(
            highlight=False, wrap=False, max_lines=max_lines, id="ui-log"
        )
        self._search_input = Input(placeholder="Type and press Enter", id="search-input")
        help_text = (
            "Diagnostics viewer\n"
            "  /      search (press Enter to jump)\n"
            "  n/N    next/prev match\n"
            "  f      toggle follow\n"
            "  r      reload logs\n"
            "  q      quit"
        )
        self._help = Static(help_text, id="help")

    def compose(self) -> ComposeResult:
        yield DiagnosticsHeader(show_clock=False)
        yield self._help
        with Horizontal(id="search-bar"):
            yield Static("Search:", id="search-label")
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
        self._panes = [
            _LogPane(
                "Menubar Server",
                self._log_dir / "menubar-server.log",
                self._menubar_log,
            ),
            _LogPane("Textual UI", self._log_dir / "textual-ui.log", self._ui_log),
        ]
        for pane in self._panes:
            pane.widget.auto_scroll = self._follow
            pane.widget.can_focus = True
        self._active_pane = self._panes[0]
        self._active_pane.widget.focus()
        self._load_initial()
        self.set_interval(0.35, self._poll_files)

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        for pane in self._panes:
            pane.widget.auto_scroll = self._follow
        status = "on" if self._follow else "off"
        self.notify(f"Follow {status}", timeout=1.0)

    def action_reload(self) -> None:
        for pane in self._panes:
            pane.widget.clear()
            pane.position = 0
            pane.missing_noted = False
            pane.lines.clear()
            pane.match_index = -1
        self._load_initial()
        self.notify("Reloaded logs", timeout=1.0)

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

    def on_focus(self, event: events.Focus) -> None:
        control = event.control
        if isinstance(control, RichLog):
            pane = self._pane_for_widget(control)
            if pane is not None:
                self._active_pane = pane

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._search_input:
            return
        query = event.value.strip()
        if not query:
            self._search_query = None
            self.notify("Search cleared", timeout=1.0)
            self.action_close_search()
            return
        self._follow = False
        for pane in self._panes:
            pane.widget.auto_scroll = False
        self._search_query = query
        self._jump_match(1, reset=True)
        self.action_close_search()

    def _load_initial(self) -> None:
        for pane in self._panes:
            if not pane.path.exists():
                if not pane.missing_noted:
                    pane.widget.write(
                        f"{pane.title} log not found yet at {pane.path}"
                    )
                    pane.missing_noted = True
                continue
            text = pane.path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            tail = lines[-200:] if len(lines) > 200 else lines
            for line in tail:
                self._append_line(pane, line)
            pane.position = pane.path.stat().st_size

    def _poll_files(self) -> None:
        for pane in self._panes:
            if not pane.path.exists():
                if not pane.missing_noted:
                    pane.widget.write(
                        f"{pane.title} log not found yet at {pane.path}"
                    )
                    pane.missing_noted = True
                continue
            try:
                size = pane.path.stat().st_size
            except OSError:
                continue
            if size < pane.position:
                pane.lines.clear()
                pane.match_index = -1
                self._append_line(pane, f"{pane.title} log rotated")
                pane.position = 0
            with pane.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(pane.position)
                chunk = handle.read()
                pane.position = handle.tell()
            if not chunk:
                continue
            for line in chunk.splitlines():
                self._append_line(pane, line)

    def _append_line(self, pane: _LogPane, line: str) -> None:
        pane.lines.append(line)
        if len(pane.lines) > self._max_lines:
            pane.lines.pop(0)
        pane.widget.write(line)

    def _pane_for_widget(self, widget: RichLog) -> _LogPane | None:
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
            return
        if reset or pane.match_index < 0:
            pane.match_index = 0
        else:
            pane.match_index = (pane.match_index + direction) % len(matches)
        line_index = matches[pane.match_index]
        pane.widget.scroll_to(y=line_index, animate=False, immediate=True)
        self.notify(
            f"{pane.title}: match {pane.match_index + 1}/{len(matches)}",
            timeout=1.0,
        )

    def _search_matches(self, pane: _LogPane, query: str) -> list[int]:
        target = query.lower()
        return [idx for idx, line in enumerate(pane.lines) if target in line.lower()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LatentScore diagnostics viewer")
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


def run() -> None:
    args = _parse_args()
    build_app(args.log_dir).run()


if __name__ == "__main__":
    run()
