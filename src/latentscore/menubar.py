from __future__ import annotations

import atexit
import importlib.util
import os
import platform
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, Optional, cast

import rumps  # type: ignore[import]
from rumps import utils as rumps_utils  # type: ignore[import]

GREETING_TITLE = "Say hi"
GREETING_MESSAGE = "Hi there!"
OPEN_UI_TITLE = "Open LatentScore"


def require_macos() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("The menu bar helper only runs on macOS.")


class MenuBarApp(rumps.App):
    """Simple macOS status bar app with a single greeting button."""

    def __init__(
        self,
        enable_alerts: bool = True,
        app_support_dir: Optional[str] = None,
        server_enabled: bool = True,
        server_host: str = "127.0.0.1",
        server_port: Optional[int] = None,
        textual_target: str = "latentscore.textual_app:LatentScoreApp",
        server_ready_timeout: float = 3.0,
        server_ready_interval: float = 0.1,
        window_title: str = "LatentScore",
        window_width: int = 1000,
        window_height: int = 700,
        window_resizable: bool = False,
        window_frameless: bool = False,
        window_easy_drag: bool = True,
        initialize: bool = True,
    ) -> None:
        self.enable_alerts = enable_alerts
        self.server_enabled = server_enabled
        self.server_host = server_host
        self._server_port = server_port
        self.textual_target = textual_target
        self.server_ready_timeout = 0.0 if not initialize else server_ready_timeout
        self.server_ready_interval = server_ready_interval
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        self.window_resizable = window_resizable
        self.window_frameless = window_frameless
        self.window_easy_drag = window_easy_drag
        self._app_support_dir = app_support_dir
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._server_log_file: IO[bytes] | None = None
        self._webview_proc: subprocess.Popen[bytes] | None = None
        if not initialize:
            self.hi_item = SimpleNamespace(title=GREETING_TITLE)
            self.open_item = SimpleNamespace(title=OPEN_UI_TITLE)
            return

        if app_support_dir:
            base = Path(app_support_dir)
            base.mkdir(parents=True, exist_ok=True)

            def _application_support(_name: str) -> str:
                base.mkdir(parents=True, exist_ok=True)
                return str(base)

            cast(Any, rumps).application_support = _application_support
            cast(Any, rumps_utils).application_support = _application_support

        super().__init__("LatentScore")  # type: ignore[misc]
        atexit.register(self._stop_server)
        atexit.register(self._stop_webview)
        self.hi_item = cast(Any, rumps).MenuItem(
            GREETING_TITLE, callback=self._on_hi_clicked
        )
        self.open_item = cast(Any, rumps).MenuItem(
            OPEN_UI_TITLE, callback=self._on_open_clicked
        )
        self.menu = [self.open_item, self.hi_item]
        if self.server_enabled:
            self._ensure_server()

    def _on_hi_clicked(self, _sender: object) -> str:
        if self.enable_alerts:
            cast(Any, rumps).alert("latentscore", GREETING_MESSAGE)
        return GREETING_MESSAGE

    def _on_open_clicked(self, _sender: object) -> str:
        if not self._ensure_server():
            return ""
        url = self._current_url()
        if not self._open_webview(url):
            webbrowser.open(url)
        return url

    def _current_url(self) -> str:
        port = self._server_port or 0
        return f"http://{self.server_host}:{port}"

    def _ensure_server(self) -> bool:
        if not self.server_enabled:
            return False
        if self._server_proc and self._server_proc.poll() is None:
            return self._wait_for_server_ready()
        if not self._start_server():
            return False
        if self._wait_for_server_ready():
            return True
        self._stop_server()
        if self.enable_alerts:
            cast(Any, rumps).alert(
                "latentscore",
                "LatentScore UI failed to start. Try again or run it from the terminal for details.",
            )
        return False

    def _start_server(self) -> bool:
        port = self._server_port or self._find_free_port()
        self._server_port = port
        cmd = self._server_command(port)
        if cmd is None:
            if self.enable_alerts:
                cast(Any, rumps).alert(
                    "latentscore",
                    "Textual web server is missing. Install textual-serve and try again.",
                )
            return False
        log_path = self._server_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=self._server_env(),
        )
        self._server_log_file = log_file
        return True

    def _server_command(self, port: int) -> list[str] | None:
        if importlib.util.find_spec("textual_serve") is None:
            return None
        return [
            sys.executable,
            "-m",
            "latentscore.textual_serve_runner",
            "--host",
            self.server_host,
            "--port",
            str(port),
            "--app",
            self.textual_target,
        ]

    def _server_log_path(self) -> Path:
        if self._app_support_dir:
            return Path(self._app_support_dir) / "menubar-server.log"
        return Path.home() / "Library" / "Logs" / "LatentScore" / "menubar-server.log"

    def _server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        pythonpath = env.get("PYTHONPATH", "")
        paths = [path for path in pythonpath.split(os.pathsep) if path]
        if source_root not in paths:
            paths.insert(0, source_root)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def _open_webview(self, url: str) -> bool:
        if self._webview_proc and self._webview_proc.poll() is None:
            return True
        cmd = self._webview_command(url)
        if cmd is None:
            if self.enable_alerts:
                cast(Any, rumps).alert(
                    "latentscore",
                    "Native window requires pywebview. Install it to avoid opening a browser.",
                )
            return False
        self._webview_proc = subprocess.Popen(cmd, env=self._server_env())
        return True

    def _webview_command(self, url: str) -> list[str] | None:
        if importlib.util.find_spec("webview") is None:
            return None
        cmd = [
            sys.executable,
            "-m",
            "latentscore.webview_app",
            "--url",
            url,
            "--title",
            self.window_title,
            "--width",
            str(self.window_width),
            "--height",
            str(self.window_height),
        ]
        cmd.append("--resizable" if self.window_resizable else "--no-resizable")
        cmd.append("--frameless" if self.window_frameless else "--no-frameless")
        cmd.append("--easy-drag" if self.window_easy_drag else "--no-easy-drag")
        return cmd

    def _wait_for_server_ready(self) -> bool:
        if self._server_port is None:
            return False
        if self.server_ready_timeout <= 0:
            return True
        deadline = time.monotonic() + self.server_ready_timeout
        while time.monotonic() < deadline:
            if self._server_proc and self._server_proc.poll() is not None:
                return False
            try:
                with socket.create_connection(
                    (self.server_host, self._server_port),
                    timeout=min(self.server_ready_interval, 0.5),
                ):
                    return True
            except OSError:
                time.sleep(self.server_ready_interval)
        return False

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.server_host, 0))
            return sock.getsockname()[1]

    def _stop_server(self) -> None:
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
        if self._server_log_file:
            self._server_log_file.close()
            self._server_log_file = None

    def _stop_webview(self) -> None:
        if self._webview_proc and self._webview_proc.poll() is None:
            self._webview_proc.terminate()


def run_menu_bar() -> None:
    require_macos()
    MenuBarApp().run()  # type: ignore[call-arg]


if __name__ == "__main__":
    run_menu_bar()
