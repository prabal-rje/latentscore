from __future__ import annotations

import atexit
import importlib.util
import os
import platform
import signal
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

from .logging_utils import LOG_DIR_ENV, default_log_dir, log_path

GREETING_TITLE = "Say hi"
GREETING_MESSAGE = "Hi there!"
OPEN_UI_TITLE = "Open LatentScore"
OPEN_LOGS_TITLE = "Open Logs Folder"
SEE_DIAGNOSTICS_TITLE = "See Diagnostics"


def _is_apple_silicon_hardware() -> bool:
    """Check if running on Apple Silicon hardware (works even under Rosetta)."""
    result = subprocess.run(
        ["sysctl", "-n", "hw.optional.arm64"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


def require_apple_silicon() -> None:
    """Ensure we're running on an Apple Silicon Mac (M1/M2/M3/M4, etc.)."""
    if platform.system() != "Darwin":
        raise RuntimeError("The menu bar helper only runs on macOS.")
    if not _is_apple_silicon_hardware():
        raise RuntimeError(
            "The menu bar helper requires Apple Silicon hardware (M1/M2/M3/M4, etc.)."
        )


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
        self._diagnostics_proc: subprocess.Popen[bytes] | None = None
        self._diagnostics_log_file: IO[bytes] | None = None
        self._diagnostics_webview_proc: subprocess.Popen[bytes] | None = None
        self._diagnostics_port: Optional[int] = None
        if not initialize:
            self.hi_item = SimpleNamespace(title=GREETING_TITLE)
            self.open_item = SimpleNamespace(title=OPEN_UI_TITLE)
            self.logs_item = SimpleNamespace(title=OPEN_LOGS_TITLE)
            self.diagnostics_item = SimpleNamespace(title=SEE_DIAGNOSTICS_TITLE)
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
        self._register_signal_handlers()
        atexit.register(self._stop_server)
        atexit.register(self._stop_webview)
        atexit.register(self._stop_diagnostics_server)
        atexit.register(self._stop_diagnostics_webview)
        self.hi_item = cast(Any, rumps).MenuItem(
            GREETING_TITLE, callback=self._on_hi_clicked
        )
        self.open_item = cast(Any, rumps).MenuItem(
            OPEN_UI_TITLE, callback=self._on_open_clicked
        )
        self.logs_item = cast(Any, rumps).MenuItem(
            OPEN_LOGS_TITLE, callback=self._on_open_logs_clicked
        )
        self.diagnostics_item = cast(Any, rumps).MenuItem(
            SEE_DIAGNOSTICS_TITLE, callback=self._on_diagnostics_clicked
        )
        self.menu = [
            self.open_item,
            self.logs_item,
            self.diagnostics_item,
            self.hi_item,
        ]
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

    def _on_open_logs_clicked(self, _sender: object) -> str:
        log_dir = self._log_dir()
        env_hint = f"(override with ${LOG_DIR_ENV})"
        if log_dir.exists():
            subprocess.Popen(["open", str(log_dir)])
        elif self.enable_alerts:
            cast(Any, rumps).alert(
                "latentscore",
                f"Logs directory not found yet: {log_dir}\n{env_hint}",
            )
        return str(log_dir)

    def _on_diagnostics_clicked(self, _sender: object) -> str:
        if not self._ensure_diagnostics_server():
            if self.enable_alerts:
                cast(Any, rumps).alert(
                    "latentscore",
                    "Diagnostics viewer unavailable. Install textual-serve to open logs.",
                )
            return ""
        url = self._diagnostics_url()
        if not self._open_diagnostics_webview(url):
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
            start_new_session=True,
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
        return log_path("menubar-server.log", self._app_support_dir)

    def _log_dir(self) -> Path:
        if self._app_support_dir:
            return Path(self._app_support_dir)
        return default_log_dir()

    def _server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if LOG_DIR_ENV not in env:
            env[LOG_DIR_ENV] = str(self._log_dir())
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
        self._webview_proc = subprocess.Popen(
            cmd, env=self._server_env(), start_new_session=True
        )
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

    def _diagnostics_url(self) -> str:
        port = self._diagnostics_port or 0
        return f"http://{self.server_host}:{port}"

    def _ensure_diagnostics_server(self) -> bool:
        if self._diagnostics_proc and self._diagnostics_proc.poll() is None:
            return self._wait_for_diagnostics_ready()
        if not self._start_diagnostics_server():
            return False
        if self._wait_for_diagnostics_ready():
            return True
        self._stop_diagnostics_server()
        return False

    def _wait_for_diagnostics_ready(self) -> bool:
        if self._diagnostics_port is None:
            return False
        if self.server_ready_timeout <= 0:
            return True
        deadline = time.monotonic() + self.server_ready_timeout
        while time.monotonic() < deadline:
            if self._diagnostics_proc and self._diagnostics_proc.poll() is not None:
                return False
            try:
                with socket.create_connection(
                    (self.server_host, self._diagnostics_port),
                    timeout=min(self.server_ready_interval, 0.5),
                ):
                    return True
            except OSError:
                time.sleep(self.server_ready_interval)
        return False

    def _start_diagnostics_server(self) -> bool:
        port = self._diagnostics_port or self._find_free_port()
        self._diagnostics_port = port
        cmd = self._diagnostics_server_command(port)
        if cmd is None:
            return False
        log_path = self._diagnostics_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")
        self._diagnostics_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=self._server_env(),
            start_new_session=True,
        )
        self._diagnostics_log_file = log_file
        return True

    def _diagnostics_server_command(self, port: int) -> list[str] | None:
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
            "latentscore.diagnostics_tui:build_app",
        ]

    def _diagnostics_log_path(self) -> Path:
        return log_path("diagnostics-server.log", self._app_support_dir)

    def _open_diagnostics_webview(self, url: str) -> bool:
        if (
            self._diagnostics_webview_proc
            and self._diagnostics_webview_proc.poll() is None
        ):
            return True
        cmd = self._diagnostics_webview_command(url)
        if cmd is None:
            return False
        self._diagnostics_webview_proc = subprocess.Popen(
            cmd, env=self._server_env(), start_new_session=True
        )
        return True

    def _diagnostics_webview_command(self, url: str) -> list[str] | None:
        if importlib.util.find_spec("webview") is None:
            return None
        cmd = [
            sys.executable,
            "-m",
            "latentscore.webview_app",
            "--url",
            url,
            "--title",
            "LatentScore Diagnostics",
            "--width",
            str(self.window_width),
            "--height",
            str(self.window_height),
        ]
        cmd.append("--resizable" if self.window_resizable else "--no-resizable")
        cmd.append("--frameless" if self.window_frameless else "--no-frameless")
        cmd.append("--easy-drag" if self.window_easy_drag else "--no-easy-drag")
        return cmd

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.server_host, 0))
            return sock.getsockname()[1]

    def _stop_server(self) -> None:
        if self._server_proc and self._server_proc.poll() is None:
            self._terminate_process_group(self._server_proc)
        if self._server_log_file:
            self._server_log_file.close()
            self._server_log_file = None

    def _stop_webview(self) -> None:
        if self._webview_proc and self._webview_proc.poll() is None:
            self._terminate_process_group(self._webview_proc)

    def _stop_diagnostics_server(self) -> None:
        if self._diagnostics_proc and self._diagnostics_proc.poll() is None:
            self._terminate_process_group(self._diagnostics_proc)
        if self._diagnostics_log_file:
            self._diagnostics_log_file.close()
            self._diagnostics_log_file = None

    def _stop_diagnostics_webview(self) -> None:
        if (
            self._diagnostics_webview_proc
            and self._diagnostics_webview_proc.poll() is None
        ):
            self._terminate_process_group(self._diagnostics_webview_proc)

    def _terminate_process_group(self, proc: subprocess.Popen[bytes]) -> None:
        if os.name == "nt" or not hasattr(os, "killpg"):
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return

    def _register_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, _signum: int, _frame: object | None) -> None:
        self._stop_diagnostics_webview()
        self._stop_diagnostics_server()
        self._stop_webview()
        self._stop_server()
        raise SystemExit(0)

def run_menu_bar() -> None:
    require_apple_silicon()
    MenuBarApp().run()  # type: ignore[call-arg]


if __name__ == "__main__":
    run_menu_bar()
