import platform
import sys

import pytest

from app import textual_serve_runner, webview_app
from app.branding import APP_NAME
from app.menubar import (
    GREETING_MESSAGE,
    GREETING_TITLE,
    OPEN_LOGS_TITLE,
    OPEN_UI_TITLE,
    QUIT_TITLE,
    SEE_DIAGNOSTICS_TITLE,
    MenuBarApp,
    require_apple_silicon,
)


def test_greeting_message_constant() -> None:
    assert GREETING_MESSAGE == "Hi there!"
    assert GREETING_TITLE == "Say hi"
    assert OPEN_UI_TITLE == f"Open {APP_NAME}"
    assert OPEN_LOGS_TITLE == "Open Logs Folder"
    assert SEE_DIAGNOSTICS_TITLE == "See Diagnostics"
    assert QUIT_TITLE == "Quit"


def test_require_apple_silicon() -> None:
    """Test that the function enforces macOS + Apple Silicon hardware."""
    from app.menubar import _is_apple_silicon_hardware

    is_macos = platform.system() == "Darwin"
    is_apple_silicon = is_macos and _is_apple_silicon_hardware()

    if is_apple_silicon:
        require_apple_silicon()  # Should pass
    else:
        with pytest.raises(RuntimeError):
            require_apple_silicon()


def test_menu_bar_app_sets_menu_item_title(tmp_path) -> None:
    app = MenuBarApp(enable_alerts=False, app_support_dir=str(tmp_path), initialize=False)
    assert app.hi_item.title == GREETING_TITLE
    assert app._on_hi_clicked(None) == GREETING_MESSAGE
    assert app.open_item.title == OPEN_UI_TITLE
    assert app.logs_item.title == OPEN_LOGS_TITLE
    assert app.diagnostics_item.title == SEE_DIAGNOSTICS_TITLE
    assert app.quit_item.title == QUIT_TITLE


def test_on_open_starts_server_and_opens_browser(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, initialize=False, server_ready_timeout=0.0)
    started = {}

    def fake_start_server() -> bool:
        started["ok"] = True
        app._server_port = 4242
        return True

    opened = {}
    app._start_server = fake_start_server  # type: ignore[assignment]

    def fake_open_webview(_self, url: str) -> bool:
        opened["url"] = url
        return True

    monkeypatch.setattr("app.menubar.MenuBarApp._open_webview", fake_open_webview)

    url = app._on_open_clicked(None)
    assert started["ok"] is True
    assert opened["url"] == "http://127.0.0.1:4242"
    assert url == opened["url"]


def test_server_disabled_skips_start(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, server_enabled=False, initialize=False)

    def fail() -> None:
        raise AssertionError("server should not start when disabled")

    app._start_server = fail  # type: ignore[assignment]
    app._ensure_server()


def test_server_command_uses_runner(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, initialize=False)

    monkeypatch.setattr("app.menubar.importlib.util.find_spec", lambda name: object())
    cmd = app._server_command(4242)
    assert cmd is not None
    assert cmd[:3] == [sys.executable, "-m", textual_serve_runner.__name__]
    assert "--app" in cmd


def test_webview_command_includes_window_args(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, initialize=False)

    monkeypatch.setattr("app.menubar.importlib.util.find_spec", lambda name: object())
    cmd = app._webview_command("http://127.0.0.1:4242")
    assert cmd is not None
    assert cmd[:3] == [sys.executable, "-m", webview_app.__name__]
    assert "--url" in cmd
    assert "--title" in cmd
    assert "--minimize-signal" in cmd
    assert "--frameless" in cmd
    assert "--easy-drag" in cmd


def test_diagnostics_webview_command_includes_screen_fraction(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, initialize=False)

    monkeypatch.setattr("app.menubar.importlib.util.find_spec", lambda name: object())
    cmd = app._diagnostics_webview_command("http://127.0.0.1:4242")
    assert cmd is not None
    assert "--screen-fraction" in cmd
    fraction_index = cmd.index("--screen-fraction") + 1
    assert cmd[fraction_index] == "0.75"
    assert "--minimize-signal" in cmd
    assert "--resizable" in cmd
    assert "--frameless" in cmd
    assert "--easy-drag" in cmd


def test_diagnostics_webview_command_respects_easy_drag(monkeypatch) -> None:
    app = MenuBarApp(
        enable_alerts=False,
        initialize=False,
        window_easy_drag=False,
    )

    monkeypatch.setattr("app.menubar.importlib.util.find_spec", lambda name: object())
    cmd = app._diagnostics_webview_command("http://127.0.0.1:4242")
    assert cmd is not None
    assert "--no-easy-drag" in cmd


def test_quit_callback_invokes_shutdown(monkeypatch) -> None:
    app = MenuBarApp(enable_alerts=False, initialize=False)
    called = {}

    def fake_shutdown() -> None:
        called["shutdown"] = True

    def fake_quit_application() -> None:
        called["quit"] = True

    app._shutdown = fake_shutdown  # type: ignore[assignment]
    monkeypatch.setattr("app.menubar.rumps.quit_application", fake_quit_application)

    app._on_quit_clicked(None)
    assert called == {"shutdown": True, "quit": True}
