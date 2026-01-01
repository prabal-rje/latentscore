import importlib
from types import SimpleNamespace

import pytest

from app.copy_hints import COPY_HINT_MESSAGE
from app.diagnostics_tui import DiagnosticsApp, DiagnosticsHeader
from app.tui_base import HeaderControl


def test_copy_text_to_clipboard_uses_pyperclip(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)

    calls: dict[str, str] = {}

    def fake_copy(text: str) -> None:
        calls["text"] = text

    fake_module = SimpleNamespace(copy=fake_copy)

    def fake_import(name: str) -> object:
        assert name == "pyperclip"
        return fake_module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert app._copy_text_to_clipboard("hello") is True
    assert calls["text"] == "hello"


def test_copy_text_to_clipboard_empty_returns_false(tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)

    assert app._copy_text_to_clipboard("") is False


def test_copy_text_to_clipboard_requires_pyperclip(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)

    def fake_import(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert app._copy_text_to_clipboard("hello") is False


def test_action_copy_selection_notifies_on_empty(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)
    messages: list[str] = []

    def fake_notify(message: str, *_args, **_kwargs) -> None:
        messages.append(message)

    monkeypatch.setattr(app, "notify", fake_notify)
    monkeypatch.setattr(app, "_selected_text", lambda: None)

    app.action_copy_selection()

    assert messages == ["No text selected to copy."]


def test_action_copy_selection_notifies_on_success(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)
    messages: list[str] = []

    def fake_notify(message: str, *_args, **_kwargs) -> None:
        messages.append(message)

    monkeypatch.setattr(app, "notify", fake_notify)
    monkeypatch.setattr(app, "_selected_text", lambda: "hello")
    monkeypatch.setattr(app, "_copy_text_to_clipboard", lambda _text: True)

    app.action_copy_selection()

    assert messages == ["Copied to clipboard."]


def test_action_copy_selection_notifies_on_failure(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)
    messages: list[str] = []

    def fake_notify(message: str, *_args, **_kwargs) -> None:
        messages.append(message)

    monkeypatch.setattr(app, "notify", fake_notify)
    monkeypatch.setattr(app, "_selected_text", lambda: "hello")
    monkeypatch.setattr(app, "_copy_text_to_clipboard", lambda _text: False)

    app.action_copy_selection()

    assert messages == ["Clipboard copy failed; ensure pyperclip is installed."]


def test_action_copy_hint_notifies(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)
    messages: list[str] = []

    def fake_notify(message: str, *_args, **_kwargs) -> None:
        messages.append(message)

    monkeypatch.setattr(app, "notify", fake_notify)

    app.action_copy_hint()

    assert messages == [COPY_HINT_MESSAGE]


@pytest.mark.asyncio
async def test_diagnostics_tui_shows_traffic_lights(tmp_path) -> None:
    async with DiagnosticsApp(tmp_path).run_test() as pilot:
        assert pilot.app.query_one("#tl-close", HeaderControl)
        assert pilot.app.query_one("#tl-minimize", HeaderControl)
        assert pilot.app.query_one("#tl-menu", HeaderControl)


@pytest.mark.asyncio
async def test_diagnostics_header_click_does_not_show_help(tmp_path) -> None:
    async with DiagnosticsApp(tmp_path).run_test() as pilot:
        help_widget = pilot.app.query_one("#help")
        assert "visible" not in help_widget.classes
        header = pilot.app.query_one(DiagnosticsHeader)
        await pilot.click(header, offset=(1, 0))
        assert "visible" not in help_widget.classes
