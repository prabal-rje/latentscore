from app.diagnostics_tui import DiagnosticsApp


def test_copy_text_to_clipboard_uses_pyperclip(monkeypatch, tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)
    import pyperclip

    calls: dict[str, str] = {}

    def fake_copy(text: str) -> None:
        calls["text"] = text

    monkeypatch.setattr(pyperclip, "copy", fake_copy)

    assert app._copy_text_to_clipboard("hello") is True
    assert calls["text"] == "hello"


def test_copy_text_to_clipboard_empty_returns_false(tmp_path) -> None:
    app = DiagnosticsApp(tmp_path)

    assert app._copy_text_to_clipboard("") is False
