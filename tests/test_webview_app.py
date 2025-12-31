from app.copy_hints import COPY_HINT_MESSAGE
from app.webview_app import _copy_hint_script, _finish_splash, _splash_html


def test_copy_hint_script_includes_message() -> None:
    script = _copy_hint_script(COPY_HINT_MESSAGE)

    assert COPY_HINT_MESSAGE in script
    assert "diagnostics-copy-hint" in script


def test_splash_html_includes_arcade_font_stack() -> None:
    html = _splash_html()

    assert "LOADING..." in html
    assert "pixel-text" in html
    assert "Press Start 2P" in html
    assert "font-size: 22px" in html
    assert "text-shadow" not in html
    assert "white-space: nowrap" in html


def test_finish_splash_shows_main_and_destroys_splash(monkeypatch) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    class Window:
        def show(self) -> None:
            calls.append("show")

        def destroy(self) -> None:
            calls.append("destroy")

    def fake_monotonic() -> float:
        return 5.0

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.webview_app.time.monotonic", fake_monotonic)
    monkeypatch.setattr("app.webview_app.time.sleep", fake_sleep)
    _finish_splash(Window(), Window(), delay=2.0, started_at=3.5)

    assert sleeps == [0.5]
    assert calls == ["destroy", "show"]
