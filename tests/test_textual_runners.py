from app import textual_app, textual_app_runner
from app.textual_app_runner import resolve_app
from app.textual_serve_runner import build_command


def test_resolve_app_from_class() -> None:
    app = resolve_app(textual_app.APP_SPEC)
    assert app.__class__.__name__ == textual_app.SampleApp.__name__


def test_build_command_contains_runner() -> None:
    cmd = build_command(textual_app.APP_SPEC)
    assert textual_app_runner.__name__ in cmd
