from latentscore.textual_app_runner import resolve_app
from latentscore.textual_serve_runner import build_command


def test_resolve_app_from_class() -> None:
    app = resolve_app("latentscore.textual_app:LatentScoreApp")
    assert app.__class__.__name__ == "LatentScoreApp"


def test_build_command_contains_runner() -> None:
    cmd = build_command("latentscore.textual_app:LatentScoreApp")
    assert "latentscore.textual_app_runner" in cmd
