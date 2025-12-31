import pytest
from textual.color import Color

from app.logging_utils import LOG_DIR_ENV
from app.textual_app import SampleApp
from app.tui_base import HeaderControl


@pytest.mark.asyncio
async def test_sample_app_shows_traffic_lights(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    async with SampleApp().run_test() as pilot:
        assert pilot.app.query_one("#tl-close", HeaderControl)
        assert pilot.app.query_one("#tl-minimize", HeaderControl)
        assert pilot.app.query_one("#tl-menu", HeaderControl)


@pytest.mark.asyncio
async def test_sample_app_traffic_lights_do_not_overlap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    async with SampleApp().run_test() as pilot:
        close = pilot.app.query_one("#tl-close", HeaderControl)
        minimize = pilot.app.query_one("#tl-minimize", HeaderControl)
        menu = pilot.app.query_one("#tl-menu", HeaderControl)
        assert close.region.x < minimize.region.x < menu.region.x


@pytest.mark.asyncio
async def test_sample_app_traffic_light_icons(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    async with SampleApp().run_test() as pilot:
        close = pilot.app.query_one("#tl-close", HeaderControl)
        minimize = pilot.app.query_one("#tl-minimize", HeaderControl)
        menu = pilot.app.query_one("#tl-menu", HeaderControl)
        assert close.render() == "\u2715"
        assert minimize.render() == "\u2212"
        assert menu.render() == "\u25cf"


@pytest.mark.asyncio
async def test_sample_app_traffic_light_colors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    async with SampleApp().run_test() as pilot:
        close = pilot.app.query_one("#tl-close", HeaderControl)
        minimize = pilot.app.query_one("#tl-minimize", HeaderControl)
        menu = pilot.app.query_one("#tl-menu", HeaderControl)
        assert close.styles.color == Color.parse("#ff5f57")
        assert minimize.styles.color == Color.parse("#febc2e")
        assert menu.styles.color == Color.parse("#28c840")
