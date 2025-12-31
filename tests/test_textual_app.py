import pytest

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
