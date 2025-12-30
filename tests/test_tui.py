import pytest
from textual.widgets import Static

from latentscore.tui import HelloWorldApp


@pytest.mark.asyncio
async def test_hello_world_app_renders_greeting() -> None:
    async with HelloWorldApp().run_test() as pilot:
        greeting = pilot.app.query_one("#greeting", Static)
        rendered = greeting.render()
        assert "Hello world!" in str(rendered)
