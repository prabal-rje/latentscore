from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import rumps  # type: ignore[import]
from rumps import utils as rumps_utils  # type: ignore[import]

GREETING_TITLE = "Say hi"
GREETING_MESSAGE = "Hi there!"


def require_macos() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("The menu bar helper only runs on macOS.")


class MenuBarApp(rumps.App):
    """Simple macOS status bar app with a single greeting button."""

    def __init__(
        self,
        enable_alerts: bool = True,
        app_support_dir: Optional[str] = None,
        initialize: bool = True,
    ) -> None:
        self.enable_alerts = enable_alerts
        if not initialize:
            self.hi_item = SimpleNamespace(title=GREETING_TITLE)
            return

        if app_support_dir:
            base = Path(app_support_dir)
            base.mkdir(parents=True, exist_ok=True)

            def _application_support(_name: str) -> str:
                base.mkdir(parents=True, exist_ok=True)
                return str(base)

            cast(Any, rumps).application_support = _application_support
            cast(Any, rumps_utils).application_support = _application_support

        super().__init__("Hi")  # type: ignore[misc]
        self.hi_item = cast(Any, rumps).MenuItem(
            GREETING_TITLE, callback=self._on_hi_clicked
        )
        self.menu = [self.hi_item]

    def _on_hi_clicked(self, _sender: object) -> str:
        if self.enable_alerts:
            cast(Any, rumps).alert("latentscore", GREETING_MESSAGE)
        return GREETING_MESSAGE


def run_menu_bar() -> None:
    require_macos()
    MenuBarApp().run()  # type: ignore[call-arg]


if __name__ == "__main__":
    run_menu_bar()
