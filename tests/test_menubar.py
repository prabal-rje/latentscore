import platform

import pytest

from latentscore.menubar import (
    GREETING_MESSAGE,
    GREETING_TITLE,
    MenuBarApp,
    require_macos,
)


def test_greeting_message_constant() -> None:
    assert GREETING_MESSAGE == "Hi there!"
    assert GREETING_TITLE == "Say hi"


def test_require_macos() -> None:
    if platform.system() == "Darwin":
        require_macos()
    else:
        with pytest.raises(RuntimeError):
            require_macos()


def test_menu_bar_app_sets_menu_item_title(tmp_path) -> None:
    app = MenuBarApp(enable_alerts=False, app_support_dir=str(tmp_path), initialize=False)
    assert app.hi_item.title == GREETING_TITLE
    assert app._on_hi_clicked(None) == GREETING_MESSAGE
