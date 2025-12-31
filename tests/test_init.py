import app


def test_menubar_reexports_are_accessible() -> None:
    MenuBarApp = getattr(app, "MenuBarApp")
    run_menu_bar = getattr(app, "run_menu_bar")

    assert MenuBarApp.__name__ == "MenuBarApp"
    assert callable(run_menu_bar)
