import latentscore


def test_menubar_reexports_are_accessible() -> None:
    MenuBarApp = getattr(latentscore, "MenuBarApp")
    run_menu_bar = getattr(latentscore, "run_menu_bar")

    assert MenuBarApp.__name__ == "MenuBarApp"
    assert callable(run_menu_bar)
