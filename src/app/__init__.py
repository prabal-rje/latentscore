from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .app import demo_run, main
from .loop import install_uvloop_policy, run
from .tui import HelloWorldApp, run_tui

if TYPE_CHECKING:
    from .menubar import MenuBarApp, run_menu_bar

__all__ = [
    "demo_run",
    "HelloWorldApp",
    "install_uvloop_policy",
    "main",
    "MenuBarApp",
    "run",
    "run_menu_bar",
    "run_tui",
]


def __getattr__(name: str) -> object:
    if name in {"MenuBarApp", "run_menu_bar"}:
        module = import_module(".menubar", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
