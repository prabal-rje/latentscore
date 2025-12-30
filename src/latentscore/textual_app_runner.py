from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from typing import Any, TypeGuard

from textual.app import App


def resolve_app(spec: str) -> App[Any]:
    module_name, sep, attr_name = spec.partition(":")
    if not module_name or not sep or not attr_name:
        raise ValueError("App spec must be in the form module:attribute")
    module = importlib.import_module(module_name)
    target: object = getattr(module, attr_name)
    app: App[Any] | None = None

    def _is_app(value: object) -> TypeGuard[App[Any]]:
        return isinstance(value, App)

    def _is_app_type(value: object) -> TypeGuard[type[App[Any]]]:
        return isinstance(value, type) and issubclass(value, App)

    def _is_factory(value: object) -> TypeGuard[Callable[[], object]]:
        return callable(value)

    if _is_app(target):
        app = target
    elif _is_app_type(target):
        app = target()
    elif _is_factory(target):
        candidate = target()
        if _is_app(candidate):
            app = candidate
    if app is None:
        raise TypeError("App spec did not resolve to a Textual App")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Textual App from a spec.")
    parser.add_argument("--app", required=True, help="App spec like package.module:AppClass")
    args = parser.parse_args(argv)
    app = resolve_app(args.app)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
