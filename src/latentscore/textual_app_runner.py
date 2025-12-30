from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from textual.app import App


def resolve_app(spec: str) -> App[Any]:
    module_name, sep, attr_name = spec.partition(":")
    if not module_name or not sep or not attr_name:
        raise ValueError("App spec must be in the form module:attribute")
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if isinstance(target, App):
        return target
    if isinstance(target, type) and issubclass(target, App):
        return target()
    if callable(target):
        app = target()
        if isinstance(app, App):
            return app
    raise TypeError("App spec did not resolve to a Textual App")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Textual App from a spec.")
    parser.add_argument("--app", required=True, help="App spec like package.module:AppClass")
    args = parser.parse_args(argv)
    app = resolve_app(args.app)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
