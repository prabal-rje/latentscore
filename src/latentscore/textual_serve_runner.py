from __future__ import annotations

import argparse
import shlex
import sys

from textual_serve.server import Server


def build_command(app_spec: str) -> str:
    args = [sys.executable, "-m", "latentscore.textual_app_runner", "--app", app_spec]
    return shlex.join(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a Textual app over the web.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--app", required=True, help="App spec like package.module:AppClass")
    args = parser.parse_args(argv)

    command = build_command(args.app)
    Server(command=command, host=args.host, port=args.port).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
