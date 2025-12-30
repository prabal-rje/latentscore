from __future__ import annotations

import argparse
import platform
from importlib.util import find_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a webview window.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--title", default="LatentScore")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument(
        "--resizable",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Allow the window to be resized.",
    )
    parser.add_argument(
        "--frameless",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Render the window without the native frame.",
    )
    parser.add_argument(
        "--easy-drag",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable easy drag mode for frameless windows.",
    )
    args = parser.parse_args(argv)

    import webview
    from webview.menu import Menu, MenuAction

    if platform.system() == "Darwin":
        _set_app_name(args.title)

    def show_yolo_dialog() -> None:
        if not webview.windows:
            return
        webview.windows[0].evaluate_js("alert('YOLO Test')")

    menu = [Menu("YOLO Test", [MenuAction("Show dialog", show_yolo_dialog)])]
    webview.create_window(
        args.title,
        args.url,
        width=args.width,
        height=args.height,
        resizable=args.resizable,
        frameless=args.frameless,
        easy_drag=args.easy_drag,
        menu=menu,
    )
    webview.start()
    return 0


def _set_app_name(title: str) -> None:
    if find_spec("Foundation") is None or find_spec("webview.platforms.cocoa") is None:
        return

    from Foundation import NSProcessInfo
    from webview.platforms import cocoa

    NSProcessInfo.processInfo().setProcessName_(title)
    cocoa.info["CFBundleName"] = title


if __name__ == "__main__":
    raise SystemExit(main())
