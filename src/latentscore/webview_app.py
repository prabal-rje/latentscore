from __future__ import annotations

import argparse
import importlib
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

    webview = importlib.import_module("webview")
    menu_module = importlib.import_module("webview.menu")
    menu_cls = getattr(menu_module, "Menu", None)
    menu_action_cls = getattr(menu_module, "MenuAction", None)
    assert menu_cls is not None
    assert menu_action_cls is not None

    if platform.system() == "Darwin":
        _set_app_name(args.title)

    def show_yolo_dialog() -> None:
        windows = getattr(webview, "windows", None)
        if not windows:
            return
        window = windows[0]
        evaluate_js = getattr(window, "evaluate_js", None)
        assert callable(evaluate_js)
        evaluate_js("alert('YOLO Test')")

    menu = [menu_cls("YOLO Test", [menu_action_cls("Show dialog", show_yolo_dialog)])]
    create_window = getattr(webview, "create_window", None)
    assert callable(create_window)
    create_window(
        args.title,
        args.url,
        width=args.width,
        height=args.height,
        resizable=args.resizable,
        frameless=args.frameless,
        easy_drag=args.easy_drag,
        menu=menu,
    )
    start = getattr(webview, "start", None)
    assert callable(start)
    start()
    return 0


def _set_app_name(title: str) -> None:
    if find_spec("Foundation") is None or find_spec("webview.platforms.cocoa") is None:
        return

    foundation = importlib.import_module("Foundation")
    cocoa = importlib.import_module("webview.platforms.cocoa")
    ns_process_info = getattr(foundation, "NSProcessInfo", None)
    assert ns_process_info is not None
    process_info = ns_process_info.processInfo()
    set_name = getattr(process_info, "setProcessName_", None)
    assert callable(set_name)
    set_name(title)
    info = getattr(cocoa, "info", None)
    if info is None:
        return
    setter = getattr(info, "__setitem__", None)
    if not callable(setter):
        return
    setter("CFBundleName", title)


if __name__ == "__main__":
    raise SystemExit(main())
