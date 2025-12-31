from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import threading
import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

from .branding import APP_NAME
from .copy_hints import COPY_HINT_MESSAGE
from .logging_utils import LOG_DIR_ENV, default_log_dir
from .parent_watch import start_parent_watchdog_from_env


def main(argv: list[str] | None = None) -> int:
    start_parent_watchdog_from_env()
    parser = argparse.ArgumentParser(description="Open a webview window.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--title", default=APP_NAME)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument(
        "--screen-fraction",
        type=float,
        default=None,
        help="Size the window to a fraction of the primary screen (0 < fraction <= 1).",
    )
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
    parser.add_argument(
        "--splash-delay",
        type=float,
        default=SPLASH_DELAY_SECONDS,
        help="Seconds to show the splash page before loading the URL.",
    )
    parser.add_argument(
        "--minimize-signal",
        default=None,
        help="Filename or path used to request a minimize action.",
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

    main_window: Any | None = None

    def show_yolo_dialog() -> None:
        if main_window is None:
            return
        evaluate_js = getattr(main_window, "evaluate_js", None)
        if not callable(evaluate_js):
            return
        evaluate_js("alert('YOLO Test')")

    menu = [menu_cls("YOLO Test", [menu_action_cls("Show dialog", show_yolo_dialog)])]
    width, height = _resolve_window_size(args, webview)

    create_window = getattr(webview, "create_window", None)
    assert callable(create_window)
    use_splash = args.splash_delay > 0
    minimize_signal = _resolve_signal_path(args.minimize_signal)
    started_at = time.monotonic() if use_splash else 0.0
    main_window = create_window(
        args.title,
        args.url,
        width=width,
        height=height,
        resizable=args.resizable,
        frameless=args.frameless,
        easy_drag=args.easy_drag,
        menu=menu,
        hidden=use_splash,
    )
    if main_window is None:
        return 1
    splash_window = None
    if use_splash:
        splash_window = create_window(
            SPLASH_TITLE,
            html=_splash_html(),
            width=SPLASH_WIDTH,
            height=SPLASH_HEIGHT,
            resizable=False,
            frameless=True,
            easy_drag=False,
            transparent=False,
            on_top=True,
            focus=False,
            menu=[],
            background_color=SPLASH_BACKGROUND_COLOR,
        )
    _install_macos_window_style(main_window, background=None)
    if splash_window is not None:
        _install_macos_window_style(splash_window, background=SPLASH_BACKGROUND_COLOR)
    _install_minimize_watch(main_window, minimize_signal)
    _install_copy_hint(main_window, COPY_HINT_MESSAGE)
    if use_splash:
        _install_main_ready_handler(
            main_window,
            splash_window,
            delay=args.splash_delay,
            started_at=started_at,
        )
    start = getattr(webview, "start", None)
    assert callable(start)
    start()
    return 0


def _resolve_window_size(args: argparse.Namespace, webview_module: Any) -> tuple[int, int]:
    width = args.width
    height = args.height
    fraction = args.screen_fraction
    if fraction is None:
        return width, height
    if not 0 < fraction <= 1:
        raise SystemExit("--screen-fraction must be between 0 and 1")
    screens = getattr(webview_module, "screens", None)
    if not screens:
        return width, height
    screen = screens[0]
    screen_width = getattr(screen, "width", None)
    screen_height = getattr(screen, "height", None)
    if not isinstance(screen_width, int) or not isinstance(screen_height, int):
        return width, height
    return max(1, int(screen_width * fraction)), max(1, int(screen_height * fraction))


def _resolve_signal_path(signal: str | None) -> Path | None:
    if not signal:
        return None
    path = Path(signal)
    if path.is_absolute():
        return path
    base = Path(os.environ.get(LOG_DIR_ENV, str(default_log_dir()))).expanduser()
    return base / path


def _install_minimize_watch(window: Any, signal_path: Path | None) -> None:
    if window is None or signal_path is None:
        return

    def _watch() -> None:
        while True:
            if signal_path.exists():
                try:
                    signal_path.unlink()
                except OSError:
                    pass
                _minimize_window(window)
            time.sleep(0.2)

    threading.Thread(target=_watch, daemon=True).start()


def _minimize_window(window: Any) -> None:
    minimize = getattr(window, "minimize", None)
    if callable(minimize):
        minimize()
        return
    hide = getattr(window, "hide", None)
    if callable(hide):
        hide()


def _install_macos_window_style(window: Any, *, background: str | None) -> None:
    if platform.system() != "Darwin":
        return
    events = getattr(window, "events", None)
    if events is None:
        return
    before_show = getattr(events, "before_show", None)
    if before_show is None:
        return

    def _on_before_show(window: Any) -> None:
        _configure_macos_window(window, background)

    before_show += _on_before_show
    if before_show.is_set():
        _on_before_show(window)


def _configure_macos_window(window: Any, background: str | None) -> None:
    if find_spec("AppKit") is None:
        return
    appkit = importlib.import_module("AppKit")
    native = getattr(window, "native", None)
    if native is None:
        return
    # Only apply aggressive styling to splash window (with background color).
    # Main window (background=None) keeps native titlebar with window controls.
    if not background:
        return
    try:
        native.setHasShadow_(False)
    except Exception:
        pass
    try:
        native.setOpaque_(True)
    except Exception:
        pass
    try:
        native.setBackgroundColor_(_nscolor_from_hex(background, appkit))
    except Exception:
        pass
    content_view = getattr(native, "contentView", None)
    if not callable(content_view):
        return
    view = content_view()
    if view is None:
        return
    _style_view_layer(cast(Any, view), appkit, background)
    superview = getattr(view, "superview", None)
    if callable(superview):
        parent = superview()
        if parent is not None:
            _style_view_layer(cast(Any, parent), appkit, background)


def _style_view_layer(view: Any, appkit: Any, background: str | None) -> None:
    try:
        view.setWantsLayer_(True)
    except Exception:
        return
    layer = view.layer()
    if layer is None:
        return
    try:
        layer.setCornerRadius_(0.0)
        layer.setMasksToBounds_(True)
        layer.setBorderWidth_(0.0)
        if background:
            # Suppress PyObjC warning about CGColor pointer - it's expected
            # when bridging NSColor.CGColor() to Core Graphics layer APIs.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", message="PyObjCPointer created")
                layer.setBackgroundColor_(_nscolor_from_hex(background, appkit).CGColor())
    except Exception:
        return


def _nscolor_from_hex(value: str, appkit: Any) -> Any:
    text = value.lstrip("#")
    if len(text) != 6:
        return appkit.NSColor.blackColor()
    red = int(text[0:2], 16) / 255.0
    green = int(text[2:4], 16) / 255.0
    blue = int(text[4:6], 16) / 255.0
    return appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(red, green, blue, 1.0)


def _install_copy_hint(window: Any, message: str) -> None:
    events = getattr(window, "events", None)
    if events is None:
        return
    loaded = getattr(events, "loaded", None)
    if loaded is None:
        return

    def _on_loaded(window: Any) -> None:
        evaluate_js = getattr(window, "evaluate_js", None)
        if not callable(evaluate_js):
            return
        evaluate_js(_copy_hint_script(message))

    loaded += _on_loaded
    if loaded.is_set():
        _on_loaded(window)


def _install_main_ready_handler(
    main_window: Any,
    splash_window: Any,
    *,
    delay: float,
    started_at: float,
) -> None:
    if delay <= 0:
        _finish_splash(main_window, splash_window, delay=0.0, started_at=started_at)
        return
    events = getattr(main_window, "events", None)
    if events is None:
        _finish_splash(main_window, splash_window, delay=delay, started_at=started_at)
        return
    loaded = getattr(events, "loaded", None)
    if loaded is None:
        _finish_splash(main_window, splash_window, delay=delay, started_at=started_at)
        return
    finished = False

    def _on_loaded(window: Any) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        _finish_splash(window, splash_window, delay=delay, started_at=started_at)

    loaded += _on_loaded
    if loaded.is_set():
        _on_loaded(main_window)


def _finish_splash(
    main_window: Any,
    splash_window: Any,
    *,
    delay: float,
    started_at: float,
) -> None:
    remaining = delay - (time.monotonic() - started_at)
    if remaining > 0:
        time.sleep(remaining)
    _call_window_method(splash_window, "destroy")
    _call_window_method(main_window, "show")


def _call_window_method(window: Any, name: str) -> None:
    if window is None:
        return
    method = getattr(window, name, None)
    if callable(method):
        method()


SPLASH_DELAY_SECONDS = 2.0
SPLASH_TITLE = "LOADING..."
SPLASH_WIDTH = 320
SPLASH_HEIGHT = 80
SPLASH_BACKGROUND_COLOR = "#000000"
SPLASH_FONT_DATA = (
    Path(__file__)
    .with_name("press_start_2p.b64")
    .read_text(encoding="utf-8")
    .strip()
    .replace("\n", "")
)


def _splash_html() -> str:
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>LOADING...</title>
    <style>
      @font-face {{
        font-family: "Press Start 2P";
        font-style: normal;
        font-weight: 400;
        font-display: swap;
        src: url(data:font/ttf;base64,{SPLASH_FONT_DATA}) format("truetype");
      }}
      :root {{
        color-scheme: light;
      }}
      body {{
        margin: 0;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: {SPLASH_BACKGROUND_COLOR};
        font-family: "Press Start 2P", "VT323", "Silom", "Monaco", "Menlo",
          "Courier New", monospace;
      }}
      .pixel-text {{
        color: #ffffff;
        font-size: 22px;
        letter-spacing: 1px;
        line-height: 1;
        text-transform: uppercase;
        white-space: nowrap;
        word-break: keep-all;
        -webkit-font-smoothing: none;
        text-rendering: geometricPrecision;
      }}
    </style>
  </head>
  <body>
    <div class="pixel-text">LOADING...</div>
  </body>
</html>
"""


def _copy_hint_script(message: str) -> str:
    payload = json.dumps(message)
    return f"""
(() => {{
  if (window.__diagnosticsCopyHintInstalled) return;
  window.__diagnosticsCopyHintInstalled = true;
  const message = {payload};
  const styleId = "diagnostics-copy-hint-style";
  if (!document.getElementById(styleId)) {{
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      .diagnostics-copy-hint {{
        position: fixed;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(24, 24, 24, 0.9);
        color: #f5f5f5;
        padding: 8px 14px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 13px;
        letter-spacing: 0.2px;
        opacity: 0;
        transition: opacity 160ms ease-in-out;
        z-index: 2147483647;
        pointer-events: none;
      }}
      .diagnostics-copy-hint.show {{
        opacity: 1;
      }}
    `;
    document.head.appendChild(style);
  }}
  const showHint = () => {{
    let node = document.getElementById("diagnostics-copy-hint");
    if (!node) {{
      node = document.createElement("div");
      node.id = "diagnostics-copy-hint";
      node.className = "diagnostics-copy-hint";
      document.body.appendChild(node);
    }}
    node.textContent = message;
    node.classList.add("show");
    if (window.__diagnosticsCopyHintTimer) {{
      clearTimeout(window.__diagnosticsCopyHintTimer);
    }}
    window.__diagnosticsCopyHintTimer = setTimeout(() => {{
      node.classList.remove("show");
    }}, 2000);
  }};
  document.addEventListener("keydown", (event) => {{
    if (!event.metaKey) return;
    const key = (event.key || "").toLowerCase();
    if (key !== "c") return;
    showHint();
  }}, true);
}})();
"""


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
