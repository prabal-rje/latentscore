# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Notebook display helpers for LatentScore.

Import in a Jupyter / Colab notebook to get a `listen(audio)` that
plays inline with the model's title + color palettes above it, and a
`check_key(model)` that short-circuits external-LLM cells when their
API key isn't set instead of dumping a stack trace.

Usage::

    from latentscore.notebook_helpers import listen, check_key

    audio = ls.render("warm sunset over water", duration=10.0)
    listen(audio)

    MODEL = "external:gemini/gemini-3-flash-preview"
    if check_key(MODEL):
        audio = ls.render("late night neon", model=MODEL)
        listen(audio)

IPython is imported lazily so the rest of the library imports cleanly
in environments without IPython installed (CLI, demo backend, etc.).
"""

from __future__ import annotations

import html
import os
from typing import Any

from .dx import Audio

__all__ = ["listen", "check_key", "live_capture"]


# Title style chosen for dark/light theme compatibility: white fill
# with a thin black outline via 4 NSEW text-shadows. Renders crisp
# in both JupyterLab themes, Colab, and the GitHub static notebook
# viewer.
_TITLE_CSS = (
    "font-family:Georgia,'Times New Roman',serif;"
    "font-style:italic;"
    "font-size:1.7em;"
    "font-weight:500;"
    "color:#ffffff;"
    "text-shadow:"
    " 0   -1px 0 #000,"
    " 0    1px 0 #000,"
    "-1px  0   0 #000,"
    " 1px  0   0 #000;"
    "letter-spacing:0.01em;"
    "margin:14px 0 6px;"
    "line-height:1.3;"
)


def _ipython_objects() -> tuple[Any, Any, Any, Any]:
    """Return (HTML, Markdown, display, Audio). Imported lazily.

    Raises a friendly ImportError if IPython isn't available - these
    helpers are only meaningful inside Jupyter / Colab kernels.
    """
    try:
        from IPython.display import HTML, Markdown, display
        from IPython.display import Audio as IPAudio
    except ImportError as exc:
        raise ImportError(
            "latentscore.notebook_helpers requires IPython. "
            "Install it with `pip install ipython`, or use these helpers "
            "only from a Jupyter / Colab notebook kernel."
        ) from exc
    return HTML, Markdown, display, IPAudio


def _render_metadata(audio: Audio) -> None:
    """Pretty-print title + color palettes (and thinking, if any).

    Skips empty fields so the fast / fast_heavy models don't show a
    bare "thinking:" with nothing after it - those are nearest-neighbor
    lookups, so there's no chain-of-thought to surface.

    LLM-controlled fields (title, thinking) are HTML-escaped before
    interpolation so they can't inject script or attributes when the
    JupyterLab signature trusts the cell output.
    """
    HTML, _Markdown, display, _IPAudio = _ipython_objects()
    meta = audio.metadata
    if meta is None:
        return
    if meta.title:
        safe_title = html.escape(meta.title)
        display(HTML(f'<div style="{_TITLE_CSS}">{safe_title}</div>'))
    if meta.palettes:
        swatches = []
        for palette in meta.palettes:
            chips = "".join(
                # c.hex is regex-constrained to ^#[0-9A-Fa-f]{6}$ and
                # c.weight to a closed enum, so neither can break out
                # of the attribute context.
                f'<div title="{c.hex} ({c.weight})" '
                f'style="width:36px;height:36px;background:{c.hex};'
                f'border:1px solid rgba(0,0,0,0.08);border-radius:4px;"></div>'
                for c in palette.colors
            )
            swatches.append(f'<div style="display:flex;gap:3px;">{chips}</div>')
        display(
            HTML(
                '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 12px;">'
                + "".join(swatches)
                + "</div>"
            )
        )
    if meta.thinking:
        safe_thinking = html.escape(meta.thinking)
        display(
            HTML(
                f'<div style="font-style:italic;color:#444;white-space:pre-wrap;'
                f'margin:6px 0 12px;line-height:1.5;">{safe_thinking}</div>'
            )
        )


def _render_config(audio: Audio) -> None:
    """Pretty-print the MusicConfig as collapsible, syntax-styled JSON.

    Only renders when `audio.metadata.config` is present (i.e. the model
    returned one). Field values are LLM/internal-controlled but all
    serialise through Pydantic, so we still HTML-escape defensively.
    """
    HTML, _Markdown, display, _IPAudio = _ipython_objects()
    meta = audio.metadata
    if meta is None:
        return
    config_json = meta.config.model_dump_json(indent=2)
    safe_config = html.escape(config_json)
    display(
        HTML(
            '<details open style="margin:6px 0 14px;">'
            '<summary style="cursor:pointer;color:#555;font-weight:500;'
            'font-family:-apple-system,Segoe UI,sans-serif;font-size:0.9em;">'
            "MusicConfig (JSON)</summary>"
            '<pre style="background:#f6f8fa;border:1px solid rgba(0,0,0,0.08);'
            "border-radius:6px;padding:12px;font-size:0.82em;line-height:1.45;"
            'overflow:auto;margin:6px 0 0;color:#24292f;">'
            f"{safe_config}</pre>"
            "</details>"
        )
    )


def listen(audio: Audio, *, with_config: bool = False) -> Any:
    """Play audio inline, with the model's title + color palettes above it.

    If `with_config=True`, also renders the full `MusicConfig` as a
    collapsible pretty-printed JSON block (useful for inspecting LLM
    output in the BYOL section). Off by default — for the fast/lookup
    path the config is verbose noise.
    """
    _HTML, _Markdown, _display, IPAudio = _ipython_objects()
    _render_metadata(audio)
    if with_config:
        _render_config(audio)
    return IPAudio(audio.samples, rate=audio.sample_rate)


def live_capture(stream: Any, seconds: float, *, show_progress: bool = True) -> Audio:
    """Capture a live-steering stream into a buffered `Audio`, paced at wall-clock.

    Notebooks (Jupyter / Colab) can't real-time stream audio — they have no
    backpressure mechanism that would slow the consumer to audio-clock rate.
    Without that pacing, a generator-based source (`ls.live(my_gen())`) never
    gets wall-clock time to yield its next item between transitions, so the
    audio gets stuck on the first piece.

    This helper drains `stream.chunks(seconds=seconds)` at one chunk per
    wall-clock second, accumulates them, and returns a buffered `Audio`
    you can pass to `listen(...)` or `.save(...)`. The cell hangs for the
    full session duration; a `tqdm` bar shows progress if installed.

    Usage::

        async def my_set():
            yield "warm jazz cafe at midnight"
            await asyncio.sleep(10)
            yield "thunderstorm on a tin roof"

        audio = live_capture(ls.live(my_set()), seconds=20)
        listen(audio)
    """
    import time

    import numpy as np

    chunks: list[Any] = []
    t0 = time.monotonic()
    bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=int(seconds), unit="s", desc="Live session")
        except ImportError:
            bar = None
    try:
        for i, chunk in enumerate(stream.chunks(seconds=seconds)):
            chunks.append(chunk)
            if bar is not None:
                bar.update(1)
            deadline = t0 + (i + 1)  # default chunk_seconds = 1.0
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    finally:
        if bar is not None:
            bar.close()
    if not chunks:
        return Audio(samples=np.array([], dtype=np.float32), sample_rate=stream.sample_rate)
    return Audio(samples=np.concatenate(chunks), sample_rate=stream.sample_rate)


def check_key(model: str) -> bool:
    """Return True if the API key for `model` is set; otherwise display a hint.

    Lets external-LLM cells fail soft instead of dumping a deep LiteLLM
    stack trace. Non-external models (fast, fast_heavy, expressive) always
    return True - they need no key, so we short-circuit before touching
    IPython (lets this function be useful even outside notebook contexts).
    """
    if not model.startswith("external:"):
        return True
    provider = model.split(":", 1)[1].split("/", 1)[0].lower()
    env_var = f"{provider.upper()}_API_KEY"
    if os.environ.get(env_var):
        return True
    _HTML, Markdown, display, _IPAudio = _ipython_objects()
    display(
        Markdown(
            f"> 🔑 **Skipping this cell** - `{env_var}` is not set in "
            f"this environment. Set it in the cell above to run `{model}`."
        )
    )
    return False
