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

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Audio  # noqa: F401


__all__ = ["listen", "check_key"]


# Title style chosen for dark/light theme compatibility: white fill
# with a 1px black outline via four offset text-shadows. Renders crisp
# in both JupyterLab themes, Colab, and the GitHub static notebook
# viewer.
_TITLE_CSS = (
    "font-family:Georgia,'Times New Roman',serif;"
    "font-style:italic;"
    "font-size:1.7em;"
    "font-weight:500;"
    "color:#ffffff;"
    "text-shadow:"
    "-1px -1px 0 #000,"
    " 1px -1px 0 #000,"
    "-1px  1px 0 #000,"
    " 1px  1px 0 #000,"
    " 0   -1px 0 #000,"
    " 0    1px 0 #000,"
    "-1px  0   0 #000,"
    " 1px  0   0 #000;"
    "letter-spacing:0.01em;"
    "margin:14px 0 6px;"
    "line-height:1.3;"
)


def _ipython_objects() -> tuple[Any, Any, Any, Any]:
    """Return (HTML, Markdown, display, Audio). Imported lazily."""
    from IPython.display import HTML, Markdown, display
    from IPython.display import Audio as IPAudio

    return HTML, Markdown, display, IPAudio


def _render_metadata(audio: Audio) -> None:
    """Pretty-print title + color palettes (and thinking, if any).

    Skips empty fields so the fast / fast_heavy models don't show a
    bare "thinking:" with nothing after it - those are nearest-neighbor
    lookups, so there's no chain-of-thought to surface.
    """
    HTML, Markdown, display, _IPAudio = _ipython_objects()
    meta = audio.metadata
    if meta is None:
        return
    if meta.title:
        display(HTML(f'<div style="{_TITLE_CSS}">{meta.title}</div>'))
    if meta.palettes:
        swatches = []
        for palette in meta.palettes:
            chips = "".join(
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
        display(Markdown(f"_{meta.thinking}_"))


def listen(audio: Audio) -> Any:
    """Play audio inline, with the model's title + color palettes above it."""
    _, _, _, IPAudio = _ipython_objects()
    _render_metadata(audio)
    return IPAudio(audio.samples, rate=audio.sample_rate)


def check_key(model: str) -> bool:
    """Return True if the API key for `model` is set; otherwise display a hint.

    Lets external-LLM cells fail soft instead of dumping a deep LiteLLM
    stack trace. Non-external models (fast, fast_heavy, expressive) always
    return True.
    """
    _HTML, Markdown, display, _IPAudio = _ipython_objects()
    if not model.startswith("external:"):
        return True
    provider = model.split(":", 1)[1].split("/", 1)[0].lower()
    env_var = f"{provider.upper()}_API_KEY"
    if os.environ.get(env_var):
        return True
    display(
        Markdown(
            f"> 🔑 **Skipping this cell** - `{env_var}` is not set in "
            f"this environment. Set it in the cell above to run `{model}`."
        )
    )
    return False
