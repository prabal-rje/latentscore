"""Display helpers for the bundled JupyterLab demo.

Shipped into the Docker image so that an IPython startup script can
pre-import these into every kernel - that way users can run any cell
in any order without first running the notebook's "Setup" section.

The same helpers also live (duplicated) in cell 3 of the source
notebook so the Colab path keeps working without this module.
"""

from __future__ import annotations

import os

from IPython.display import HTML, Markdown, display
from IPython.display import Audio as IPAudio

import latentscore as ls


def _render_metadata(audio: ls.Audio) -> None:
    """Pretty-print title + color palettes (and thinking, if any).

    Skips empty fields so the fast / fast_heavy models don't show a
    bare "thinking:" with nothing after it - those are nearest-neighbor
    lookups, so there's no chain-of-thought to surface.
    """
    meta = audio.metadata
    if meta is None:
        return
    if meta.title:
        # Georgia + italic + letter-spacing reads as a "title card"
        # and visually steps away from the surrounding sans-serif
        # notebook prose. Georgia ships on virtually every OS.
        display(
            HTML(
                f"<div style=\"font-family:Georgia,'Times New Roman',serif;"
                f"font-style:italic;font-size:1.55em;font-weight:400;"
                f"color:#2c3e50;letter-spacing:0.01em;"
                f'margin:14px 0 4px;line-height:1.3;">'
                f"{meta.title}</div>"
            )
        )
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


def listen(audio: ls.Audio) -> IPAudio:
    """Play audio inline, with the model's title + color palettes above it."""
    _render_metadata(audio)
    return IPAudio(audio.samples, rate=audio.sample_rate)


def check_key(model: str) -> bool:
    """Return True if the API key for `model` is set; otherwise display a hint.

    Lets external-LLM cells fail soft instead of dumping a deep LiteLLM
    stack trace::

        MODEL = "external:gemini/gemini-3-flash-preview"
        if check_key(MODEL):
            audio = ls.render("late night neon", model=MODEL)
            listen(audio)

    Non-external models (fast, fast_heavy, expressive) always return True.
    """
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


__all__ = ["_render_metadata", "listen", "check_key"]
