#!/usr/bin/env python3
"""Strip Colab-only content from the quickstart notebook.

The bundled notebook (`notebooks/quickstart.ipynb`) is authored for
Colab and contains two Colab-specific bits that don't belong in the
local Docker image, where latentscore is already pre-installed and
the user is already running JupyterLab:

  1. A setup code cell that does:
       !pip install -q "latentscore[external] @ git+..."
       os.kill(os.getpid(), 9)  # restarts the Colab runtime
  2. An "Open in Colab" badge in the title markdown cell.

This script reads the source notebook, drops any code cell whose
content references `pip install` and `latentscore` together, strips
any line in markdown cells that points at colab.research.google.com,
and writes the cleaned version to the destination.

Usage:
    python strip_colab_cells.py <source.ipynb> <dest.ipynb>
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _source_text(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(line) for line in src)  # type: ignore[reportUnknownVariableType]
    return str(src)


def _is_colab_install(cell: dict[str, Any]) -> bool:
    if cell.get("cell_type") != "code":
        return False
    text = _source_text(cell)
    return "pip install" in text and "latentscore" in text


def _strip_colab_badge(cell: dict[str, Any]) -> dict[str, Any]:
    """Remove any line in a markdown cell that links to colab.research.google.com."""
    if cell.get("cell_type") != "markdown":
        return cell
    src = cell.get("source", "")
    if isinstance(src, list):
        lines: list[str] = [str(line) for line in src]  # type: ignore[reportUnknownVariableType]
    else:
        # Preserve trailing newlines if we split on them.
        lines = str(src).splitlines(keepends=True)
    kept = [line for line in lines if "colab.research.google.com" not in line]
    if kept == lines:
        return cell
    cleaned = dict(cell)
    cleaned["source"] = kept
    return cleaned


def main(source: str, dest: str) -> None:
    with open(source) as fp:
        nb = json.load(fp)

    nb["cells"] = [
        _strip_colab_badge(c) for c in nb["cells"] if not _is_colab_install(c)
    ]

    with open(dest, "w") as fp:
        json.dump(nb, fp, indent=1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source.ipynb> <dest.ipynb>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
