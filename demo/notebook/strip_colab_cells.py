#!/usr/bin/env python3
"""Strip Colab-only cells from the quickstart notebook.

The bundled notebook (`notebooks/quickstart.ipynb`) is authored for
Colab and contains a setup cell that does:

    !pip install -q "latentscore[external] @ git+..."
    os.kill(os.getpid(), 9)  # restarts the Colab runtime

Both are wrong for the local Docker context, where:
  - latentscore is pre-installed in the image (no re-fetch needed)
  - killing the kernel just confuses the user

This script reads the source notebook, drops any code cell whose
content references `pip install` and `latentscore` together, and
writes the cleaned version to the destination.

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


def main(source: str, dest: str) -> None:
    with open(source) as fp:
        nb = json.load(fp)

    nb["cells"] = [c for c in nb["cells"] if not _is_colab_install(c)]

    with open(dest, "w") as fp:
        json.dump(nb, fp, indent=1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source.ipynb> <dest.ipynb>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
