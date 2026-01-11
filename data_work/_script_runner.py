"""Helpers for module-based entrypoints."""

from __future__ import annotations

import runpy
from pathlib import Path


def run_script(filename: str) -> None:
    script_path = Path(__file__).with_name(filename)
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")
