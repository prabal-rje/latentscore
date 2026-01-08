from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

from rich.console import Console

from .audio import SAMPLE_RATE
from .dx import render
from .errors import ModelNotAvailableError
from .logging_utils import configure_logging, log_exception
from .spinner import Spinner, render_error

_EXPRESSIVE_REPO = "mlx-community/gemma-3-1b-it-qat-8bit"
_EXPRESSIVE_DIR = "gemma-3-1b-it-qat-8bit"
_LOGGER = logging.getLogger("latentscore.cli")
_CONSOLE = Console()


def _default_model_base() -> Path:
    configured = os.environ.get("LATENTSCORE_MODEL_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "latentscore" / "models"


def _download_expressive(model_base: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError as exc:
        _LOGGER.warning("huggingface_hub not installed: %s", exc, exc_info=True)
        raise ModelNotAvailableError("huggingface_hub is not installed") from exc

    target = model_base / _EXPRESSIVE_DIR
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(_EXPRESSIVE_REPO, local_dir=str(target))
    return target


def _doctor_report(lines: Iterable[str]) -> None:
    for line in lines:
        _CONSOLE.print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latentscore")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Render a short demo clip.")
    demo.add_argument("--duration", type=float, default=2.5)
    demo.add_argument("--output", type=str, default="demo.wav")

    download = sub.add_parser("download", help="Download model assets.")
    download.add_argument("model", choices=["expressive"], type=str)

    sub.add_parser("doctor", help="Check model availability and cache paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "demo":
            with Spinner("Rendering demo audio"):
                audio = render("warm sunrise", duration=args.duration)
            path = audio.save(args.output)
            _CONSOLE.print(f"Wrote demo to {path} (sr={SAMPLE_RATE})")
            return 0

        if args.command == "download":
            model_base = _default_model_base()
            model_base.mkdir(parents=True, exist_ok=True)
            if args.model == "expressive":
                target = model_base / _EXPRESSIVE_DIR
                if not target.exists():
                    with Spinner("Downloading expressive model"):
                        _download_expressive(model_base)
                _CONSOLE.print(f"Downloaded expressive model to {target}")
                return 0

        if args.command == "doctor":
            model_base = _default_model_base()
            expressive_dir = model_base / _EXPRESSIVE_DIR
            embeddings_dir = Path("models") / "all-MiniLM-L6-v2"
            report = [
                f"Model cache base: {model_base}",
                f"Expressive model present: {expressive_dir.exists()}",
                f"Embeddings model present: {embeddings_dir.exists()}",
                "Hints:",
                "- Run `latentscore download expressive` to prefetch the LLM weights.",
                "- Set LATENTSCORE_MODEL_DIR to point at a preseeded models directory.",
                "- In production, run `latentscore doctor` and prefetch missing models to avoid runtime downloads.",
            ]
            _doctor_report(report)
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        debug = bool(os.environ.get("LATENTSCORE_DEBUG"))
        _LOGGER.warning("latentscore CLI failed: %s", exc, exc_info=debug)
        log_exception("latentscore CLI", exc)
        render_error("latentscore CLI", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
