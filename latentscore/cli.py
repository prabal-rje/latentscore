from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rich.console import Console

from .audio import SAMPLE_RATE
from .dx import render
from .errors import ModelNotAvailableError
from .logging_utils import configure_logging, log_exception
from .spinner import Spinner, render_error

_EXPRESSIVE_REPO = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_REPO",
    "guprab/latentscore-gemma3-270m-v5-merged",
)
_EXPRESSIVE_DIR = os.environ.get(
    "LATENTSCORE_EXPRESSIVE_DIR",
    "latentscore-gemma3-270m-v5-merged",
)
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
        raise ModelNotAvailableError(
            "huggingface_hub is a required dependency of latentscore but is not "
            "importable. Reinstall with: pip install --force-reinstall latentscore"
        ) from exc

    target = model_base / _EXPRESSIVE_DIR
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(_EXPRESSIVE_REPO, local_dir=str(target))
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latentscore")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Render a short demo clip.")
    demo.add_argument("--duration", type=float, default=2.5)
    demo.add_argument("--output", type=str, default="demo.wav")

    download = sub.add_parser("download", help="Download model assets.")
    download.add_argument("model", choices=["expressive", "fast", "fast_heavy"], type=str)

    doctor = sub.add_parser(
        "doctor",
        help="Run install health checks (Python version, license, render, retrieval, ...).",
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor.add_argument(
        "--strict", action="store_true", help="Exit nonzero when required checks fail."
    )
    doctor.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE for network-free checks.",
    )
    doctor.add_argument(
        "--require-external",
        action="store_true",
        help="Treat [external] (LiteLLM) availability as required.",
    )
    doctor.add_argument(
        "--require-heavy",
        action="store_true",
        help="Treat [heavy] (laion-clap) availability as required.",
    )
    doctor.add_argument(
        "--require-expressive",
        action="store_true",
        help="Treat [expressive] availability as required.",
    )
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
            if args.model in ("fast", "fast_heavy"):
                from .prefetch import prefetch as _prefetch

                _prefetch(args.model)  # type: ignore[arg-type]
                return 0

        if args.command == "doctor":
            from .doctor import build_doctor_report, doctor_exit_code, render_json, render_text

            report = build_doctor_report(
                strict=args.strict,
                offline=args.offline,
                require_external=args.require_external,
                require_heavy=args.require_heavy,
                require_expressive=args.require_expressive,
            )
            if args.json:
                print(render_json(report))
            else:
                # `markup=False` so Rich doesn't parse `[external]`/`[heavy]`/
                # `[expressive]` in hint strings as styling tags and silently
                # strip them.
                from functools import partial

                render_text(report, partial(_CONSOLE.print, markup=False))
            return doctor_exit_code(report)

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
