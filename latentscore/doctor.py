"""Reviewer-grade install health checks.

`latentscore doctor` runs a battery of checks against the current install
so reviewers / users can verify the package works without trial-and-error
in a notebook. Output renders as a Rich table by default, or machine-
readable JSON via `--json`. With `--strict`, any failing required check
exits non-zero.

Optional-extra checks (`--require-external` / `--require-heavy` /
`--require-expressive`) flip those extras from "warn if missing" to
"fail if missing" — useful for CI verifying a specific install profile.

`--offline` sets `HF_HUB_OFFLINE` + `TRANSFORMERS_OFFLINE` so the checks
don't go to the network. Useful in air-gapped CI.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .audio import SAMPLE_RATE
from .config import MusicConfig

__all__ = [
    "CheckStatus",
    "DoctorCheck",
    "DoctorReport",
    "build_doctor_report",
    "doctor_exit_code",
    "render_json",
    "render_text",
]

CheckStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    required: bool
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    status: CheckStatus
    strict: bool
    offline: bool
    checks: tuple[DoctorCheck, ...]


# ---------------------------------------------------------------------------
# Offline-mode env-var context manager
# ---------------------------------------------------------------------------

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@contextmanager
def _offline_env(enabled: bool) -> Generator[None, None, None]:
    if not enabled:
        yield
        return
    saved = {k: os.environ.get(k) for k in _OFFLINE_ENV_VARS}
    for k in _OFFLINE_ENV_VARS:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_python_version() -> DoctorCheck:
    v = sys.version_info
    in_sweet_spot = (3, 11) <= (v.major, v.minor) <= (3, 12)
    return DoctorCheck(
        name="python_version",
        status="pass" if in_sweet_spot else "warn",
        required=False,
        detail=f"{v.major}.{v.minor}.{v.micro}",
        hint=None
        if in_sweet_spot
        else (
            "latentscore is tested on Python 3.11-3.12; other versions "
            "may have dependency or behaviour quirks."
        ),
    )


def _check_package_version() -> DoctorCheck:
    try:
        from importlib.metadata import version as pkg_version

        version_str = pkg_version("latentscore")
    except Exception as exc:
        return DoctorCheck(
            name="package_version",
            status="fail",
            required=True,
            detail=f"could not resolve: {type(exc).__name__}: {exc}",
            hint="Reinstall: pip install --force-reinstall latentscore",
        )
    return DoctorCheck(
        name="package_version",
        status="pass",
        required=True,
        detail=version_str,
    )


def _check_license_present() -> DoctorCheck:
    try:
        from importlib.metadata import metadata as pkg_metadata

        meta = pkg_metadata("latentscore")
    except Exception as exc:
        return DoctorCheck(
            name="license_present",
            status="fail",
            required=True,
            detail=f"could not read package metadata: {exc}",
            hint=None,
        )
    license_field: str | None = None
    for field_name in ("License-Expression", "License"):
        try:
            value = meta[field_name]
        except KeyError:
            continue
        if value:
            license_field = str(value)
            break
    if license_field:
        return DoctorCheck(
            name="license_present",
            status="pass",
            required=True,
            detail=license_field,
        )
    return DoctorCheck(
        name="license_present",
        status="fail",
        required=True,
        detail="no License or License-Expression field in package metadata",
        hint='Ensure pyproject.toml declares `license = "Apache-2.0"`.',
    )


def _check_audio_write() -> DoctorCheck:
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doctor.wav"
            samples = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            sf.write(str(path), samples, SAMPLE_RATE)  # type: ignore[reportUnknownMemberType]
            size = path.stat().st_size if path.exists() else 0
        return DoctorCheck(
            name="audio_write",
            status="pass" if size > 44 else "fail",
            required=True,
            detail=f"wrote {size} bytes",
            hint=None if size > 44 else "soundfile/libsndfile may be unavailable.",
        )
    except Exception as exc:
        return DoctorCheck(
            name="audio_write",
            status="fail",
            required=True,
            detail=f"{type(exc).__name__}: {exc}",
            hint="Install libsndfile or reinstall soundfile.",
        )


def _check_schema_export() -> DoctorCheck:
    try:
        schema = MusicConfig.model_json_schema()
        n_props = len(schema.get("properties", {}))
        return DoctorCheck(
            name="schema_export",
            status="pass" if n_props > 0 else "fail",
            required=True,
            detail=f"MusicConfig fields={n_props}",
            hint=None if n_props > 0 else "Pydantic schema export returned no properties.",
        )
    except Exception as exc:
        return DoctorCheck(
            name="schema_export",
            status="fail",
            required=True,
            detail=f"{type(exc).__name__}: {exc}",
            hint=None,
        )


def _check_render_core() -> DoctorCheck:
    """Render a 2-second MusicConfig — no model needed, pure synthesis."""
    try:
        import latentscore as ls

        audio = ls.render(ls.MusicConfig(), duration=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = audio.save(Path(tmp) / "doctor.wav")
            size = Path(path).stat().st_size if Path(path).exists() else 0
        return DoctorCheck(
            name="render_core",
            status="pass" if size > 44 else "fail",
            required=True,
            detail=f"samples={len(audio.samples)} sr={audio.sample_rate} bytes={size}",
            hint=None if size > 44 else "Render produced no valid WAV output.",
        )
    except Exception as exc:
        return DoctorCheck(
            name="render_core",
            status="fail",
            required=True,
            detail=f"{type(exc).__name__}: {exc}",
            hint=None,
        )


def _check_render_retrieval() -> DoctorCheck:
    """Render a text prompt — verifies sentence-transformers retrieval is wired."""
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("latentscore.models")
    logger.addHandler(handler)
    try:
        import latentscore as ls

        audio = ls.render("warm sunset over water", duration=2.0)
        logs = log_buf.getvalue()
        used_fallback = "Fast model fallback" in logs
        if used_fallback:
            return DoctorCheck(
                name="render_retrieval",
                status="warn",
                required=False,
                detail="text-prompt render produced audio via heuristic fallback",
                hint=(
                    "Embedding retrieval failed at runtime (likely model not cached "
                    "and --offline set). Run `latentscore download fast` first."
                ),
            )
        return DoctorCheck(
            name="render_retrieval",
            status="pass",
            required=True,
            detail=f"samples={len(audio.samples)} sr={audio.sample_rate}",
        )
    except Exception as exc:
        return DoctorCheck(
            name="render_retrieval",
            status="fail",
            required=True,
            detail=f"{type(exc).__name__}: {exc}",
            hint=None,
        )
    finally:
        logger.removeHandler(handler)


def _check_extra(
    name: str,
    probe_modules: tuple[str, ...],
    extra_label: str,
    required: bool,
) -> DoctorCheck:
    """Probe one or more module specs to verify an optional extra is installed."""
    missing = [m for m in probe_modules if importlib.util.find_spec(m) is None]
    if missing:
        return DoctorCheck(
            name=name,
            status="fail" if required else "warn",
            required=required,
            detail=f"missing: {', '.join(missing)}",
            hint=f'pip install "latentscore[{extra_label}]"',
        )
    return DoctorCheck(
        name=name,
        status="pass",
        required=required,
        detail=f"installed: {', '.join(probe_modules)}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _overall_status(checks: tuple[DoctorCheck, ...]) -> CheckStatus:
    if any(c.status == "fail" and c.required for c in checks):
        return "fail"
    if any(c.status in {"fail", "warn"} for c in checks):
        return "warn"
    return "pass"


def build_doctor_report(
    *,
    strict: bool,
    offline: bool,
    require_external: bool,
    require_heavy: bool,
    require_expressive: bool,
) -> DoctorReport:
    """Run all checks under the requested settings and return a report."""
    with _offline_env(offline):
        checks_list: list[DoctorCheck] = [
            _check_python_version(),
            _check_package_version(),
            _check_license_present(),
            _check_audio_write(),
            _check_schema_export(),
            _check_render_core(),
            _check_render_retrieval(),
            _check_extra(
                "external_available",
                ("litellm",),
                "external",
                required=require_external,
            ),
            _check_extra(
                "heavy_available",
                ("laion_clap",),
                "heavy",
                required=require_heavy,
            ),
            _check_extra(
                "expressive_available",
                ("outlines",),
                "expressive",
                required=require_expressive,
            ),
        ]
    checks = tuple(checks_list)
    return DoctorReport(
        status=_overall_status(checks),
        strict=strict,
        offline=offline,
        checks=checks,
    )


def doctor_exit_code(report: DoctorReport) -> int:
    """0 unless strict + a required check failed."""
    if not report.strict:
        return 0
    if any(c.required and c.status == "fail" for c in report.checks):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_json(report: DoctorReport) -> str:
    """Machine-readable JSON. No Rich markup."""
    return json.dumps(
        {
            "status": report.status,
            "strict": report.strict,
            "offline": report.offline,
            "checks": [asdict(c) for c in report.checks],
        },
        indent=2,
    )


def render_text(report: DoctorReport, console_print: Callable[[str], None]) -> None:
    """Human-readable table. Takes a console_print callable so callers can plug
    in Rich, plain print, or whatever."""
    status_glyph = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    name_width = max(len(c.name) for c in report.checks)
    console_print("LatentScore doctor\n")
    for c in report.checks:
        glyph = status_glyph[c.status]
        console_print(f"{glyph} {c.name.ljust(name_width)}  {c.detail}")
        if c.hint:
            console_print(f"     {' ' * name_width}  Hint: {c.hint}")
    console_print(f"\nOverall: {status_glyph[report.status]}")
    if report.strict:
        console_print(f"(strict mode — exit code: {doctor_exit_code(report)})")
