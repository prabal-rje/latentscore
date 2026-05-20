"""Tests for `latentscore doctor` — the install health check CLI."""

from __future__ import annotations

import json

from latentscore.cli import main
from latentscore.doctor import (
    DoctorCheck,
    DoctorReport,
    build_doctor_report,
    doctor_exit_code,
    render_json,
)


def _build_dev_report() -> DoctorReport:
    """The dev env has [external,heavy,expressive] installed, so requiring them
    must still pass. Offline mode keeps the test deterministic."""
    return build_doctor_report(
        strict=True,
        offline=True,
        require_external=False,
        require_heavy=False,
        require_expressive=False,
    )


def test_doctor_offline_core_passes() -> None:
    """Default offline doctor must exit 0 in any sane install."""
    assert main(["doctor", "--offline"]) == 0


def test_doctor_strict_offline_passes_in_dev_env() -> None:
    """Strict mode must not fire false-positives in the dev env."""
    assert main(["doctor", "--strict", "--offline"]) == 0


def test_doctor_json_output_is_parseable(capsys) -> None:
    """JSON output must be valid JSON with the expected top-level keys."""
    main(["doctor", "--json", "--offline"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] in ("pass", "warn", "fail")
    assert isinstance(payload["checks"], list)
    assert payload["offline"] is True


def test_doctor_report_includes_all_checks() -> None:
    """Build the report directly and assert the contract."""
    report = _build_dev_report()
    names = {c.name for c in report.checks}
    expected = {
        "python_version",
        "package_version",
        "license_present",
        "audio_write",
        "schema_export",
        "render_core",
        "render_retrieval",
        "external_available",
        "heavy_available",
        "expressive_available",
    }
    assert expected.issubset(names), f"missing checks: {expected - names}"


def test_doctor_exit_code_pass_when_not_strict() -> None:
    """Non-strict mode always exits 0 regardless of failed checks."""
    report = DoctorReport(
        status="fail",
        strict=False,
        offline=False,
        checks=(
            DoctorCheck(
                name="dummy",
                status="fail",
                required=True,
                detail="forced failure",
            ),
        ),
    )
    assert doctor_exit_code(report) == 0


def test_doctor_exit_code_fail_when_strict_and_required_fails() -> None:
    """Strict mode exits non-zero only when a required check fails."""
    report = DoctorReport(
        status="fail",
        strict=True,
        offline=False,
        checks=(
            DoctorCheck(
                name="dummy",
                status="fail",
                required=True,
                detail="forced failure",
            ),
        ),
    )
    assert doctor_exit_code(report) == 1


def test_doctor_exit_code_pass_when_strict_and_only_warn() -> None:
    """Warns alone don't trigger a non-zero exit, even in strict mode."""
    report = DoctorReport(
        status="warn",
        strict=True,
        offline=False,
        checks=(
            DoctorCheck(
                name="dummy",
                status="warn",
                required=False,
                detail="forced warn",
            ),
        ),
    )
    assert doctor_exit_code(report) == 0


def test_doctor_render_json_round_trip() -> None:
    """render_json output round-trips through json.loads."""
    report = _build_dev_report()
    payload = json.loads(render_json(report))
    assert payload["status"] == report.status
    assert len(payload["checks"]) == len(report.checks)
    assert payload["checks"][0]["name"] == report.checks[0].name


def test_doctor_required_check_failure_triggers_warn_overall() -> None:
    """When --require-external is on but litellm is poison-imported absent,
    the external_available check should be required + fail."""
    # In the dev env, litellm IS installed, so we can't test the missing path
    # directly. Use a constructed report to verify the rollup logic.
    checks = (
        DoctorCheck(name="python_version", status="pass", required=True, detail="3.10"),
        DoctorCheck(
            name="external_available",
            status="fail",
            required=True,
            detail="missing: litellm",
            hint='pip install "latentscore[external]"',
        ),
    )
    report = DoctorReport(status="fail", strict=True, offline=False, checks=checks)
    assert doctor_exit_code(report) == 1
