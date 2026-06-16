"""Tests for :mod:`llmctl.integrations.fleet`.

Pins the stop list for the TP target, the ordering, and the typed
report so the CLI can use ``not report.all_clean`` as an abort signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llmctl.config import FleetUnitsConfig
from llmctl.integrations.fleet import (
    FleetRole,
    preflight_stop,
    units_to_stop,
)
from llmctl.integrations.systemctl import SystemctlRunner


@dataclass
class _FakeCompleted:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _runner(active: set[str], *, fail_on: set[str] | None = None) -> SystemctlRunner:
    """Build a SystemctlRunner with scripted is-active + stop."""
    fail_on = fail_on or set()

    def fake(argv: list[str]) -> _FakeCompleted:
        verb = argv[-2]
        unit = argv[-1]
        if verb == "is-active":
            return _FakeCompleted(stdout="active\n" if unit in active else "inactive\n")
        if verb == "stop":
            if unit in fail_on:
                return _FakeCompleted(returncode=1, stderr=f"could not stop {unit}\n")
            active.discard(unit)
            return _FakeCompleted()
        return _FakeCompleted()

    runner = SystemctlRunner(runner=fake)
    runner.available = lambda: True  # type: ignore[method-assign]
    return runner


def test_units_to_stop_for_tp() -> None:
    """Order matters: ollama before the TP restart."""
    cfg = FleetUnitsConfig()
    stops = units_to_stop(FleetRole.TP, cfg)
    assert stops == [cfg.ollama, cfg.tp]


def test_units_to_stop_rejects_unknown_role() -> None:
    """Defensive: invalid role raises so callers fail fast."""
    with pytest.raises(ValueError):
        units_to_stop("nonsense", FleetUnitsConfig())  # type: ignore[arg-type]


def test_preflight_stop_skips_inactive_units() -> None:
    """try_stop short-circuit: inactive units land in `skipped`, not `failed`."""
    active = {"vllm-tp"}  # only one active
    runner = _runner(active)
    report = preflight_stop(
        FleetRole.TP,
        FleetUnitsConfig(),
        runner,
        logger=lambda _: None,
    )
    assert report.stopped == ["vllm-tp"]
    assert set(report.skipped) == {"ollama"}
    assert report.failed == []
    assert report.all_clean is True


def test_preflight_stop_records_failures() -> None:
    """Failed stops surface in `failed` and flip all_clean to False."""
    active = {"ollama", "vllm-tp"}
    runner = _runner(active, fail_on={"ollama"})
    logged: list[str] = []
    report = preflight_stop(
        FleetRole.TP,
        FleetUnitsConfig(),
        runner,
        logger=logged.append,
    )
    assert "ollama" in report.failed
    assert "vllm-tp" in report.stopped
    assert report.all_clean is False
    # Failure was surfaced to the logger
    assert any("FAILED to stop ollama" in line for line in logged)


def test_preflight_stop_uses_custom_unit_names() -> None:
    """A re-targeted FleetUnitsConfig is respected end-to-end."""
    cfg = FleetUnitsConfig(
        tp="my-tp",
        ollama="my-ollama",
    )
    active = {"my-ollama", "my-tp"}
    runner = _runner(active)
    report = preflight_stop(FleetRole.TP, cfg, runner, logger=lambda _: None)
    assert set(report.stopped) == {"my-ollama", "my-tp"}


def test_preflight_logs_each_stopped_unit() -> None:
    """Logger gets one line per successful stop for operator audibility."""
    active = {"vllm-tp", "ollama"}
    runner = _runner(active)
    logged: list[str] = []
    preflight_stop(
        FleetRole.TP,
        FleetUnitsConfig(),
        runner,
        logger=logged.append,
    )
    assert any("stopped vllm-tp" in line for line in logged)
    assert any("stopped ollama" in line for line in logged)
