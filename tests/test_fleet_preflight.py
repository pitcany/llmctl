"""Tests for :mod:`llmctl.integrations.fleet`.

Pins the stop list for each target role, the ordering, and the
typed report so the CLI can use ``not report.all_clean`` as an abort
signal.
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


def test_units_to_stop_for_tp_includes_fleet_target_first() -> None:
    """Order matters: agents.target before its child services."""
    cfg = FleetUnitsConfig()
    stops = units_to_stop(FleetRole.TP, cfg)
    assert stops == [
        cfg.fleet_target,
        cfg.coder,
        cfg.reasoner,
        cfg.ollama,
        cfg.tp,
    ]


def test_units_to_stop_for_slot_excludes_other_slot() -> None:
    """Coder doesn't need reasoner stopped — they coexist by design."""
    cfg = FleetUnitsConfig()
    stops_coder = units_to_stop(FleetRole.CODER, cfg)
    stops_reasoner = units_to_stop(FleetRole.REASONER, cfg)
    assert cfg.coder not in stops_coder  # not in its own stop list
    assert cfg.reasoner not in stops_coder  # sibling slot preserved
    assert cfg.coder not in stops_reasoner  # mirror
    assert cfg.tp in stops_coder
    assert cfg.ollama in stops_coder


def test_units_to_stop_rejects_unknown_role() -> None:
    """Defensive: invalid role raises so callers fail fast."""
    with pytest.raises(ValueError):
        units_to_stop("nonsense", FleetUnitsConfig())  # type: ignore[arg-type]


def test_preflight_stop_skips_inactive_units() -> None:
    """try_stop short-circuit: inactive units land in `skipped`, not `failed`."""
    active = {"vllm-coder"}  # only one active
    runner = _runner(active)
    report = preflight_stop(
        FleetRole.TP,
        FleetUnitsConfig(),
        runner,
        logger=lambda _: None,
    )
    assert report.stopped == ["vllm-coder"]
    assert set(report.skipped) == {"agents.target", "vllm-reasoner", "ollama", "vllm-tp"}
    assert report.failed == []
    assert report.all_clean is True


def test_preflight_stop_records_failures() -> None:
    """Failed stops surface in `failed` and flip all_clean to False."""
    active = {"vllm-coder", "vllm-reasoner"}
    runner = _runner(active, fail_on={"vllm-coder"})
    logged: list[str] = []
    report = preflight_stop(
        FleetRole.TP,
        FleetUnitsConfig(),
        runner,
        logger=logged.append,
    )
    assert "vllm-coder" in report.failed
    assert "vllm-reasoner" in report.stopped
    assert report.all_clean is False
    # Failure was surfaced to the logger
    assert any("FAILED to stop vllm-coder" in line for line in logged)


def test_preflight_stop_uses_custom_unit_names() -> None:
    """A re-targeted FleetUnitsConfig is respected end-to-end."""
    cfg = FleetUnitsConfig(
        tp="my-tp",
        coder="my-coder",
        reasoner="my-reasoner",
        ollama="my-ollama",
        fleet_target="my-fleet.target",
    )
    active = {"my-coder", "my-tp"}
    runner = _runner(active)
    report = preflight_stop(FleetRole.TP, cfg, runner, logger=lambda _: None)
    assert set(report.stopped) == {"my-coder", "my-tp"}
    assert "my-fleet.target" in report.skipped


def test_preflight_stop_for_coder_target() -> None:
    """Starting coder doesn't stop the reasoner slot."""
    active = {"vllm-tp", "vllm-coder", "vllm-reasoner"}
    runner = _runner(active)
    report = preflight_stop(
        FleetRole.CODER,
        FleetUnitsConfig(),
        runner,
        logger=lambda _: None,
    )
    assert "vllm-tp" in report.stopped
    assert "vllm-reasoner" not in report.stopped  # left running
    assert "vllm-reasoner" not in report.skipped  # not even checked


def test_preflight_logs_each_stopped_unit() -> None:
    """Logger gets one line per successful stop for operator audibility."""
    active = {"vllm-tp", "ollama"}
    runner = _runner(active)
    logged: list[str] = []
    preflight_stop(
        FleetRole.CODER,
        FleetUnitsConfig(),
        runner,
        logger=logged.append,
    )
    assert any("stopped vllm-tp" in line for line in logged)
    assert any("stopped ollama" in line for line in logged)
