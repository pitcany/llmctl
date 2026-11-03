"""`bench --synthetic` replaces `bench --dry-run` (FIXPLAN 2026-09-18 §1.4).

`start --dry-run` executes nothing; `bench --dry-run` ran and persisted a
fake benchmark. The bench flag is now `--synthetic`; `--dry-run` still works
but warns.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from typer.testing import CliRunner

from llmctl import cli


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []

    class _Svc:
        def __init__(self, _db: Any) -> None: ...

        def run(self, payload: Any) -> Any:
            seen.append(payload)
            raise SystemExit(0)  # stop before result rendering; payload is the contract

    @contextmanager
    def _fake_session():  # type: ignore[no-untyped-def]
        yield None

    monkeypatch.setattr(cli, "BenchmarkService", _Svc)
    monkeypatch.setattr(cli, "_session", _fake_session)
    monkeypatch.setattr(cli, "_resolve_bench_target", lambda t, m, s: ("m1", None))
    return seen


def test_synthetic_flag_requests_a_mock_run(captured: list[Any]) -> None:
    res = CliRunner().invoke(cli.app, ["bench", "m1", "--synthetic"])
    assert captured and captured[0].dry_run is True, res.output
    assert "deprecated" not in res.output


def test_default_is_live(captured: list[Any]) -> None:
    CliRunner().invoke(cli.app, ["bench", "m1"])
    assert captured and captured[0].dry_run is False


def test_dry_run_still_works_but_warns(captured: list[Any]) -> None:
    res = CliRunner().invoke(cli.app, ["bench", "m1", "--dry-run"])
    assert captured and captured[0].dry_run is True
    assert "deprecated" in res.output
    assert "--synthetic" in res.output


def test_bad_sweep_is_a_usage_error_not_a_traceback(captured: list[Any]) -> None:
    res = CliRunner().invoke(cli.app, ["bench", "m1", "--sweep", "1,a"])
    assert res.exit_code == 2, res.output
    assert "--sweep" in res.output
    assert captured == []


def test_default_request_is_strict_live(captured: list[Any]) -> None:
    """FIXPLAN §1A: a plain `llmctl bench` run must not silently mock."""
    res = CliRunner().invoke(cli.app, ["bench", "m1"])
    assert res.exit_code in (0, 1), res.output  # SystemExit(0) from the stub
    assert captured
    assert captured[0].dry_run is False
    assert captured[0].require_live is True


def test_repetitions_flag_maps_to_payload(captured: list[Any]) -> None:
    res = CliRunner().invoke(cli.app, ["bench", "m1", "--repetitions", "3"])
    assert res.exit_code in (0, 1), res.output
    assert captured and captured[0].repetitions == 3


def test_repetitions_must_be_positive(captured: list[Any]) -> None:
    res = CliRunner().invoke(cli.app, ["bench", "m1", "--repetitions", "0"])
    assert res.exit_code == 2, res.output
    assert captured == []
