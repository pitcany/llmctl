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
