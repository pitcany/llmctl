"""Persistent operation records: idempotency, phases, recovery (§2B/§2C)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from llmctl.api.app import create_app
from llmctl.db import OperationRecord, get_engine
from llmctl.services.operations import (
    OperationPhase,
    OperationStore,
    cli_operation_guard,
    sanitize_line,
)
from llmctl.services.swap_jobs import OperationConflict, SwapJobs


@pytest.fixture
def store(tmp_path) -> OperationStore:
    return OperationStore(f"sqlite:///{tmp_path / 'ops.db'}")


def _state():

    from llmctl.services.lanes import SwapState, SwapTarget

    catalog = [
        SwapTarget(key="sess-b", kind="session", model="b", engine="llama.cpp", upstream="")
    ]
    return SwapState(
        catalog=catalog, serving=set(), active_units=set(), vllm_unit="v", vllm_active=False
    )


# -- record lifecycle --------------------------------------------------------


def test_begin_persists_intent_before_side_effects(store: OperationStore) -> None:
    record, created = store.begin(
        kind="swap",
        target="k",
        interface="api",
        purpose="swap to m",
        request={"name": "k"},
        previous_state={"serving": [{"key": "old"}]},
    )
    assert created is True
    assert record.phase == OperationPhase.PREPARING.value
    assert record.owner["pid"] == os.getpid()
    assert record.owner["interface"] == "api"
    assert record.previous_state == {"serving": [{"key": "old"}]}
    assert record.request == {"name": "k"}
    assert record.finished_at is None


def test_idempotent_replay_returns_existing(store: OperationStore) -> None:
    first, created = store.begin(
        kind="swap", target="k", interface="api", purpose="p",
        request={"name": "k"}, idempotency_key="key-1",
    )
    assert created
    second, created2 = store.begin(
        kind="swap", target="k", interface="api", purpose="p",
        request={"name": "k"}, idempotency_key="key-1",
    )
    assert created2 is False
    assert second.id == first.id
    # Still exactly one record: no second switch was started.
    assert len(store.unfinished()) == 1


def test_idempotency_key_reuse_for_different_request_is_refused(store: OperationStore) -> None:
    from llmctl.services.operations import IdempotencyConflict

    store.begin(
        kind="swap", target="k", interface="api", purpose="p",
        request={"name": "k"}, idempotency_key="key-1",
    )
    with pytest.raises(IdempotencyConflict):
        store.begin(
            kind="swap", target="other", interface="api", purpose="p",
            request={"name": "other"}, idempotency_key="key-1",
        )


def test_phase_updates_and_terminal_finish(store: OperationStore) -> None:
    record, _ = store.begin(
        kind="swap", target="k", interface="cli", purpose="p", request={}
    )
    for phase in (OperationPhase.STOPPING, OperationPhase.LOADING):
        store.set_phase(record.id, phase)
        assert store.get(record.id).phase == phase.value
    store.finish(record.id, OperationPhase.COMPLETED)
    final = store.get(record.id)
    assert final.phase == OperationPhase.COMPLETED.value
    assert final.finished_at is not None
    assert final.id not in [r.id for r in store.unfinished()]


def test_output_is_bounded_and_sanitized(store: OperationStore) -> None:
    record, _ = store.begin(kind="stop", target="x", interface="api", purpose="p", request={})
    store.append_output(record.id, "api_key=supersecret")
    store.append_output(record.id, "x" * 1000)
    out = store.get(record.id).output
    assert "supersecret" not in out[0]
    assert "<redacted>" in out[0]
    assert len(out[1]) <= 400


def test_sanitize_line_redacts_common_secret_shapes() -> None:
    assert "sk-123" not in sanitize_line("token=sk-123 hello")
    assert "hunter2" not in sanitize_line("password: hunter2")


# -- executor-side finalization ---------------------------------------------


def test_cli_guard_failed_exit_records_failed(tmp_path) -> None:
    import typer

    db_url = f"sqlite:///{tmp_path / 'g.db'}"
    with pytest.raises(typer.Exit) as excinfo:
        with cli_operation_guard(db_url=db_url, kind="swap", target="k", model="m"):
            raise typer.Exit(1)
    assert excinfo.value.exit_code == 1
    store = OperationStore(db_url)
    assert store.unfinished() == []
    from sqlmodel import Session

    from llmctl.db import OperationRecord, get_engine

    with Session(get_engine(db_url)) as db:
        row = db.exec(select(OperationRecord)).one()
    assert row.phase == OperationPhase.FAILED.value


def test_cli_guard_restore_failure_upgrades_to_needs_attention(tmp_path) -> None:
    import typer

    from llmctl.services.operations import (
        bind_current,
        clear_current,
        note_restore_failure,
    )

    db_url = f"sqlite:///{tmp_path / 'n.db'}"
    with pytest.raises(typer.Exit):
        with cli_operation_guard(db_url=db_url, kind="swap", target="k", model="m"):
            # Simulate the rollback path recording a failure to restore.
            store = OperationStore(db_url)
            record = store.unfinished()[0]
            bind_current(store, record.id)
            note_restore_failure(["lane-x"])
            raise typer.Exit(1)
    clear_current()
    store = OperationStore(db_url)

    from sqlmodel import Session

    from llmctl.db import OperationRecord, get_engine

    with Session(get_engine(db_url)) as db:
        row = db.exec(select(OperationRecord)).one()
    assert row.phase == OperationPhase.NEEDS_ATTENTION.value
    assert row.recovery["not_restored"] == ["lane-x"]


def test_cli_guard_keyboard_interrupt_is_interrupted(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'i.db'}"
    with pytest.raises(KeyboardInterrupt):
        with cli_operation_guard(db_url=db_url, kind="swap", target="k", model="m"):
            raise KeyboardInterrupt()
    from sqlmodel import Session

    from llmctl.db import OperationRecord, get_engine

    with Session(get_engine(db_url)) as db:
        row = db.exec(select(OperationRecord)).one()
    assert row.phase == OperationPhase.INTERRUPTED.value
    assert "guidance" in row.recovery


def test_cli_guard_adopts_parent_record(tmp_path) -> None:
    """The API parent hands over a record; the CLI adopts it instead of
    creating a second one."""

    db_url = f"sqlite:///{tmp_path / 'a.db'}"
    parent = OperationStore(db_url)
    record, _ = parent.begin(
        kind="swap", target="k", interface="api", purpose="swap to m", request={}
    )
    os.environ["LLMCTL_OPERATION_ID"] = record.id
    os.environ["LLMCTL_OPERATION_DB_URL"] = db_url
    try:
        with cli_operation_guard(db_url=db_url, kind="swap", target="k", model="m"):
            pass
    finally:
        os.environ.pop("LLMCTL_OPERATION_ID")
        os.environ.pop("LLMCTL_OPERATION_DB_URL")

    from sqlmodel import Session

    from llmctl.db import OperationRecord, get_engine

    with Session(get_engine(db_url)) as db:
        rows = db.exec(select(OperationRecord)).all()
    assert len(rows) == 1
    assert rows[0].phase == OperationPhase.COMPLETED.value
    assert rows[0].owner["interface"] == "cli"


# -- recovery (§2C) ----------------------------------------------------------


def test_reconcile_dead_owner_interrupted_with_guidance(store: OperationStore) -> None:
    record, _ = store.begin(kind="swap", target="k", interface="api", purpose="p", request={})
    store.set_phase(record.id, OperationPhase.LOADING)
    # Owner "died": point at a pid that does not exist.
    dead = dict(record.owner)
    dead["pid"] = 99999999
    from sqlmodel import Session

    from llmctl.db import OperationRecord, get_engine

    with Session(get_engine(store.database_url)) as db:
        row = db.get(OperationRecord, record.id)
        row.owner = dead
        db.add(row)
        db.commit()

    results = store.reconcile()
    assert results and results[0].outcome == "interrupted"
    final = store.get(record.id)
    assert final.phase == OperationPhase.INTERRUPTED.value
    assert "llmctl status" in final.recovery["guidance"]
    assert store.unfinished() == []


def test_reconcile_pid_reuse_is_detected(store: OperationStore) -> None:
    """Same pid, different start time: the owner is gone (PID recycled)."""
    record, _ = store.begin(kind="swap", target="k", interface="api", purpose="p", request={})
    with Session(get_engine(store.database_url)) as db:
        row = db.get(OperationRecord, record.id)
        owner = dict(row.owner)
        owner["pid"] = os.getpid()  # alive...
        owner["pid_start_time"] = "1"  # ...but a different process than ours
        row.owner = owner
        db.add(row)
        db.commit()
    results = store.reconcile()
    assert results[0].outcome == "interrupted"


def test_reconcile_restoring_phase_becomes_needs_attention(store: OperationStore) -> None:
    record, _ = store.begin(kind="swap", target="k", interface="api", purpose="p", request={})
    with Session(get_engine(store.database_url)) as db:
        row = db.get(OperationRecord, record.id)
        owner = dict(row.owner)
        owner["pid"] = 99999999
        row.owner = owner
        row.phase = OperationPhase.RESTORING.value
        db.add(row)
        db.commit()
    results = store.reconcile()
    assert results[0].outcome == "needs_attention"
    assert store.get(record.id).phase == OperationPhase.NEEDS_ATTENTION.value


def test_surviving_child_blocks_new_swap(tmp_path) -> None:
    """§2C acceptance: a child that outlives its API parent prevents a
    conflicting new operation; a restart does not clear it."""
    db_url = f"sqlite:///{tmp_path / 'survive.db'}"
    parent_store = OperationStore(db_url)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        record, _ = parent_store.begin(
            kind="swap", target="k", interface="api", purpose="p", request={}
        )
        # The child "adopted" the operation: owner is the child process.
        with Session(get_engine(db_url)) as db:
            row = db.get(OperationRecord, record.id)
            row.owner = {
                "pid": child.pid,
                "hostname": row.owner["hostname"],
                "interface": "cli",
                "purpose": "run m",
                "pid_start_time": None,
                "started_cmdline": ["python", "-c"],
            }
            db.add(row)
            db.commit()

        # A fresh manager (API restart) reconciles: the child is verified
        # alive, so the operation is reattached, not cleared.
        jobs = SwapJobs(database_url=db_url)
        assert any(r.outcome == "reattached" for r in jobs.reconciliation)
        with pytest.raises(OperationConflict):
            jobs.start_with_operation("k2", "m2", ["argv"])
    finally:
        child.kill()
        child.wait(timeout=10)


# -- API surface (§2D groundwork) -------------------------------------------


def test_api_idempotent_run_starts_one_switch(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.db'}"))
    gate = threading.Event()

    def fake(argv, on_line):
        on_line("working")
        gate.wait(5)
        return 0

    with patch("llmctl.services.lanes.gather_state", return_value=_state()), patch(
        "llmctl.services.swap_jobs.run_argv", fake
    ):
        first = client.post(
            "/swap/run", json={"name": "sess-b", "idempotency_key": "op-1"}
        )
        assert first.status_code == 202
        operation_id = first.json()["operation_id"]
        assert first.json()["phase"] == "preparing"

        replay = client.post(
            "/swap/run", json={"name": "sess-b", "idempotency_key": "op-1"}
        )
        assert replay.status_code == 200
        assert replay.json()["status"] == "already-started"
        assert replay.json()["operation_id"] == operation_id

        # A different request under the same key is refused.
        conflict = client.post(
            "/swap/run", json={"name": "other", "idempotency_key": "op-1"}
        )
        # 'other' does not resolve; patch a resolvable target instead:
        assert conflict.status_code in (404, 409)

        gate.set()
        from llmctl.api.deps import get_swap_jobs

        jobs = client.app.dependency_overrides[get_swap_jobs]()
        jobs.wait(first.json()["job_id"], 5)
        detail = client.get(f"/swap/jobs/{first.json()['job_id']}").json()
    assert detail["status"] == "succeeded"
    assert detail["operation_id"] == operation_id
    assert detail["phase"] == "completed"


def test_api_operation_survives_app_restart(tmp_path: Path) -> None:
    """§2D: recovery history is preserved after an API restart."""
    url = f"sqlite:///{tmp_path / 'restart.db'}"
    client = TestClient(create_app(database_url=url))
    with patch("llmctl.services.lanes.gather_state", return_value=_state()), patch(
        "llmctl.services.swap_jobs.run_argv", lambda a, cb: 1
    ):
        resp = client.post("/swap/run", json={"name": "sess-b"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        from llmctl.api.deps import get_swap_jobs

        client.app.dependency_overrides[get_swap_jobs]().wait(job_id, 5)

    # Fresh app over the same database: the finished operation is history,
    # not a conflict, and a new swap may start.
    client2 = TestClient(create_app(database_url=url))
    with patch("llmctl.services.lanes.gather_state", return_value=_state()), patch(
        "llmctl.services.swap_jobs.run_argv", lambda a, cb: 0
    ):
        again = client2.post("/swap/run", json={"name": "sess-b"})
    assert again.status_code == 202
