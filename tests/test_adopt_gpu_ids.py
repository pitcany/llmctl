"""Adopted sessions track GPU pinning + stay current with model/GPU swaps.

Covers the fix for the TUI showing an adopted vLLM unit as "cpu": the adopt
path now derives ``gpu_ids`` from the systemd unit, and reconcile refreshes both
``gpu_ids`` and ``served_name`` as the unit is re-pointed at new presets.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.services.sessions import SessionService


def _make_service(
    tmp_path: Path,
    probe: Callable[[str, float], list[str] | None],
    gpu_ids_for_unit: Callable[[str | None], list[int]],
    *,
    db_name: str = "adopt_gpu.sqlite3",
) -> tuple[Session, SessionService]:
    url = f"sqlite:///{tmp_path / db_name}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, probe=probe, gpu_ids_for_unit=gpu_ids_for_unit)
    return db, service


def test_adopt_populates_gpu_ids_from_unit(tmp_path: Path) -> None:
    db, service = _make_service(
        tmp_path, lambda u, _t: ["qwen3.6-27b"], lambda unit: [0, 1]
    )
    try:
        session = service.adopt(
            RuntimeName.VLLM,
            "http://127.0.0.1:8003",
            systemd_unit="vllm-tp.service",
        )
        assert session.gpu_ids == [0, 1]
    finally:
        db.close()


def test_adopt_without_unit_leaves_gpu_ids_empty(tmp_path: Path) -> None:
    # No systemd_unit => nothing to introspect; the deriver must not be called.
    calls: list[str | None] = []

    def deriver(unit: str | None) -> list[int]:
        calls.append(unit)
        return [0, 1]

    db, service = _make_service(tmp_path, lambda u, _t: ["m"], deriver)
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert session.gpu_ids == []
        assert calls == []
    finally:
        db.close()


def test_reconcile_backfills_gpu_ids_for_existing_row(tmp_path: Path) -> None:
    # Simulate a pre-fix row: adopted with empty gpu_ids (deriver returned []),
    # then the unit becomes introspectable on a later reconcile.
    derived: list[list[int]] = [[], [0, 1]]

    def deriver(_unit: str | None) -> list[int]:
        return derived.pop(0) if derived else [0, 1]

    db, service = _make_service(tmp_path, lambda u, _t: ["qwen3.6-27b"], deriver)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp.service"
        )
        assert session.gpu_ids == []  # first derive returned []
        changed = service.reconcile()
        assert changed == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.gpu_ids == [0, 1]
    finally:
        db.close()


def test_reconcile_refreshes_served_name_after_model_swap(tmp_path: Path) -> None:
    # Unit adopted while serving llama; later re-pointed at qwen with no re-adopt.
    served: list[list[str]] = [["llama-3.3-70b"], ["qwen3.6-27b"]]

    def probe(_url: str, _t: float) -> list[str] | None:
        return served.pop(0) if served else ["qwen3.6-27b"]

    db, service = _make_service(tmp_path, probe, lambda unit: [0, 1])
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp.service"
        )
        assert session.served_name == "llama-3.3-70b"
        changed = service.reconcile()
        assert changed == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.served_name == "qwen3.6-27b"
    finally:
        db.close()


def test_reconcile_does_not_clobber_gpu_ids_on_transient_failure(
    tmp_path: Path,
) -> None:
    # First derive yields [0,1] (adopt); next derive yields [] (systemctl hiccup).
    # Reconcile must keep the known [0,1] rather than flap the row to "cpu".
    derived: list[list[int]] = [[0, 1], []]

    def deriver(_unit: str | None) -> list[int]:
        return derived.pop(0) if derived else []

    db, service = _make_service(tmp_path, lambda u, _t: ["qwen3.6-27b"], deriver)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp.service"
        )
        assert session.gpu_ids == [0, 1]
        changed = service.reconcile()  # served_name unchanged, derive returns []
        assert changed == 0
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.gpu_ids == [0, 1]
    finally:
        db.close()


def test_reconcile_no_change_when_metadata_already_current(tmp_path: Path) -> None:
    db, service = _make_service(
        tmp_path, lambda u, _t: ["qwen3.6-27b"], lambda unit: [0, 1]
    )
    try:
        service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp.service"
        )
        assert service.reconcile() == 0  # nothing drifted
    finally:
        db.close()
