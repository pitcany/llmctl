"""Session API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session as DBSession

from llmctl.api.deps import get_db_session, get_settings
from llmctl.config import Settings
from llmctl.schemas import LaunchPlan, Session, SessionStartRequest, StatusChange
from llmctl.services.op_lock import LockBusy
from llmctl.services.scheduler import SchedulerError
from llmctl.services.sessions import AdoptError, SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[Session])
def list_sessions(
    db: DBSession = Depends(get_db_session), settings: Settings = Depends(get_settings),
) -> list[Session]:
    """List sessions."""
    return SessionService(db, settings=settings).list_sessions()


@router.post("/plan", response_model=LaunchPlan)
def plan_session(
    payload: SessionStartRequest, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> LaunchPlan:
    """Return an inspectable launch plan without launching."""
    return SessionService(db, settings=settings).plan(payload)


@router.post("/cleanup")
def cleanup_sessions(
    remove_stale: bool = False, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Reconcile dead sessions, free ports, and optionally purge stale records."""
    return SessionService(db, settings=settings).cleanup(remove_stale=remove_stale)


@router.post("/start", response_model=Session, status_code=status.HTTP_201_CREATED)
def start_session(
    payload: SessionStartRequest, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> Session:
    """Plan or launch a session."""
    try:
        return SessionService(db, settings=settings).start(payload)
    except SchedulerError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LockBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/{session_id}/stop", response_model=Session)
def stop_session(
    session_id: str, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> Session:
    """Stop a session safely.

    Returns 409 Conflict for adopted sessions — their lifecycle belongs to
    systemd, not llmctl. Use ``systemctl stop <unit>`` or ``llmctl detach``.
    """
    try:
        result = SessionService(db, settings=settings).stop(session_id)
    except AdoptError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LockBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return result


@router.post("/{session_id}/restart", response_model=Session)
def restart_session(
    session_id: str, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> Session:
    """Plan a session restart.

    Returns 409 Conflict for adopted sessions; ``systemctl restart <unit>``
    and a subsequent ``reconcile`` is the right path.
    """
    try:
        result = SessionService(db, settings=settings).restart(session_id)
    except AdoptError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LockBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return result


@router.get("/{session_id}/history", response_model=list[StatusChange])
def session_history(
    session_id: str, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> list[StatusChange]:
    """The session's status trail, oldest first (``docs/STATE-MACHINES.md`` §1).

    Returns 409 Conflict when the id is a prefix that matches several sessions.
    """
    try:
        session = SessionService(db, settings=settings).get_session(session_id)
    except AdoptError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session.status_history


@router.get("/{session_id}/systemd-unit")
def session_systemd_unit(
    session_id: str, user: bool = True, db: DBSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Return a systemd unit that relaunches the session on boot.

    Returns 409 Conflict for adopted sessions — they have no llmctl-
    issued launch command; the caller should install the upstream
    ``systemd_unit`` directly.
    """
    from llmctl.services.systemd import render_session_unit

    try:
        session = SessionService(db, settings=settings).get_session(session_id)
    except AdoptError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    try:
        unit = render_session_unit(session, user=user)
    except AdoptError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {
        "name": unit.name,
        "content": unit.content,
        "warnings": unit.warnings,
        "install_commands": unit.install_commands(),
    }
