"""Session lifecycle service.

Orchestrates session start/stop/restart by combining the scheduler (plan
building), the runtime router (adapter execution), and persistence. Honors the
``dry_run``/``safe_mode`` policy: a dry-run start records a ``PLANNED`` session
and never launches a process; a real start performs genuine process control.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import Settings, load_settings
from llmctl.db import EventLevel, SessionRecord, SessionStatus, utcnow
from llmctl.schemas import LaunchPlan, Session, SessionStartRequest
from llmctl.services.events import log_event
from llmctl.services.router import RuntimeRouter
from llmctl.services.scheduler import SchedulerService

_ACTIVE_STATES = {SessionStatus.RUNNING, SessionStatus.STARTING}


def record_to_session(record: SessionRecord) -> Session:
    """Convert a database session record to schema."""
    plan = LaunchPlan.model_validate(record.launch_plan) if record.launch_plan else None
    return Session(
        id=record.id,
        model_id=record.model_id,
        profile_id=record.profile_id,
        runtime=record.runtime,
        status=record.status,
        pid=record.pid,
        port=record.port,
        endpoint_url=record.endpoint_url,
        log_path=record.log_path,
        gpu_ids=record.gpu_ids,
        launch_plan=plan,
        error=record.error,
        created_at=record.created_at,
        started_at=record.started_at,
        stopped_at=record.stopped_at,
    )


class SessionService:
    """Service interface for runtime session lifecycle."""

    def __init__(
        self,
        db: DBSession,
        settings: Settings | None = None,
        router: RuntimeRouter | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or load_settings()
        self.router = router or RuntimeRouter(self.settings)
        self.scheduler = SchedulerService(db, self.settings)

    def list_sessions(self) -> list[Session]:
        """List all known sessions after reconciling dead processes."""
        self.reconcile()
        records = self.db.exec(select(SessionRecord)).all()
        return [record_to_session(record) for record in records]

    def reconcile(self) -> int:
        """Mark active sessions whose supervised process has died as stopped.

        Returns the number of sessions transitioned.
        """
        records = self.db.exec(
            select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE_STATES))  # type: ignore[attr-defined]
        ).all()
        changed = 0
        for record in records:
            if record.pid and not self.router.supervisor.is_running(record.pid):
                record.status = SessionStatus.STOPPED
                record.stopped_at = utcnow()
                record.error = "Process exited unexpectedly."
                record.updated_at = utcnow()
                record.pid = None
                self.db.add(record)
                changed += 1
                log_event(
                    self.db,
                    EventLevel.WARNING,
                    "session",
                    f"Session {record.id} marked dead; process is no longer running.",
                    session_id=record.id,
                    model_id=record.model_id,
                )
        if changed:
            self.db.commit()
        return changed

    def get_session(self, session_id: str) -> Session | None:
        """Return a single session by id."""
        record = self.db.get(SessionRecord, session_id)
        return record_to_session(record) if record else None

    def plan(self, request: SessionStartRequest) -> LaunchPlan:
        """Return an inspectable launch plan without launching anything."""
        return self.scheduler.create_launch_plan(request)

    def cleanup(self, *, remove_stale: bool = False) -> dict[str, object]:
        """Reconcile dead sessions and optionally purge terminal ones.

        Returns a report describing how many sessions were marked dead, how many
        stale (stopped/failed) records were removed, the ports that were freed,
        and the number of still-active sessions remaining.
        """
        dead_marked = self.reconcile()
        terminal = {SessionStatus.STOPPED, SessionStatus.FAILED}

        stale_records = self.db.exec(
            select(SessionRecord).where(SessionRecord.status.in_(terminal))  # type: ignore[attr-defined]
        ).all()

        host = self.settings.scheduler.default_host
        freed_ports: list[int] = []
        for record in stale_records:
            if record.port and SchedulerService._is_port_free(host, record.port):
                freed_ports.append(record.port)

        stale_removed = 0
        if remove_stale:
            for record in stale_records:
                self.db.delete(record)
                stale_removed += 1
            if stale_removed:
                self.db.commit()

        active_remaining = len(
            self.db.exec(
                select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE_STATES))  # type: ignore[attr-defined]
            ).all()
        )
        return {
            "dead_marked": dead_marked,
            "stale_removed": stale_removed,
            "freed_ports": sorted(set(freed_ports)),
            "active_remaining": active_remaining,
        }

    def tail_log(self, session_id: str, lines: int = 50) -> str | None:
        """Return the last ``lines`` of a session's log file.

        Returns ``None`` when the session does not exist, and an empty string
        when no log file is present yet.
        """
        record = self.db.get(SessionRecord, session_id)
        if record is None:
            return None
        if not record.log_path:
            return ""
        path = Path(record.log_path)
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])

    def start(self, request: SessionStartRequest) -> Session:
        """Plan and (when not dry-run) launch a runtime session."""
        plan = self.scheduler.create_launch_plan(request)
        self.scheduler.validate(plan, force=request.force, dry_run=request.dry_run)
        record = SessionRecord(
            model_id=request.model_id,
            profile_id=request.profile_id,
            runtime=request.runtime,
            status=SessionStatus.PLANNED,
            port=plan.port,
            gpu_ids=plan.gpu_ids,
            env=plan.env,
            command=plan.command,
            endpoint_url=plan.endpoint_url,
            health_url=plan.health_url,
            launch_plan=plan.model_dump(mode="json"),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return self._launch_record(record, plan)

    def stop(self, session_id: str) -> Session | None:
        """Stop a session, terminating its process when applicable."""
        record = self.db.get(SessionRecord, session_id)
        if not record:
            return None
        self._terminate_record(record)
        record.status = SessionStatus.STOPPED
        record.stopped_at = utcnow()
        record.pid = None
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        log_event(
            self.db,
            EventLevel.INFO,
            "session",
            f"Session {record.id} stopped.",
            session_id=record.id,
            model_id=record.model_id,
        )
        return record_to_session(record)

    def restart(self, session_id: str) -> Session | None:
        """Stop and relaunch a session, reusing its stored launch plan."""
        record = self.db.get(SessionRecord, session_id)
        if not record:
            return None
        self._terminate_record(record)
        plan = LaunchPlan.model_validate(record.launch_plan) if record.launch_plan else None
        record.status = SessionStatus.PLANNED
        record.error = None
        record.pid = None
        record.stopped_at = None
        record.started_at = None
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        if plan is None:
            return record_to_session(record)
        return self._launch_record(record, plan)

    def _launch_record(self, record: SessionRecord, plan: LaunchPlan) -> Session:
        """Apply launch policy to a persisted record and return the schema."""
        plan.log_name = f"session_{record.id}"
        record.launch_plan = plan.model_dump(mode="json")
        self.db.add(record)
        self.db.commit()

        if plan.dry_run:
            log_event(
                self.db,
                EventLevel.INFO,
                "session",
                f"Planned session {record.id} ({record.runtime.value}); no process launched.",
                session_id=record.id,
                model_id=record.model_id,
                data={"dry_run": True},
            )
            return record_to_session(record)

        record.status = SessionStatus.STARTING
        self.db.add(record)
        self.db.commit()

        adapter = self.router.get_adapter(record.runtime)
        result = asyncio.run(adapter.start(plan))

        record.status = result.status
        record.pid = result.pid
        record.endpoint_url = result.endpoint_url or record.endpoint_url
        record.log_path = result.log_path
        record.error = result.error
        record.started_at = result.started_at
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        if result.status == SessionStatus.RUNNING:
            log_event(
                self.db,
                EventLevel.INFO,
                "session",
                f"Started session {record.id} ({record.runtime.value}) pid={record.pid}.",
                session_id=record.id,
                model_id=record.model_id,
                data={"pid": record.pid, "endpoint": record.endpoint_url},
            )
        else:
            log_event(
                self.db,
                EventLevel.ERROR,
                "session",
                f"Failed to start session {record.id}: {result.error}",
                session_id=record.id,
                model_id=record.model_id,
                data={"error": result.error},
            )
        return record_to_session(record)

    def _terminate_record(self, record: SessionRecord) -> None:
        """Terminate the runtime process backing ``record`` when it is live."""
        if not record.pid and record.status not in _ACTIVE_STATES:
            return
        adapter = self.router.get_adapter(record.runtime)
        status = asyncio.run(adapter.stop(record_to_session(record)))
        log_event(
            self.db,
            EventLevel.INFO,
            "session",
            status.message,
            session_id=record.id,
            model_id=record.model_id,
            data=status.details,
        )
