"""Session lifecycle service.

Orchestrates session start/stop/restart by combining the scheduler (plan
building), the runtime router (adapter execution), and persistence. Honors the
``dry_run``/``safe_mode`` policy: a dry-run start records a ``PLANNED`` session
and never launches a process; a real start performs genuine process control.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import or_
from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import Settings, load_settings
from llmctl.db import (
    EventLevel,
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    utcnow,
)
from llmctl.schemas import LaunchPlan, Session, SessionStartRequest
from llmctl.services.backends import probe_openai_v1_models
from llmctl.services.events import log_event
from llmctl.services.router import RuntimeRouter
from llmctl.services.scheduler import SchedulerService
from llmctl.services.unit_gpus import unit_gpu_ids

_ACTIVE_STATES = {SessionStatus.RUNNING, SessionStatus.STARTING}

#: Statuses that block a new adopt at the same endpoint URL. Anything except
#: ``FAILED`` and (for OWNED rows) ``STOPPED`` reserves the endpoint enough
#: that adopting on top of it would produce ambiguous routing. ``PLANNED``
#: matters here because dry-run starts record a port/endpoint without yet
#: launching — Bugbot would otherwise let that overlap a real adopt.
_ADOPT_BLOCKING_STATES = {
    SessionStatus.PLANNED,
    SessionStatus.STARTING,
    SessionStatus.RUNNING,
    SessionStatus.STOPPING,
    SessionStatus.UNKNOWN,
}

#: Default per-call timeout (seconds) for HTTP probes against adopted endpoints.
_ADOPT_PROBE_TIMEOUT_S = 1.5


class AdoptError(ValueError):
    """Raised when an adopt request cannot be honored.

    Used for the three "user input or state is wrong" cases the CLI/API
    should surface back to the caller: probe failure (endpoint not
    serving), duplicate adoption (already-tracked endpoint), and
    unsupported runtimes.
    """


#: Probe function signature: ``(base_url, timeout) -> list[str] | None``.
ProbeFn = Callable[[str, float], list[str] | None]

#: Resolves the GPU indices a systemd unit is pinned to: ``(unit_name) -> [int]``.
GpuIdsFn = Callable[[str | None], list[int]]


def record_to_session(record: SessionRecord) -> Session:
    """Convert a database session record to schema."""
    plan = LaunchPlan.model_validate(record.launch_plan) if record.launch_plan else None
    return Session(
        id=record.id,
        model_id=record.model_id,
        profile_id=record.profile_id,
        runtime=record.runtime,
        status=record.status,
        kind=record.kind or SessionKind.OWNED,
        pid=record.pid,
        port=record.port,
        endpoint_url=record.endpoint_url,
        log_path=record.log_path,
        gpu_ids=record.gpu_ids,
        launch_plan=plan,
        error=record.error,
        systemd_unit=record.systemd_unit,
        served_name=record.served_name,
        adopted_at=record.adopted_at,
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
        *,
        probe: ProbeFn | None = None,
        gpu_ids_for_unit: GpuIdsFn | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or load_settings()
        self.router = router or RuntimeRouter(self.settings)
        self.scheduler = SchedulerService(db, self.settings)
        self._probe: ProbeFn = probe or (
            lambda url, timeout: probe_openai_v1_models(url, timeout)
        )
        self._gpu_ids_for_unit: GpuIdsFn = gpu_ids_for_unit or unit_gpu_ids

    def list_sessions(self) -> list[Session]:
        """Return all known sessions as currently persisted.

        This is a *pure DB read*: it does not probe adopted endpoints, does
        not check OWNED PIDs, and does not synchronize tracking with
        reality. That keeps it sub-millisecond regardless of how many
        adopted endpoints are down — important because both the FastAPI
        ``GET /sessions`` route and the TUI dashboard call it on every
        refresh.

        Synchronization is the job of :meth:`reconcile`, which the gateway
        runs on a periodic timer (``router.reconcile_interval_s``) and which
        ``cleanup()`` and the ``llmctl reconcile`` command call explicitly.
        Callers that need *fresh* state should call :meth:`reconcile` first.
        """
        records = self.db.exec(select(SessionRecord)).all()
        return [record_to_session(record) for record in records]

    def reconcile(self) -> int:
        """Reconcile session liveness with reality.

        - **OWNED** sessions: if their supervised PID is no longer running,
          mark ``STOPPED``.
        - **ADOPTED** sessions: their lifecycle lives in systemd, not in our
          process tree. Probe ``{endpoint_url}/v1/models`` fresh each pass
          (no cache — every reconcile sees ground truth, never stale).
          A probe failure on a previously ``RUNNING``/``STARTING`` adopted
          row flips it to ``STOPPED``; a probe success on a previously
          ``STOPPED`` adopted row flips it back to ``RUNNING`` so a
          systemd-restarted unit becomes routable again without manual
          re-adopt.

        Returns the number of sessions transitioned.
        """
        # Load only rows reconcile can act on:
        #   - active OWNED/ADOPTED (PID check / probe-failure detection), and
        #   - STOPPED ADOPTED (auto-revival when systemd brings the unit back).
        # OWNED STOPPED rows would be skipped by the inner loop anyway; not
        # loading them keeps the query proportional to active session count
        # rather than full session history.
        records = self.db.exec(
            select(SessionRecord).where(
                or_(
                    SessionRecord.status.in_(_ACTIVE_STATES),  # type: ignore[attr-defined]
                    (SessionRecord.status == SessionStatus.STOPPED)
                    & (SessionRecord.kind == SessionKind.ADOPTED),
                )
            )
        ).all()
        changed = 0
        for record in records:
            if record.kind == SessionKind.ADOPTED:
                changed += self._reconcile_adopted(record)
            elif record.status in _ACTIVE_STATES:
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

    def _reconcile_adopted(self, record: SessionRecord) -> int:
        """Update one adopted record based on a fresh probe; return 1 if changed."""
        if not record.endpoint_url:
            return 0
        served = self._probe(record.endpoint_url, _ADOPT_PROBE_TIMEOUT_S)
        alive = served is not None and len(served) > 0
        now = utcnow()
        if record.status in _ACTIVE_STATES and not alive:
            record.status = SessionStatus.STOPPED
            record.stopped_at = now
            record.error = "Adopted endpoint failed to respond on /v1/models."
            record.updated_at = now
            self.db.add(record)
            log_event(
                self.db,
                EventLevel.WARNING,
                "session",
                f"Adopted session {record.id} marked stopped; "
                f"{record.endpoint_url} no longer responds.",
                session_id=record.id,
                model_id=record.model_id,
                data={"endpoint_url": record.endpoint_url, "kind": "adopted"},
            )
            return 1
        changed = False
        if record.status == SessionStatus.STOPPED and alive:
            record.status = SessionStatus.RUNNING
            record.stopped_at = None
            record.error = None
            changed = True
            log_event(
                self.db,
                EventLevel.INFO,
                "session",
                f"Adopted session {record.id} revived; "
                f"{record.endpoint_url} is responding again.",
                session_id=record.id,
                model_id=record.model_id,
                data={"endpoint_url": record.endpoint_url, "kind": "adopted"},
            )

        # While the endpoint is alive, keep the row's served_name + GPU pinning
        # current. A managed unit can be re-pointed at a different preset (via
        # `llmctl vllm <preset>`) with no re-adopt, so reconcile is the only
        # place a model/GPU swap on the same endpoint becomes visible.
        if alive and self._refresh_adopted_metadata(record, served):
            changed = True

        if changed:
            record.updated_at = now
            self.db.add(record)
            return 1
        return 0

    def _refresh_adopted_metadata(
        self, record: SessionRecord, served: list[str] | None
    ) -> bool:
        """Sync an alive adopted row's ``served_name`` + ``gpu_ids`` with reality.

        Returns ``True`` when either field changed. GPU derivation is
        best-effort: an empty result (the unit probe failed) leaves the stored
        ids untouched rather than flapping the row to a CPU label on a transient
        ``systemctl`` hiccup. Does not commit — :meth:`reconcile` commits once
        per pass.
        """
        updated = False
        new_name = served[0] if served else None
        if new_name and record.served_name != new_name:
            record.served_name = new_name
            updated = True
        if record.systemd_unit:
            gpu_ids = self._gpu_ids_for_unit(record.systemd_unit)
            if gpu_ids and gpu_ids != (record.gpu_ids or []):
                record.gpu_ids = gpu_ids
                updated = True
        return updated

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
        """Stop a session, terminating its process when applicable.

        Refuses for ``ADOPTED`` sessions: their lifecycle belongs to
        systemd, not llmctl. The caller should use ``systemctl stop
        <unit>`` or ``llmctl detach`` instead.
        """
        record = self.db.get(SessionRecord, session_id)
        if not record:
            return None
        if record.kind == SessionKind.ADOPTED:
            raise AdoptError(
                f"Session {record.id} is adopted ({record.systemd_unit or record.endpoint_url}); "
                "llmctl does not manage its lifecycle. Use `systemctl stop <unit>` to stop the "
                "upstream, or `llmctl detach <session_id>` to remove it from tracking."
            )
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
        """Stop and relaunch a session, reusing its stored launch plan.

        Refuses for ``ADOPTED`` sessions — llmctl never spawned the
        upstream, so it cannot restart it. The caller can use ``systemctl
        restart <unit>`` and ``reconcile()`` will revive the row.
        """
        record = self.db.get(SessionRecord, session_id)
        if not record:
            return None
        if record.kind == SessionKind.ADOPTED:
            raise AdoptError(
                f"Session {record.id} is adopted ({record.systemd_unit or record.endpoint_url}); "
                "llmctl does not manage its lifecycle. Use `systemctl restart <unit>` and the "
                "session will be revived on the next reconcile."
            )
        self._terminate_record(record)
        plan = (
            LaunchPlan.model_validate(record.launch_plan)
            if record.launch_plan
            else None
        )
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

    def adopt(
        self,
        runtime: RuntimeName,
        endpoint_url: str,
        *,
        served_name: str | None = None,
        systemd_unit: str | None = None,
        timeout_s: float = _ADOPT_PROBE_TIMEOUT_S,
    ) -> Session:
        """Track an externally-managed endpoint as a ``kind=ADOPTED`` session.

        Probes ``{endpoint_url}/v1/models``; on success, inserts a
        ``RUNNING`` row so the gateway can route to it. ``served_name``
        is taken from the first probed model when not supplied. Refuses
        if the endpoint is unreachable or already tracked by another
        non-terminal session.
        """
        if not endpoint_url:
            raise AdoptError("adopt() requires a non-empty endpoint_url.")
        normalized = self._normalize_endpoint_url(endpoint_url)

        existing = self.db.exec(
            select(SessionRecord).where(SessionRecord.endpoint_url == normalized)
        ).all()
        for prior in existing:
            prior_kind = prior.kind or SessionKind.OWNED
            # Refuse against any non-terminal status (PLANNED/STARTING/
            # RUNNING/STOPPING/UNKNOWN). PLANNED is the subtle one: a
            # dry-run start records the endpoint URL without launching,
            # and adopting on top would silently fork the routing.
            if prior.status in _ADOPT_BLOCKING_STATES:
                raise AdoptError(
                    f"Endpoint {normalized} is already tracked by session {prior.id} "
                    f"(status={prior.status.value}, kind={prior_kind.value})."
                )
            # A STOPPED adopted row at the same URL would be auto-revived by
            # the next reconcile, producing two RUNNING records pointing at
            # the same endpoint. Refuse and point the user at detach so the
            # gateway never has to disambiguate identical routes.
            if (
                prior_kind == SessionKind.ADOPTED
                and prior.status == SessionStatus.STOPPED
            ):
                raise AdoptError(
                    f"Endpoint {normalized} has a stopped adopted session {prior.id} that will "
                    "auto-revive on the next reconcile. Run "
                    f"`llmctl detach {prior.id}` first if you want a fresh record."
                )

        served_ids = self._probe(normalized, timeout_s)
        if not served_ids:
            raise AdoptError(
                f"Probe of {normalized}/v1/models failed or returned no models. "
                "The endpoint must be serving and answer OpenAI /v1/models before adoption."
            )

        resolved_served_name = served_name or served_ids[0]
        port = self._extract_port(normalized)
        # Adopted units run under systemd, so we can't read a launch plan for
        # GPU placement — but the unit's MainPID environ exposes the
        # CUDA_VISIBLE_DEVICES it was started with. Best-effort: [] when the
        # unit isn't running or the host can't be introspected.
        gpu_ids = self._gpu_ids_for_unit(systemd_unit) if systemd_unit else []

        plan = LaunchPlan(
            runtime=runtime,
            command=[],
            env={},
            gpu_ids=gpu_ids,
            port=port,
            endpoint_url=normalized,
            health_url=f"{normalized}/v1/models",
            dry_run=False,
            notes=[
                "adopted",
                f"served_name={resolved_served_name}",
                f"probed_models={','.join(served_ids)}",
                *([f"systemd_unit={systemd_unit}"] if systemd_unit else []),
            ],
        )
        now = utcnow()
        record = SessionRecord(
            runtime=runtime,
            status=SessionStatus.RUNNING,
            kind=SessionKind.ADOPTED,
            port=port,
            endpoint_url=normalized,
            health_url=plan.health_url,
            gpu_ids=gpu_ids,
            systemd_unit=systemd_unit,
            served_name=resolved_served_name,
            adopted_at=now,
            started_at=now,
            launch_plan=plan.model_dump(mode="json"),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        log_event(
            self.db,
            EventLevel.INFO,
            "session",
            f"Adopted {runtime.value} session {record.id} at {normalized} "
            f"(served_name={resolved_served_name}).",
            session_id=record.id,
            data={
                "endpoint_url": normalized,
                "served_name": resolved_served_name,
                "systemd_unit": systemd_unit,
                "kind": "adopted",
            },
        )
        return record_to_session(record)

    def detach(self, session_id: str) -> Session | None:
        """Remove an ``ADOPTED`` session from tracking.

        Deletes the SessionRecord outright — the upstream systemd unit
        is untouched. Returns the schema view of the deleted session for
        UX feedback, or ``None`` when the id is unknown. Refuses for
        ``OWNED`` sessions: those have a process llmctl spawned, so the
        right verb is ``stop`` followed by ``cleanup``.
        """
        record = self.db.get(SessionRecord, session_id)
        if not record:
            return None
        if record.kind != SessionKind.ADOPTED:
            raise AdoptError(
                f"Session {record.id} is not adopted (kind="
                f"{(record.kind or SessionKind.OWNED).value}); use `llmctl stop` then "
                "`llmctl cleanup --remove-stale` to retire it."
            )
        snapshot = record_to_session(record)
        self.db.delete(record)
        self.db.commit()
        log_event(
            self.db,
            EventLevel.INFO,
            "session",
            f"Detached adopted session {session_id} "
            f"({snapshot.systemd_unit or snapshot.endpoint_url}).",
            data={
                "endpoint_url": snapshot.endpoint_url,
                "served_name": snapshot.served_name,
                "systemd_unit": snapshot.systemd_unit,
                "kind": "adopted",
            },
        )
        return snapshot

    @staticmethod
    def _normalize_endpoint_url(endpoint_url: str) -> str:
        """Canonicalize an endpoint URL for duplicate detection + storage.

        - Strips trailing slash.
        - Lowercases scheme and host.
        - Folds ``localhost`` to ``127.0.0.1`` so the two loopback aliases
          that point at the same listener compare equal. Does **not**
          resolve DNS; only handles this one common alias case so the
          duplicate-adopt check can't be bypassed by spelling.

        IPv6 loopback (``[::1]``) is intentionally left alone — it's a
        different protocol family and folding across IPv4/IPv6 would
        misclassify in setups that legitimately bind both.
        """
        from urllib.parse import urlparse, urlunparse

        trimmed = endpoint_url.rstrip("/")
        parsed = urlparse(trimmed)
        host = (parsed.hostname or "").lower()
        if host == "localhost":
            host = "127.0.0.1"
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urlunparse((parsed.scheme.lower(), netloc, parsed.path, "", "", ""))

    @staticmethod
    def _extract_port(endpoint_url: str) -> int | None:
        """Best-effort port extraction from an http(s) URL."""
        try:
            return urlparse(endpoint_url).port
        except ValueError:
            return None

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
