"""OpenAI-compatible gateway resolution service.

The gateway maps an inbound OpenAI ``model`` string to a concrete
upstream ``(base_url, served_name)`` running on this host. It performs
*no* HTTP itself — the FastAPI layer in :mod:`llmctl.api.gateway` does
the proxy and streaming. Keeping resolution in a pure service module
lets the CLI (``llmctl aliases`` / ``llmctl router-status``) reuse the
same code paths the gateway uses on every request, which means the two
views cannot disagree.

Resolution order on every ``resolve(model)`` call:

1. **Explicit model** — match the requested string against an active
   session's served name (the vLLM ``--served-model-name`` parameter
   when set, else the model record name), the session id, or the model
   record id. First active session wins; planned/stopped sessions are
   ignored.
2. **Alias** — match against ``router.aliases`` (settings) overlaid by
   ``<config_dir>/router_aliases.json`` (set-alias overlay). The alias
   target is itself resolved via step 1, so an alias may point at a
   session id, a profile name, or a served model name.
3. **Fallback** — when ``router.fallback_policy == "fallback"`` and
   ``router.fallback_target`` is set, resolve the target via step 1.
   Otherwise the request 404s.

Aliases that don't have a target (target=None) are tracked in the
``aliases`` view but never resolve. That's deliberate: it lets the
``llmctl aliases`` table show every role the host *intends* to support
even before any are bound, so the user sees what they can `set-alias`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import Settings, load_settings
from llmctl.db import ProfileRecord, SessionRecord, SessionStatus

_ACTIVE = (SessionStatus.RUNNING, SessionStatus.STARTING)
_OVERLAY_FILENAME = "router_aliases.json"


@dataclass(frozen=True)
class RouteTarget:
    """The concrete upstream a request should be proxied to.

    ``base_url`` is the active session's ``endpoint_url`` (e.g.
    ``http://127.0.0.1:8003``). ``served_name`` is the name the upstream
    expects in the JSON body's ``model`` field — for vLLM this is the
    ``--served-model-name`` arg when set, else the HuggingFace repo id.
    The gateway rewrites the inbound ``model`` to ``served_name`` before
    proxying so callers can use the alias ("local-coding") while the
    upstream sees its own native name.
    """

    base_url: str
    served_name: str
    session_id: str
    via: str  # "explicit" | "alias:<key>" | "fallback"


@dataclass(frozen=True)
class AliasView:
    """One row in the ``llmctl aliases`` table."""

    name: str
    target: str | None
    resolved_session_id: str | None
    resolved_served_name: str | None
    healthy: bool


class GatewayService:
    """Resolves OpenAI ``model`` strings to running sessions on this host."""

    def __init__(self, db: DBSession, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or load_settings()

    # -- public API ---------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-style ``/v1/models`` entries.

        One entry per active session (id = served name) plus one virtual
        entry per bound alias (id = ``local-<alias>``). The virtual
        entries let callers configure their OpenAI client with a stable
        role name even when the underlying model gets swapped — the
        canonical use case for the alias feature.
        """
        sessions = self._active_sessions()
        created = int(time.time())
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record, served in sessions:
            if served in seen:
                continue
            seen.add(served)
            items.append(
                {
                    "id": served,
                    "object": "model",
                    "created": created,
                    "owned_by": "llmctl",
                    "session_id": record.id,
                    "runtime": record.runtime.value,
                }
            )
        aliases = self._merged_aliases()
        for alias, target in aliases.items():
            if target is None:
                continue
            virtual_id = self._alias_public_id(alias)
            if virtual_id in seen:
                continue
            resolved = self._resolve_target_string(target)
            if resolved is None:
                continue
            seen.add(virtual_id)
            items.append(
                {
                    "id": virtual_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "llmctl",
                    "alias": alias,
                    "session_id": resolved.session_id,
                }
            )
        return items

    def resolve(self, model: str) -> RouteTarget | None:
        """Resolve an inbound ``model`` to an upstream, or ``None``."""
        if not model:
            return None
        explicit = self._resolve_target_string(model)
        if explicit is not None:
            return RouteTarget(
                base_url=explicit.base_url,
                served_name=explicit.served_name,
                session_id=explicit.session_id,
                via="explicit",
            )
        alias_key = self._alias_key_for(model)
        if alias_key is not None:
            target = self._merged_aliases().get(alias_key)
            if target is not None:
                resolved = self._resolve_target_string(target)
                if resolved is not None:
                    return RouteTarget(
                        base_url=resolved.base_url,
                        served_name=resolved.served_name,
                        session_id=resolved.session_id,
                        via=f"alias:{alias_key}",
                    )
        if (
            self.settings.router.fallback_policy == "fallback"
            and self.settings.router.fallback_target
        ):
            resolved = self._resolve_target_string(self.settings.router.fallback_target)
            if resolved is not None:
                return RouteTarget(
                    base_url=resolved.base_url,
                    served_name=resolved.served_name,
                    session_id=resolved.session_id,
                    via="fallback",
                )
        return None

    def alias_view(self) -> list[AliasView]:
        """Return one :class:`AliasView` per known alias (bound or not)."""
        merged = self._merged_aliases()
        result: list[AliasView] = []
        for alias in sorted(merged):
            target = merged[alias]
            resolved = self._resolve_target_string(target) if target else None
            result.append(
                AliasView(
                    name=alias,
                    target=target,
                    resolved_session_id=resolved.session_id if resolved else None,
                    resolved_served_name=resolved.served_name if resolved else None,
                    healthy=resolved is not None,
                )
            )
        return result

    def set_alias(self, alias: str, target: str | None) -> None:
        """Persist ``alias -> target`` in the overlay file.

        Passing ``target=None`` *unsets* the alias (removes the overlay
        entry, falling back to whatever ``settings.yaml`` declared, if
        anything). The settings YAML is never rewritten — overlay-only
        keeps user edits to YAML safe from CLI clobbering.
        """
        path = self._overlay_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        overlay = self._read_overlay()
        if target is None:
            overlay.pop(alias, None)
        else:
            overlay[alias] = target
        path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")

    # -- helpers ------------------------------------------------------------

    def _active_sessions(self) -> list[tuple[SessionRecord, str]]:
        """Return ``[(record, served_name), ...]`` for active sessions."""
        records = self.db.exec(
            select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE))  # type: ignore[attr-defined]
        ).all()
        out: list[tuple[SessionRecord, str]] = []
        for record in records:
            if not record.endpoint_url:
                continue
            served = self._served_name(record)
            if not served:
                continue
            out.append((record, served))
        return out

    @staticmethod
    def _served_name(record: SessionRecord) -> str | None:
        """Return the model id this session actually serves over HTTP."""
        plan = record.launch_plan or {}
        # Prefer the explicit served_model_name when the launch plan set one.
        if isinstance(plan, dict):
            params = plan.get("command") or []
            if isinstance(params, list):
                for idx, token in enumerate(params):
                    if token == "--served-model-name" and idx + 1 < len(params):
                        return str(params[idx + 1])
            # vLLM serves the HF repo id when no served-model-name override.
            command = plan.get("command")
            if isinstance(command, list) and len(command) >= 3 and command[1] == "serve":
                return str(command[2])
        # Llama.cpp / python-script sessions: use the session id as the served
        # identity. They don't have a canonical "model name" the same way
        # vLLM does, but a stable id is sufficient to address them.
        return record.id

    def _resolve_target_string(self, target: str | None) -> RouteTarget | None:
        """Resolve a target string (session id, served name, profile) once."""
        if not target:
            return None
        for record, served in self._active_sessions():
            if target == served or target == record.id or target == record.model_id:
                return RouteTarget(
                    base_url=str(record.endpoint_url),
                    served_name=served,
                    session_id=str(record.id),
                    via="explicit",
                )
        # Profile name -> first active session bound to that profile.
        profile = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == target)
        ).first()
        if profile is not None:
            for record, served in self._active_sessions():
                if record.profile_id == profile.id:
                    return RouteTarget(
                        base_url=str(record.endpoint_url),
                        served_name=served,
                        session_id=str(record.id),
                        via="explicit",
                    )
        return None

    def _merged_aliases(self) -> dict[str, str | None]:
        """Settings.yaml aliases overlaid by the JSON overlay file."""
        merged: dict[str, str | None] = dict(self.settings.router.aliases)
        merged.update(self._read_overlay())
        return merged

    def _alias_key_for(self, model: str) -> str | None:
        """Return the alias key matched by ``model``.

        Accepts both the bare alias (``coding``) and the conventional
        ``local-<alias>`` prefix the spec mentions, so callers can use
        whichever is friendlier to their OpenAI client config.
        """
        aliases = self._merged_aliases()
        if model in aliases:
            return model
        if model.startswith("local-"):
            tail = model[len("local-") :]
            if tail in aliases:
                return tail
        return None

    @staticmethod
    def _alias_public_id(alias: str) -> str:
        """Return the ``local-<alias>`` id exposed in /v1/models."""
        return f"local-{alias}"

    def _overlay_path(self) -> Path:
        """Return the per-host overlay path for set-alias persistence."""
        return Path(self.settings.config_dir) / _OVERLAY_FILENAME

    def _read_overlay(self) -> dict[str, str | None]:
        """Load the JSON overlay, returning ``{}`` when absent or malformed."""
        path = self._overlay_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): (str(v) if v is not None else None) for k, v in data.items()}
