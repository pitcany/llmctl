"""vLLM runtime adapter.

vLLM in production runs as one or more **externally-managed systemd
units** (see :mod:`llmctl.adapters.vllm_systemd`). This adapter is the
read-only health + discovery surface used by the TUI / scheduler /
registry; it probes each configured managed-unit port for
``/v1/models`` and surfaces what's currently being served.

Falls back to the legacy behaviour (binary lookup + filesystem
discovery) when no managed units are configured or when their HTTP
endpoints don't respond. That keeps the adapter useful on hosts that
don't run vLLM under systemd at all (e.g. a developer laptop).
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, NamedTuple

from llmctl.adapters._common import ProcessRuntimeAdapter
from llmctl.config import (
    ManagedUnitConfig,
    ManagedUnitsConfig,
    RuntimeConfig,
    default_runtime_configs,
)
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import AdapterStatus, HealthState, Model
from llmctl.telemetry.process import ProcessSupervisor


class ServedModel(NamedTuple):
    """One entry of a vLLM ``/v1/models`` response.

    ``id`` is the served name clients call (vLLM's ``--served-model-name``,
    falling back to the ``--model`` value). ``root`` is what ``--model``
    was actually set to — normally the checkpoint directory on disk, but
    a Hugging Face repo id when the server was pointed at the hub. The
    two differ exactly when a served name is aliased over a checkpoint,
    which is the case worth surfacing: ``ornith-35b`` the served name vs.
    the ``-refusal-v6`` directory it really loads.
    """

    id: str
    root: str | None = None


class VLLMAdapter(ProcessRuntimeAdapter):
    """Adapter for the vLLM runtime, with HTTP-first health + discovery.

    Args:
        config: Runtime config (binary lookup fallback).
        supervisor: Process supervisor (legacy subprocess launch path).
        managed_units: The configured managed systemd units to probe.
            ``vllm_tp`` is probed via the OpenAI ``/v1/models`` API.
        http_get: Injected HTTP getter for tests. Default uses
            :func:`urllib.request.urlopen` on ``http://localhost:<port>``.
        probe_timeout_s: Per-port probe timeout. Short by design so an
            unreachable unit doesn't stall the TUI refresh.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        supervisor: ProcessSupervisor | None = None,
        *,
        managed_units: ManagedUnitsConfig | None = None,
        http_get: Callable[[str, float], Any] | None = None,
        probe_timeout_s: float = 1.5,
    ) -> None:
        super().__init__(
            RuntimeName.VLLM,
            "vLLM",
            config or default_runtime_configs()["vllm"],
            supervisor,
            filesystem_discovery=True,
        )
        self.managed_units = managed_units or ManagedUnitsConfig()
        self._http_get = http_get or _default_http_get
        self._probe_timeout_s = probe_timeout_s

    def _candidate_units(self) -> list[ManagedUnitConfig]:
        """Return all configured managed units to probe.

        Currently fixed to (vllm_tp,) — the only role we model in
        config.
        """
        return [
            self.managed_units.vllm_tp,
        ]

    def _probe_unit(self, unit: ManagedUnitConfig) -> list[ServedModel] | None:
        """Probe ``http://localhost:<port>/v1/models``.

        Returns the served models on success, ``None`` on failure (unit
        not running, port not bound, network blip — all treated the
        same). Entries without a usable ``id`` are skipped; ``root`` is
        optional and left ``None`` when the server omits it.
        """
        url = f"http://localhost:{unit.default_port}/v1/models"
        try:
            resp = self._http_get(url, self._probe_timeout_s)
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        served: list[ServedModel] = []
        for m in payload.get("data", []):
            if not isinstance(m, dict):
                continue  # tolerate malformed entries without crashing the probe
            mid = m.get("id")
            if not isinstance(mid, str):
                continue
            root = m.get("root")
            served.append(ServedModel(id=mid, root=root if isinstance(root, str) else None))
        return served

    def capabilities(self) -> dict[str, bool]:
        caps = super().capabilities()
        caps.update({"list_loaded_models": True, "version": True})
        return caps

    async def version(self) -> str | None:
        """Return the serving vLLM version from a managed unit's ``/version``."""

        def fetch(unit: ManagedUnitConfig) -> str | None:
            url = f"http://localhost:{unit.default_port}/version"
            try:
                resp = self._http_get(url, self._probe_timeout_s)
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                return None
            version = payload.get("version") if isinstance(payload, dict) else None
            return version if isinstance(version, str) else None

        for unit in self._candidate_units():
            # The probe is sync urllib; off-thread it so concurrent inventory
            # probes of other runtimes aren't stalled behind this timeout.
            version = await asyncio.to_thread(fetch, unit)
            if version is not None:
                return version
        return None

    async def list_loaded_models(self) -> list[Model] | None:
        """Return the models the managed unit(s) are actually serving now."""
        loaded: list[Model] = []
        any_answered = False
        for unit in self._candidate_units():
            served = await asyncio.to_thread(self._probe_unit, unit)
            if served is None:
                continue
            any_answered = True
            for model in served:
                loaded.append(
                    Model(
                        name=model.id,
                        runtime=RuntimeName.VLLM,
                        source=model.id,
                        path=model.root,
                        status=ModelStatus.DISCOVERED,
                        metadata={
                            "managed_unit": unit.unit_name,
                            "port": unit.default_port,
                        },
                    )
                )
        return loaded if any_answered else None

    async def health_check(self) -> AdapterStatus:
        """Report OK when any managed-unit port serves models.

        Probes each candidate unit's port for ``/v1/models``. If any
        responds with at least one served model, the adapter is OK.
        Otherwise falls back to the binary-on-PATH check (matters for
        non-systemd hosts where vLLM is invoked directly).
        """
        live: dict[str, list[str]] = {}
        for unit in self._candidate_units():
            served = await asyncio.to_thread(self._probe_unit, unit)
            if served:
                live[unit.unit_name] = [m.id for m in served]

        if live:
            served_summary = ", ".join(
                f"{unit}:{','.join(ids)}" for unit, ids in sorted(live.items())
            )
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"vLLM serving via managed unit(s): {served_summary}.",
                details={"served": live},
            )
        # No live HTTP endpoints — fall back to binary lookup so dev
        # hosts that run vLLM ad-hoc still get a useful answer.
        return await super().health_check()

    async def discover_models(self) -> list[Model]:
        """Discover models by probing managed-unit ports first.

        Each served model is registered with ``runtime=vllm``,
        ``status=DISCOVERED``, and metadata recording which unit
        serves it + what port. ``path`` carries the ``/v1/models``
        ``root`` field so the registry records *which checkpoint* a
        served name resolves to, not just the name. Filesystem
        discovery (config.json sweep) is then appended for unscheduled
        local checkpoints; duplicates by ``source`` are dropped,
        HTTP-discovered models win.
        """
        seen_source: set[str] = set()
        models: list[Model] = []

        for unit in self._candidate_units():
            served_models = await asyncio.to_thread(self._probe_unit, unit)
            if not served_models:
                continue
            for served in served_models:
                if served.id in seen_source:
                    continue
                seen_source.add(served.id)
                models.append(
                    Model(
                        name=served.id,
                        runtime=RuntimeName.VLLM,
                        source=served.id,
                        path=served.root,
                        status=ModelStatus.DISCOVERED,
                        metadata={
                            "managed_unit": unit.unit_name,
                            "port": unit.default_port,
                            "discovered_via": "http",
                        },
                    )
                )

        # Filesystem fallback — only for any model_ids not already
        # surfaced by HTTP discovery. Useful on dev hosts with cached
        # checkpoints that aren't currently being served.
        for fs_model in await super().discover_models():
            key = fs_model.source or fs_model.name
            if key in seen_source:
                continue
            seen_source.add(key)
            models.append(fs_model)

        return models


def _default_http_get(url: str, timeout: float) -> Any:
    """Production HTTP GET — patched in tests."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only
