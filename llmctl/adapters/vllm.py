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

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

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

    def _probe_unit(self, unit: ManagedUnitConfig) -> list[str] | None:
        """Probe ``http://localhost:<port>/v1/models``.

        Returns the served model IDs on success, ``None`` on failure
        (unit not running, port not bound, network blip — all treated
        the same).
        """
        url = f"http://localhost:{unit.default_port}/v1/models"
        try:
            resp = self._http_get(url, self._probe_timeout_s)
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        ids: list[str] = []
        for m in payload.get("data", []):
            if not isinstance(m, dict):
                continue  # tolerate malformed entries without crashing the probe
            mid = m.get("id")
            if isinstance(mid, str):
                ids.append(mid)
        return ids

    async def health_check(self) -> AdapterStatus:
        """Report OK when any managed-unit port serves models.

        Probes each candidate unit's port for ``/v1/models``. If any
        responds with at least one served model, the adapter is OK.
        Otherwise falls back to the binary-on-PATH check (matters for
        non-systemd hosts where vLLM is invoked directly).
        """
        live: dict[str, list[str]] = {}
        for unit in self._candidate_units():
            ids = self._probe_unit(unit)
            if ids:
                live[unit.unit_name] = ids

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
        serves it + what port. Filesystem discovery (config.json sweep)
        is then appended for unscheduled local checkpoints; duplicates
        by ``source`` are dropped, HTTP-discovered models win.
        """
        seen_source: set[str] = set()
        models: list[Model] = []

        for unit in self._candidate_units():
            ids = self._probe_unit(unit)
            if not ids:
                continue
            for served in ids:
                if served in seen_source:
                    continue
                seen_source.add(served)
                models.append(
                    Model(
                        name=served,
                        runtime=RuntimeName.VLLM,
                        source=served,
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
