"""Managed Units screen — live status of llmctl-controlled systemd units.

Shows what the CLI's ``llmctl status`` shows, plus live information
the CLI doesn't surface: which units are currently active in systemd,
and what models each one is serving right now (via the same
``/v1/models`` probe Phase A wired into VLLMAdapter).

Bound to ``u`` from any screen.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.config import ManagedUnitConfig, load_settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.tui._base import C_ACCENT, C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc

PROBE_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class _UnitRow:
    """One row to render: a managed unit, with live status."""

    role: str
    unit_name: str
    env_path: str
    port: int
    is_active: bool
    served: list[str]
    role_kind: str  # "managed"


class UnitsScreen(DataScreen):
    """Live table of managed units with per-unit probes."""

    BINDINGS = [
        ("ctrl+r", "refresh_now", "Refresh"),
    ]

    def __init__(
        self,
        *,
        systemctl: SystemctlRunner | None = None,
        http_get: Callable[[str, float], Any] | None = None,
    ) -> None:
        super().__init__()
        # Both deps are injectable so tests don't hit real systemd /
        # real HTTP. Production wiring uses sensible defaults.
        self._systemctl = systemctl
        self._http_get = http_get or _default_http_get

    def compose(self) -> ComposeResult:
        """Compose the units table chrome with screen-scoped Header/Footer."""
        yield Header()
        yield Static(
            f"Managed Units  -  [{C_MUTED}]probes vllm-tp every 3s; "
            f"ctrl+r = refresh now[/]",
            classes="panel safe",
            id="units-title",
        )
        table: DataTable[str] = DataTable(id="units-table", cursor_type="row")
        table.add_columns(
            "Role", "Unit", "Active", "Port", "Served (live)", "Env file"
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Probe each managed unit in a worker thread.

        Returns a list of :class:`_UnitRow` ready to render. Per-port
        probes use a 1.5s timeout so a down unit adds at most ~1.5s to
        the refresh — well within the 3s auto-refresh cadence's
        tolerance (worker is exclusive so back-pressured refreshes
        replace prior ones).
        """
        settings = load_settings()
        sysctl = self._systemctl or SystemctlRunner()
        rows: list[_UnitRow] = []

        # Managed-unit roles (vllm-tp)
        managed = [
            ("vllm-tp", settings.managed_units.vllm_tp),
        ]
        for role, cfg in managed:
            rows.append(self._row_for_managed(role, cfg, sysctl))

        return rows

    def render_data(self, data: Any) -> None:
        """Render the units table with color-coded state."""
        rows: list[_UnitRow] = list(data or [])
        table = self.query_one("#units-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        for r in rows:
            active_cell = (
                f"[{C_OK}]active[/]" if r.is_active else f"[{C_MUTED}]inactive[/]"
            )
            if r.served:
                served_cell = f"[{C_ACCENT}]{esc(', '.join(r.served))}[/]"
            elif r.is_active:
                # Unit running but no /v1/models — likely starting up
                served_cell = f"[{C_WARN}]starting?[/]"
            else:
                served_cell = f"[{C_MUTED}]-[/]"
            table.add_row(
                f"[{C_MUTED}]unit[/] {esc(r.role)}",
                esc(r.unit_name),
                active_cell,
                str(r.port),
                served_cell,
                esc(r.env_path),
            )
        if not rows:
            table.add_row(
                f"[{C_ERR}]No units configured[/]", "-", "-", "-", "-", "-"
            )
        if 0 <= cursor < len(rows):
            table.move_cursor(row=cursor)

    def action_refresh_now(self) -> None:
        """Manual refresh shortcut."""
        self.refresh_data()

    def _row_for_managed(
        self,
        role: str,
        cfg: ManagedUnitConfig,
        sysctl: SystemctlRunner,
    ) -> _UnitRow:
        """Build a row for one managed unit."""
        is_active = self._is_active_safe(cfg.unit_name, sysctl)
        served = self._probe_served(cfg.default_port)
        return _UnitRow(
            role=role,
            unit_name=cfg.unit_name,
            env_path=str(cfg.resolve_env_file()),
            port=cfg.default_port,
            is_active=is_active,
            served=served,
            role_kind="managed",
        )

    def _is_active_safe(self, unit: str, sysctl: SystemctlRunner) -> bool:
        """Return True iff systemctl reports active; False on any error.

        Wraps :meth:`SystemctlRunner.is_active` to swallow subprocess
        spawn failures (no systemd in container / etc.) — the UI just
        shows 'inactive', the user isn't crashed out of the screen.
        """
        try:
            return sysctl.is_active(unit)
        except OSError:
            return False

    def _probe_served(self, port: int) -> list[str]:
        """Probe ``http://localhost:<port>/v1/models``; ``[]`` on failure."""
        url = f"http://localhost:{port}/v1/models"
        try:
            resp = self._http_get(url, PROBE_TIMEOUT_S)
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return []
        ids: list[str] = []
        for m in payload.get("data", []):
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if isinstance(mid, str):
                ids.append(mid)
        return ids


def _default_http_get(url: str, timeout: float) -> Any:
    """Production HTTP GET — patched in tests."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only
