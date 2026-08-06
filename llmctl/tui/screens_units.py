"""Managed Units screen — live status of llmctl-controlled systemd units.

Shows what the CLI's ``llmctl status`` shows, plus live information
the CLI doesn't surface: which units are currently active in systemd,
and what models each one is serving right now (via the same
``/v1/models`` probe Phase A wired into VLLMAdapter).

Bound to ``u`` from any screen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.config import load_settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.tui._base import C_ACCENT, C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc

PROBE_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class _UnitRow:
    """One row to render: a managed unit, with live status."""

    role: str
    unit_name: str
    env_path: str
    #: False when ``resolve_env_file()`` names a file that is not there. The
    #: CLI stopped rendering phantom paths in 10d44ac1; this screen must too,
    #: or operators edit a file nothing reads.
    env_exists: bool
    port: int
    is_active: bool
    #: Tri-state from the shared probe: a list (possibly empty) means the
    #: endpoint answered, ``None`` means it could not be reached.
    served: list[str] | None
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
        # None means "use the shared default in services.backends". Keeping a
        # private transport here is what let this screen reach :8003 by a path
        # the rest of the codebase could not see or stub.
        self._http_get = http_get

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
        from llmctl.services.units import unit_rows

        sysctl = self._systemctl or SystemctlRunner()
        # Facts shared with `llmctl status` come from one producer, so the two
        # surfaces cannot describe the same unit differently. `is_active` is
        # this screen's own addition -- the CLI table does not show it.
        return [
            _UnitRow(
                role=r.role,
                unit_name=r.unit_name,
                env_path=str(r.env_file),
                env_exists=r.env_file_exists,
                port=r.port,
                is_active=self._is_active_safe(r.unit_name, sysctl),
                served=r.served,
                role_kind="managed",
            )
            for r in unit_rows(load_settings(), http_get=self._http_get)
        ]

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
            # Three distinct outcomes, not two. Collapsing "answered with
            # nothing loaded" into "could not be reached" made an idle unit and
            # a dead one look identical here, and made both disagree with
            # `llmctl status`.
            if r.served:
                served_cell = f"[{C_ACCENT}]{esc(', '.join(r.served))}[/]"
            elif r.served == []:
                served_cell = f"[{C_OK}]idle (none loaded)[/]"
            elif r.is_active:
                # Active in systemd but the endpoint did not answer: still loading.
                served_cell = f"[{C_WARN}]starting?[/]"
            else:
                served_cell = f"[{C_MUTED}]-[/]"
            env_cell = esc(r.env_path) if r.env_exists else f"[{C_MUTED}]—[/]"
            table.add_row(
                f"[{C_MUTED}]unit[/] {esc(r.role)}",
                esc(r.unit_name),
                active_cell,
                str(r.port),
                served_cell,
                env_cell,
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
