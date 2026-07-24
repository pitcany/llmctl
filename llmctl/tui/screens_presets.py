"""Presets screen — the daily-driver TUI surface for vLLM operators.

Shows every preset alias from ``~/.config/llmctl/presets/*.yaml``. Pressing
enter on a row pops a launch picker (TP fleet); confirming kicks off the
orchestrator call in a worker thread so the UI stays responsive during
the 1-3 minute vLLM cold-start.

Bound to ``p`` from any screen — see :class:`MissionControlApp`.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.config import load_settings
from llmctl.presets import Model as PresetModel
from llmctl.presets import PresetSchemaError
from llmctl.services.preset_loader import PresetView
from llmctl.services.vllm_orchestrator import (
    OrchestratorOptions,
    OrchestratorResult,
    start_vllm_tp,
)
from llmctl.tui import _data
from llmctl.tui._base import C_ACCENT, C_ERR, C_MUTED, C_OK, DataScreen, esc
from llmctl.tui._modals_presets import (
    PresetFormModal,
    PresetLaunchModal,
    PresetLaunchTarget,
)
from llmctl.tui._modals_registry import (
    CloneModal,
    CloneRequest,
    ConfirmDelete,
    DeleteModal,
)


def _format_link_cell(view: PresetView) -> str:
    """Render the linkage badge for a preset row.

    ``●`` linked (explicit model_ref), ``○`` auto-matched,
    ``⚠`` model_ref set but registry has no such id, ``-`` no link.
    Falls back to ``-`` when no resolution has been attempted.
    """
    state = view.linkage_state
    if state is None or state == "unlinked":
        return f"[{C_MUTED}]-[/]"
    if state == "missing":
        return f"[{C_ERR}]⚠ missing[/]"
    glyph = "●" if state == "explicit" else "○"
    color = C_OK if state == "explicit" else C_ACCENT
    label = view.linked_model_name or (view.linked_model_id or "")
    return f"[{color}]{glyph}[/] {esc(label)}"


class PresetsScreen(DataScreen):
    """Live preset table with one-key launch into the TP fleet unit."""

    BINDINGS = [
        Binding("enter", "launch_selected", "Launch", show=True),
        Binding("ctrl+r", "refresh_now", "Refresh", show=True),
        Binding("a", "add_preset", "Add", show=True),
        Binding("e", "edit_preset", "Edit", show=True),
        Binding("c", "clone_preset", "Clone", show=True),
        Binding("d", "delete_preset", "Delete", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        #: alias -> PresetView, populated on every render so action_launch
        #: can look up the selected row without re-fetching.
        self._views_by_alias: dict[str, PresetView] = {}
        self._row_aliases: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the preset table chrome with screen-scoped Header/Footer.

        Each screen yields its own Header/Footer because Textual's
        ``push_screen`` covers the App-level chrome.
        """
        yield Header()
        yield Static(
            f"Presets  -  [{C_MUTED}]enter = launch, a = add, e = edit, "
            f"c = clone, d = delete, ctrl+r = refresh[/]",
            classes="panel safe",
            id="presets-title",
        )
        table: DataTable[str] = DataTable(id="presets-table", cursor_type="row")
        table.add_columns(
            "Alias",
            "Served name",
            "Model ID",
            "Family",
            "Size (B)",
            "TP",
            "Quant",
            "Linked",
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Load all preset views, including registry linkage (worker thread)."""
        return _data.get_preset_views_with_links()

    def render_data(self, data: Any) -> None:
        """Render the preset table, preserving the cursor position."""
        views: list[PresetView] = list(data or [])
        table = self.query_one("#presets-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._views_by_alias = {v.alias: v for v in views}
        self._row_aliases = [v.alias for v in views]

        if not views:
            table.add_row(
                "[dim]No presets[/]",
                "(write one to ~/.config/llmctl/presets/<alias>.yaml)",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
            return

        for v in views:
            table.add_row(
                f"[{C_ACCENT}]{esc(v.alias)}[/]",
                esc(v.served_name),
                esc(v.model_id),
                esc(v.family or "-"),
                f"{v.param_count_b:.0f}" if v.param_count_b else "-",
                str(v.tensor_parallel),
                esc(v.quantization),
                _format_link_cell(v),
            )
        if 0 <= cursor < len(self._row_aliases):
            table.move_cursor(row=cursor)

    def _selected_alias(self) -> str | None:
        """Return the alias under the cursor or ``None`` if no row is selected."""
        if not self._row_aliases:
            return None
        table = self.query_one("#presets-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._row_aliases):
            return self._row_aliases[row]
        return None

    def action_refresh_now(self) -> None:
        """Manual refresh shortcut (ctrl+r) — same as auto-refresh."""
        self.refresh_data()

    def action_launch_selected(self) -> None:
        """Open the launch-target picker for the selected preset."""
        alias = self._selected_alias()
        if alias is None:
            self.app.notify("No preset selected.", severity="warning")
            return
        view = self._views_by_alias.get(alias)
        if view is None:
            return

        def _on_pick(target: PresetLaunchTarget | None) -> None:
            if target is None:
                return
            self._launch(alias, target)

        self.app.push_screen(PresetLaunchModal(view), _on_pick)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Treat row activation (enter/click) the same as the bound action."""
        self.action_launch_selected()

    def _launch(self, alias: str, target: PresetLaunchTarget) -> None:
        """Kick off the orchestrator call in a worker thread."""
        settings = load_settings()
        options = OrchestratorOptions(dry_run=False)
        label = f"vLLM TP: {alias}"

        def _run() -> OrchestratorResult:
            return start_vllm_tp(
                alias,
                managed_unit=settings.managed_units.vllm_tp,
                defaults=settings.vllm.defaults,
                fleet=settings.managed_units.fleet,
                options=options,
            )

        self.app.notify(f"Starting {label}...", title="Launch")
        self.run_action_worker(_run, lambda res: self._after_launch(label, res))

    def action_add_preset(self) -> None:
        """Open the add-preset form and persist on submit."""

        def _on_close(model: PresetModel | None) -> None:
            if model is None:
                return
            self.run_action_worker(
                lambda: _data.add_preset(model), self._after_mutation
            )

        self.app.push_screen(PresetFormModal(), _on_close)

    def action_edit_preset(self) -> None:
        """Open the cursor-row preset's YAML in ``$EDITOR``.

        Suspends Textual so the editor inherits the real TTY, then
        revalidates the file on return. Preset YAMLs carry too many
        knobs (and field-level comments worth preserving) for a TUI
        form to win against a real editor; the form path remains for
        ``a`` (add) where the user is starting from scratch.
        """
        alias = self._selected_alias()
        if alias is None:
            self.app.notify("No preset selected.", severity="warning")
            return
        view = self._views_by_alias.get(alias)
        path = view.source_path if view else None
        if path is None:
            self.app.notify(
                f"Preset {alias!r} has no resolved source path.",
                severity="warning",
            )
            return
        if not path.exists():
            self.app.notify(
                f"Preset file vanished: {path}", severity="error"
            )
            self.refresh_data()
            return

        try:
            with self.app.suspend():
                _data.run_editor_on_preset(path)
        except PresetSchemaError as exc:
            self.app.notify(
                f"Preset YAML is invalid; on-disk file left as-is.\n{exc}",
                severity="error",
                title=f"Edit {alias}",
            )
            self.refresh_data()
            return
        except OSError as exc:
            self.app.notify(
                f"Failed to launch editor: {exc}",
                severity="error",
                title="Edit preset",
            )
            return

        self.app.notify(f"Saved {alias} ({path}).", title="Edit preset")
        self.refresh_data()

    def action_clone_preset(self) -> None:
        """Clone the cursor-row preset under a new alias."""
        alias = self._selected_alias()
        if alias is None:
            self.app.notify("No preset selected.", severity="warning")
            return

        def _on_close(req: CloneRequest | None) -> None:
            if req is None:
                return
            self.run_action_worker(
                lambda: _data.clone_preset(req.source_id, req.new_name),
                self._after_mutation,
            )

        self.app.push_screen(CloneModal(alias, alias), _on_close)

    def action_delete_preset(self) -> None:
        """Confirm and delete the cursor-row preset YAML file."""
        alias = self._selected_alias()
        if alias is None:
            self.app.notify("No preset selected.", severity="warning")
            return

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.delete_preset(alias),
                lambda removed: self._after_delete(alias, removed),
            )

        self.app.push_screen(
            DeleteModal(
                f"preset '{alias}'",
                "The preset YAML file is unlinked from disk (both the canonical "
                "and legacy preset directories). There is no registry row and no "
                "soft-delete: this cannot be undone from llmctl.",
            ),
            _on_close,
        )

    def _after_mutation(self, _result: object) -> None:
        """Refresh the table after add/edit/clone completes."""
        self.app.notify("Preset saved.", title="Presets")
        self.refresh_data()

    def _after_delete(self, alias: str, removed: list[object]) -> None:
        """Refresh and notify after a delete completes."""
        if removed:
            self.app.notify(f"Deleted preset {alias!r}.", title="Presets")
        else:
            self.app.notify(
                f"No on-disk file found for {alias!r}.", severity="warning"
            )
        self.refresh_data()

    def _after_launch(self, label: str, result: OrchestratorResult) -> None:
        """Surface success/failure as a notification."""
        if result.ok:
            self.app.notify(
                f"[{C_OK}]{label} ready[/] on port {result.spec.port}",
                title="Launch succeeded",
            )
        else:
            reason = "unknown"
            if result.fleet_failed:
                reason = f"fleet preflight failed: {', '.join(result.fleet_failed)}"
            elif result.restart is not None and result.restart.error:
                reason = result.restart.error
            self.app.notify(
                f"[{C_ERR}]{label} failed[/]: {reason}",
                severity="error",
                title="Launch failed",
            )
