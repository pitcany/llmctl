"""Swap screen — one list of everything that can serve, one key to switch.

Same list and preview as ``llmctl swap``: vLLM presets plus the lanes the
plugins list (FreeToken, llama.cpp, ...). Enter shows what will
stop and start and how long the last load took; confirming runs the swap in
a worker thread, with rollback on failure.

Bound to ``w`` from any screen — see :class:`MissionControlApp`.
"""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.config import load_settings
from llmctl.services.lane_swap import quiet_deps, swap_to_lane, swap_to_vllm
from llmctl.services.lanes import (
    SwapState,
    SwapTarget,
    cards_label,
    format_duration,
    gather_state,
    last_load_s,
    record_load,
)
from llmctl.services.session_swap import (
    pause_for_swap,
    resume_after_failed_swap,
    run_session_swap,
)
from llmctl.services.vllm_orchestrator import OrchestratorOptions, start_vllm_tp
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc
from llmctl.tui._modals import ConfirmActionModal


def run_swap(state: SwapState, target: SwapTarget) -> tuple[bool, str]:
    """Perform the swap with terminal-silent deps. Returns ``(ok, message)``.

    Shared by the Swap screen and the Presets screen's launch so both free
    the lanes before vLLM takes the GPUs.
    """
    deps = quiet_deps()
    settings = load_settings()
    if target.kind == "session":
        done = run_session_swap(state, target, settings=settings, deps=deps)
        if done.ok:
            sid = (getattr(done.session, "id", "") or "")[:8]
            url = getattr(done.session, "endpoint_url", None) or "its port"
            return True, (
                f"{target.model} answers at {url} ({format_duration(done.seconds)}). "
                f"Stop it: llmctl stop {sid}"
            )
        return False, _failure(done.error, done)

    # llama.cpp processes llmctl started are in the way of vLLM and lanes too.
    paused, error = pause_for_swap(state, target, settings=settings, log=deps.log)
    if error:
        again = resume_after_failed_swap(paused, settings=settings, log=deps.log)
        return False, _failure(error, again)

    if target.kind == "lane":
        assert target.lane is not None
        result = swap_to_lane(
            target.lane,
            other_active=state.lanes_in_the_way(target),
            vllm_active=state.vllm_active,
            vllm_unit=state.vllm_unit,
            deps=deps,
        )
        if result.ok:
            return True, f"{target.model} ready ({format_duration(result.seconds)})"
        again = resume_after_failed_swap(paused, settings=settings, log=deps.log)
        result.restored += again.restored
        result.restore_failed += again.restore_failed
        return False, _failure(result.error, result)

    started = time.monotonic()

    def _start() -> Any:
        return start_vllm_tp(
            target.key,
            managed_unit=settings.managed_units.vllm_tp,
            defaults=settings.vllm.defaults,
            fleet=settings.managed_units.fleet,
            options=OrchestratorOptions(dry_run=False),
        )

    vllm, lanes = swap_to_vllm(state.active_lanes(), _start, deps=deps)
    if vllm is None:
        again = resume_after_failed_swap(paused, settings=settings, log=deps.log)
        lanes.restored += again.restored
        lanes.restore_failed += again.restore_failed
        return False, _failure(lanes.error, lanes)
    if getattr(vllm, "ok", False):
        record_load(target.key, time.monotonic() - started)
        message = f"{target.model} ready on vLLM"
        failed = getattr(vllm, "companions_failed", [])
        if failed:
            message += f"\nNot started again: {esc(', '.join(failed))} — start by hand."
        return True, message
    restart = getattr(vllm, "restart", None)
    reason = getattr(restart, "error", None) or "vLLM start did not complete cleanly"
    again = resume_after_failed_swap(paused, settings=settings, log=deps.log)
    lanes.restored += again.restored
    lanes.restore_failed += again.restore_failed
    return False, _failure(reason, lanes)


def _failure(reason: str | None, lanes: Any) -> str:
    text = esc(reason or "unknown error")
    if lanes.restored:
        text += f"\nStarted again: {esc(', '.join(lanes.restored))}"
    if lanes.restore_failed:
        text += f"\nStill stopped: {esc(', '.join(lanes.restore_failed))} — start it by hand."
    return text


class SwapScreen(DataScreen):
    """Every swap target with its engine, state and last load time."""

    BINDINGS = [
        Binding("enter", "swap_selected", "Swap", show=True, priority=True),
        Binding("ctrl+r", "refresh_now", "Refresh", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._state: SwapState | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Swap  -  [{C_MUTED}]enter = swap to the selected model (shows what stops "
            f"first), ctrl+r = refresh. Lanes come from the installed plugins.[/]",
            classes="panel safe",
            id="swap-title",
        )
        table: DataTable[str] = DataTable(id="swap-table", cursor_type="row")
        table.add_columns("Model", "Engine", "Name", "Cards", "State", "Last load")
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Build the list and probe what is serving (worker thread)."""
        return gather_state(load_settings())

    def render_data(self, data: Any) -> None:
        self._state = data
        table = self.query_one("#swap-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        if data is None:
            return
        for row in data.catalog:
            status = data.state_of(row)
            color = {"serving": C_OK, "loading": C_WARN}.get(status)
            seconds = last_load_s(row.key)
            table.add_row(
                esc(row.model),
                row.engine,
                esc(row.key),
                cards_label(row),
                f"[{color}]{status}[/]" if color else f"[{C_MUTED}]—[/]",
                format_duration(seconds) if seconds else f"[{C_MUTED}]—[/]",
            )
        if data.catalog:
            table.move_cursor(row=min(cursor, len(data.catalog) - 1))

    def action_refresh_now(self) -> None:
        self.refresh_data()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_swap_selected()

    def action_swap_selected(self) -> None:
        state = self._state
        if state is None or not state.catalog:
            self.app.notify("Nothing to swap to yet.", severity="warning")
            return
        row = self.query_one("#swap-table", DataTable).cursor_row
        if not 0 <= row < len(state.catalog):
            return
        target = state.catalog[row]
        preview = state.preview(target)
        if preview.already_serving and target.kind in ("lane", "session"):
            self.app.notify(preview.lines()[0], title="Swap")
            return

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.app.notify(f"Swapping to {esc(target.model)}...", title="Swap")
            self.run_action_worker(lambda: run_swap(state, target), self._after_swap)

        self.app.push_screen(
            ConfirmActionModal(
                f"Swap to {target.model} ({target.engine})",
                "\n".join(preview.lines()),
                confirm_label="Swap",
            ),
            _on_confirm,
        )

    def _after_swap(self, outcome: tuple[bool, str]) -> None:
        ok, message = outcome
        if ok:
            self.app.notify(f"[{C_OK}]{message}[/]", title="Swap succeeded")
        else:
            self.app.notify(
                f"[{C_ERR}]Swap failed[/]: {message}", severity="error", title="Swap failed"
            )
        self.refresh_data()
