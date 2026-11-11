"""Swap screen — one list of everything that can serve, one key to switch.

Same list and preview as ``llmctl run``: vLLM presets plus the lanes the
plugins list (FreeToken, llama.cpp, ...). Enter shows what will
stop and start and how long the last load took; confirming runs the swap in
a worker thread, with rollback on failure.

Bound to ``w`` from any screen — see :class:`MissionControlApp`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
from llmctl.services.op_lock import gpu_operation_lock
from llmctl.services.operations import describe_phase
from llmctl.services.session_swap import (
    pause_for_swap,
    resume_after_failed_swap,
    run_session_swap,
)
from llmctl.services.vllm_orchestrator import OrchestratorOptions, start_vllm_tp
from llmctl.services.vllm_slots import cuda_devices, engine_config, place
from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc, tone_color
from llmctl.tui._modals import ConfirmActionModal

#: Icon+text state markers: color is never the only signal (§3A).
_STATE_ICONS = {
    "serving": ("●", C_OK),
    "loading": ("◐", C_WARN),
    "stopped": ("○", C_MUTED),
}


def _state_cell(status: str) -> str:
    icon, color = _STATE_ICONS.get(status, ("·", C_MUTED))
    return f"[{color}]{icon} {esc(status)}[/]"


def run_recorded[T](
    key: str,
    model: str,
    work: Callable[[], T],
    outcome: Callable[[T], tuple[bool, str]],
) -> T:
    """Run ``work`` under the GPU operation lock, as a recorded operation.

    Every TUI path that swaps what holds the GPUs goes through here (§2B):
    phases marked by the swap services land on the record, and the outcome,
    including a rollback that could not restore, survives a TUI restart.
    ``outcome`` turns the result into ``(ok, error message)`` for the record.
    Raises :class:`~llmctl.services.op_lock.LockBusy` when another swap runs.
    """
    from llmctl.services import operations
    from llmctl.services.operations import OperationPhase

    settings = load_settings()
    db_url = getattr(settings, "database_url", None) or "sqlite:///:memory:"
    with gpu_operation_lock(purpose=f"run {model}", interface="tui"):
        store = operations.OperationStore(db_url)
        record, _ = store.begin(
            kind="swap",
            target=key,
            interface="tui",
            purpose=f"run {model}",
            request={"key": key, "model": model},
        )
        operations.bind_current(store, record.id or "")
        try:
            result = work()
            ok, message = outcome(result)
            if ok:
                operations.finish_current(OperationPhase.COMPLETED)
            else:
                operations.finish_current(OperationPhase.FAILED, error=message)
            return result
        except BaseException as exc:  # record interruption, then let the worker report it
            operations.interrupt_current(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            operations.clear_current()


def run_swap(state: SwapState, target: SwapTarget) -> tuple[bool, str]:
    """Perform the swap with terminal-silent deps. Returns ``(ok, message)``.

    Used by the Swap screen and the Models screen. The Presets screen's
    launch calls :func:`run_recorded` itself, around the orchestrator.
    """
    return run_recorded(
        target.key, target.model, lambda: _do_swap(state, target), lambda done: done
    )


def _do_swap(state: SwapState, target: SwapTarget) -> tuple[bool, str]:
    """The swap itself; called with the operation record already bound."""
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

    # A one-card vLLM preset gets a card and an engine (see vllm_slots).
    target = place(state, target, target.tp)
    # A lane that fits beside what runs (measured VRAM) stops nothing. Free
    # VRAM is read now: the state was gathered before the confirm dialog.
    if target.kind == "lane":
        from llmctl.services.vram_fit import free_mib_now

        state.free_mib = free_mib_now()
    fit = state.fits_alongside(target) if target.kind == "lane" else None
    # llama.cpp processes llmctl started are in the way of vLLM and lanes too.
    paused: list[Any] = []
    if fit is None:
        paused, error = pause_for_swap(state, target, settings=settings, log=deps.log)
        if error:
            again = resume_after_failed_swap(paused, settings=settings, log=deps.log)
            return False, _failure(error, again)

    if target.kind == "lane":
        assert target.lane is not None
        result = swap_to_lane(
            target.lane,
            other_active=state.lanes_in_the_way(target, fit),
            vllm_active=state.vllm_in_the_way(target),
            vllm_unit=state.vllm_unit,
            deps=deps,
            share_gpus=fit is not None,
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
            managed_unit=engine_config(settings, target),
            defaults=settings.vllm.defaults,
            fleet=settings.managed_units.fleet,
            options=OrchestratorOptions(dry_run=False, gpus=cuda_devices(target)),
        )

    # Every lane for a preset that takes every card; only those on its card otherwise.
    vllm, lanes = swap_to_vllm(state.lanes_in_the_way(target), _start, deps=deps)
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
    from llmctl.services.operations import note_restore_failure

    text = esc(reason or "unknown error")
    if lanes.restored:
        text += f"\nStarted again: {esc(', '.join(lanes.restored))}"
    if lanes.restore_failed:
        note_restore_failure(lanes.restore_failed)
        text += f"\nStill stopped: {esc(', '.join(lanes.restore_failed))} — start it by hand."
    return text


class SwapScreen(DataScreen):
    """Every swap target with its engine, state and last load time.

    Above the list: the current operation (phase, elapsed, recovery), so a
    long swap explains itself instead of showing a frozen table (§3C).
    """

    BINDINGS = [
        Binding("enter", "swap_selected", "Swap", show=True, priority=True),
        Binding("ctrl+r", "refresh_now", "Refresh", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._state: SwapState | None = None
        #: The key of the row under the cursor, kept across refreshes so a
        #: refresh or list change cannot retarget Enter to another model.
        self._anchor_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Run  -  [{C_MUTED}]enter = run the selected model (shows what stops "
            f"first), ctrl+r = refresh. Lanes come from the installed plugins.[/]",
            classes="panel safe",
            id="swap-title",
        )
        yield Static(f"[{C_MUTED}]no operation[/]", classes="panel", id="swap-operation")
        table: DataTable[str] = DataTable(id="swap-table", cursor_type="row")
        table.add_columns("Model", "Engine", "Name", "Cards", "State", "Last load")
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Build the list and probe what is serving (worker thread)."""
        return gather_state(load_settings())

    def render_data(self, data: Any) -> None:
        self._state = data
        self._render_operation()
        table = self.query_one("#swap-table", DataTable)
        table.clear()
        if data is None:
            return
        anchored_row: int | None = None
        for index, row in enumerate(data.catalog):
            status = data.state_of(row)
            if row.key == self._anchor_key:
                anchored_row = index
            seconds = last_load_s(row.key)
            table.add_row(
                esc(row.model),
                row.engine,
                esc(row.key),
                cards_label(row),
                _state_cell(status),
                format_duration(seconds) if seconds else f"[{C_MUTED}]—[/]",
            )
        if data.catalog:
            # Re-locate the anchored row (not the old row number) so a
            # refresh or a changed list keeps the same target selected.
            target_row = anchored_row if anchored_row is not None else 0
            table.move_cursor(row=min(target_row, len(data.catalog) - 1))
            self._anchor_key = data.catalog[table.cursor_row].key

    def _render_operation(self) -> None:
        """Show the current operation above the list (§3C)."""
        op = self.query_data(_data.get_current_operation, what="Operation lookup")
        panel = self.query_one("#swap-operation", Static)
        if op is None:
            panel.update(f"[{C_MUTED}]no operation[/]")
            return
        view = describe_phase(op.get("phase"))
        color = tone_color(view.tone)
        if op.get("terminal"):
            text = (
                f"[{color}]{view.icon} last operation: {esc(view.label)} — "
                f"{esc(op.get('kind', '?'))} {esc(op.get('target', '?'))}[/]"
            )
            if view.hint:
                text += f"\n  [{color}]{view.icon} {esc(view.hint)}[/]"
        else:
            elapsed = op.get("elapsed_s")
            elapsed_label = (
                f", {elapsed:.0f}s elapsed" if isinstance(elapsed, (int, float)) else ""
            )
            text = (
                f"[{color}]{view.icon} {esc(view.label)}{elapsed_label}[/] — "
                f"{esc(op.get('kind', '?'))} {esc(op.get('target', '?'))} "
                f"[{C_MUTED}]via {esc(op.get('interface') or '?')}[/]"
            )
        recovery = op.get("recovery") or {}
        guidance = recovery.get("guidance")
        if guidance:
            text += f"\n  [{C_ERR}]! {esc(guidance)}[/]"
        not_restored = recovery.get("not_restored") or []
        if not_restored:
            text += (
                f"\n  [{C_ERR}]! not restored: "
                f"{esc(', '.join(map(str, not_restored)))}[/]"
            )
        panel.update(text)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the highlighted row by its stable key, not row number."""
        state = self._state
        if state is None:
            return
        row = event.cursor_row
        if 0 <= row < len(state.catalog):
            self._anchor_key = state.catalog[row].key

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
        target = place(state, state.catalog[row], state.catalog[row].tp)
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
