"""Models registry TUI screen with live data and start/scan actions."""

from __future__ import annotations

import functools
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.db import ModelStatus
from llmctl.schemas import ModelCreate, ModelUpdate
from llmctl.services.registry import is_missing
from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc
from llmctl.tui._modals import ConfirmActionModal, LaunchPlanModal
from llmctl.tui._modals_registry import (
    AddModelModal,
    AddRequest,
    CloneModal,
    CloneRequest,
    ConfirmDelete,
    DeleteModal,
    ModelFormModal,
)


class ModelsScreen(DataScreen):
    """Model registry screen: lists models and plans sessions."""

    BINDINGS = [
        Binding("enter", "start_model", "Serve", show=True, priority=True),
        Binding("ctrl+s", "scan", "Scan", show=True),
        Binding("a", "add_model", "Add", show=True),
        Binding("e", "edit_model", "Edit", show=True),
        Binding("delete", "delete_model", "Remove", show=True, key_display="del"),
        Binding("x", "prune_missing", "Prune missing", show=True),
        Binding("c", "clone_model", "Clone", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        #: model id -> (runtime_value, backend_available)
        self._meta: dict[str, tuple[str, bool]] = {}
        self._missing_count: int = 0

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Open the launch-plan preview when a row is activated (enter/click)."""
        self.action_start_model()

    def compose(self) -> ComposeResult:
        """Compose the models table with screen-scoped Header/Footer."""
        yield Header()
        yield Static(
            f"Models Registry  -  [{C_MUTED}]a = add (any engine), del = remove, "
            "enter = serve (any engine), ctrl+s = scan[/]",
            classes="panel safe",
            id="models-title",
        )
        table: DataTable[str] = DataTable(id="models-table", cursor_type="row")
        table.add_columns(
            "ID",
            "Name",
            "Runtime",
            "Backend",
            "Status",
            "Quant",
            "Path",
            "Presets",
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Return the current model list, backend availability, preset counts."""
        return {
            "models": _data.get_models(),
            "availability": _data.get_backend_map(),
            "preset_counts": _data.get_preset_count_by_model(),
            "missing_count": _data.get_missing_count(),
        }

    def render_data(self, data: Any) -> None:
        """Render the models table, dimming models with a missing backend."""
        models = data["models"]
        self._missing_count = data.get(
            "missing_count",
            sum(1 for m in models if is_missing(m)),
        )
        availability = data["availability"]
        preset_counts: dict[str, int] = data.get("preset_counts", {})
        table = self.query_one("#models-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        self._meta = {}
        for model in models:
            mid = model.id or ""
            runtime = model.runtime.value
            available = availability.get(runtime, True)
            missing = is_missing(model)
            self._ids.append(mid)
            self._meta[mid] = (runtime, available)
            backend_cell = f"[{C_OK}]ready[/]" if available else f"[{C_ERR}]no binary[/]"
            if available and not missing:
                name_cell = esc(model.name)
            else:
                name_cell = f"[{C_MUTED}]{esc(model.name)}[/]"
            status_cell = (
                f"[{C_WARN}]{ModelStatus.MISSING.value}[/]" if missing else model.status.value
            )
            count = preset_counts.get(mid, 0)
            preset_cell = str(count) if count else f"[{C_MUTED}]-[/]"
            table.add_row(
                mid[:8],
                name_cell,
                runtime,
                backend_cell,
                status_cell,
                esc(model.quantization or "-"),
                # Truncate first, then escape: slicing a balanced path can
                # strand a closing tag that only escaping renders harmless.
                esc((model.path or "-")[-32:]),
                preset_cell,
            )
        if not models:
            table.add_row(
                "-", "No models yet: a to add one, ctrl+s to scan", "-", "-", "-", "-", "-", "-"
            )
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)

    def _selected_id(self) -> str | None:
        """Return the model id under the cursor, if any."""
        table = self.query_one("#models-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._ids):
            return self._ids[row]
        return None

    def action_start_model(self) -> None:
        """Serve the cursor row the way ``llmctl run NAME`` would.

        A vLLM checkpoint swaps to its preset (offering to write one first), a
        llama.cpp row swaps the GPUs over to a llama-server, and Ollama / LM
        Studio say they load on demand. Routing probes what is serving, so it runs off
        the UI thread.
        """
        model_id = self._selected_id()
        if not model_id:
            self.app.notify("No model selected.", severity="warning")
            return
        runtime, available = self._meta.get(model_id, ("", True))
        if not available:
            hint = _data.BACKEND_INSTALL_HINTS.get(runtime, "")
            message = f"'{runtime}' backend is not installed." + (f" {hint}" if hint else "")
            self.app.notify(message, severity="warning", title="Backend unavailable")
            return
        self.run_action_worker(
            lambda: self._route(model_id),
            lambda routed: self._serve(model_id, *routed),
        )

    @staticmethod
    def _route(model_id: str) -> tuple[Any, Any, Any]:
        """Worker: the route, the swap state, and a launch plan when one applies.

        Everything a follow-up needs is fetched here, in one worker: a second
        worker started from this one's callback is refused as "busy".
        """
        served, state = _data.serve_route(model_id)
        plan = _data.get_launch_plan(model_id) if served and served.kind == "start" else None
        return served, state, plan

    def _serve(self, model_id: str, served: Any, state: Any, plan: Any = None) -> None:
        """Act on a :class:`~llmctl.services.serve.ServeRoute`."""
        if served is None:
            self.app.notify("Nothing changed: the model was not found.", severity="warning")
            return
        if served.kind == "start":
            self._show_plan(model_id, plan)
        elif served.kind in ("preset", "session", "lane") and served.target is not None:
            self._confirm_swap(state, served.target)
        elif served.kind == "no-preset":
            self._offer_starter_preset(model_id, served.model.name)
        else:
            self.app.notify(esc(served.message), title="Serve")

    def _confirm_swap(self, state: Any, target: Any) -> None:
        """The Swap screen's confirm, for this row's preset or llama-server."""
        from llmctl.services.vllm_slots import place
        from llmctl.tui.screens_swap import run_swap

        target = place(state, target, target.tp)
        preview = state.preview(target)
        if preview.already_serving and target.kind in ("lane", "session"):
            self.app.notify(esc(preview.lines()[0]), title="Swap")
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
            self.app.notify(f"[{C_OK}]{esc(message)}[/]", title="Swap succeeded")
        else:
            self.app.notify(
                f"[{C_ERR}]Swap failed[/]: {esc(message)}", severity="error", title="Swap failed"
            )
        self.refresh_data()

    def _offer_starter_preset(self, model_id: str, name: str) -> None:
        """No preset serves this checkpoint: write one, then offer the swap."""

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            def _work() -> tuple[Any, Any, Any]:
                starter = _data.write_starter_preset(model_id)
                served, state = _data.serve_route(model_id)
                return starter, served, state

            self.run_action_worker(_work, lambda out: self._after_starter(*out))

        self.app.push_screen(
            ConfirmActionModal(
                f"Write a preset for {name}?",
                "vLLM serves presets, and none names this checkpoint yet. llmctl "
                "writes a starter one from its config.json (context capped at "
                "settings.yaml's default, no tool parser), then shows the swap.",
                confirm_label="Write preset",
            ),
            _on_confirm,
        )

    def _after_starter(self, starter: Any, served: Any, state: Any) -> None:
        """Report the preset, then show the swap it enables (a modal, no worker)."""
        if starter is None:
            self.app.notify("Nothing changed: the model was not found.", severity="warning")
            return
        first = (
            f"Wrote starter preset {starter.alias}."
            if starter.created
            else f"Preset {starter.alias} already serves it."
        )
        self.app.notify(esc("\n".join([first, *starter.notes])), title="Presets", timeout=15)
        self.refresh_data()
        if served is not None and served.kind == "preset" and served.target is not None:
            self._confirm_swap(state, served.target)

    def _show_plan(self, model_id: str, plan: Any) -> None:
        """Push the launch-plan modal; plan or launch based on the choice."""

        def _on_close(outcome: str | None) -> None:
            if outcome not in ("plan", "launch"):
                return
            dry_run = outcome == "plan"
            self.run_action_worker(
                lambda: _data.start_model(model_id, dry_run=dry_run, force=dry_run),
                self._after_start,
            )

        self.app.push_screen(LaunchPlanModal(plan), _on_close)

    def _after_start(self, session: Any) -> None:
        """Notify with the real outcome (planned / started / failed) and refresh."""
        short_id = (session.id or "")[:8]
        status = session.status.value
        if status == "planned":
            self.app.notify(
                f"Planned session {short_id} ({session.runtime.value}); no process launched.",
                title="Session planned",
            )
        elif status in ("running", "starting"):
            detail = f" at {esc(session.endpoint_url)}" if session.endpoint_url else ""
            self.app.notify(
                f"Session {short_id} {status}{detail}.",
                title="Session launched",
            )
        else:
            self.app.notify(
                f"Session {short_id} {status}: {esc(session.error or 'unknown error')}",
                title="Launch failed",
                severity="error",
                timeout=10,
            )
        self.refresh_data()

    def action_scan(self) -> None:
        """Confirm, then run adapter discovery and persist the results.

        This is ``scan --import``, not the CLI's non-persisting preview: it
        upserts every discovered model and flags absent ones MISSING, which
        the prune action then removes irreversibly. Worth a gate on a chord
        that everywhere else means "save".
        """

        def _on_close(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.run_action_worker(_data.scan_models, self._after_scan)

        self.app.push_screen(
            ConfirmActionModal(
                "Scan runtimes and import?",
                "Discovered models are written to the registry. Models a "
                "reachable runtime no longer reports are flagged MISSING, "
                "which prune (x) then removes irreversibly.",
                confirm_label="Scan",
            ),
            _on_close,
        )

    def _after_scan(self, found: Any) -> None:
        """Notify and refresh after a scan completes.

        ``scan()`` returns the whole registry, not just this pass's finds, so
        report it as a total rather than implying a discovery count.
        """
        self.app.notify(f"Scan complete. Registry now holds {len(found)} models.")
        self.refresh_data()

    def action_prune_missing(self) -> None:
        """Confirm and soft-delete the models flagged MISSING right now.

        The ids are captured here, before the dialog opens, and the prune acts
        on exactly that set. Re-deriving it at confirm time let a scan landing
        in between widen the deletion past the number the operator agreed to.
        """
        ids = self.query_data(_data.get_missing_model_ids, what="Missing-model lookup")
        if ids is None:
            return
        if not ids:
            self.app.notify("No missing models to prune.", severity="information")
            return

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.prune_missing_models(ids=ids),
                self._after_prune,
            )

        self.app.push_screen(
            DeleteModal(
                f"{len(ids)} missing model(s)",
                "Each row is hidden from listings. llmctl has no undelete and "
                "a rescan will not revive them. Files on disk are untouched.",
            ),
            _on_close,
        )

    def _after_prune(self, count: Any) -> None:
        """Notify and refresh after a prune completes."""
        self.app.notify(f"Pruned {count} missing model(s).")
        self.refresh_data()

    def _selected_model(self) -> Any:
        """Return the full Model schema for the cursor row, if any.

        Runs from key handlers (``e``/``c``/``d``), so the read is guarded —
        an unguarded registry query here would kill the app on a locked DB.
        """
        model_id = self._selected_id()
        if not model_id:
            return None
        models = self.query_data(_data.get_models, what="Model lookup")
        if models is None:
            return None
        for model in models:
            if model.id == model_id:
                return model
        return None

    def action_add_model(self) -> None:
        """Open the add dialog: one source string, any engine."""

        def _on_close(request: AddRequest | None) -> None:
            if request is None:
                return
            if request.manual:
                self._open_manual_form()
            elif request.plan is not None:
                self._run_add(request.plan)

        self.app.push_screen(AddModelModal(), _on_close)

    def _run_add(self, plan: Any) -> None:
        """Fetch + register ``plan`` off-thread, reporting progress as toasts.

        Pulls and downloads run minutes to hours; the worker keeps the screen
        responsive and each progress line lands as a short toast.
        """
        app = self.app
        app.notify(esc(plan.summary), title="Adding model")

        def progress(message: str) -> None:
            self._on_ui(
                app, functools.partial(app.notify, esc(message), title=plan.name, timeout=4)
            )

        self.run_action_worker(
            lambda: _data.add_model_from_source(plan, on_progress=progress),
            self._after_add,
        )

    def _after_add(self, result: Any) -> None:
        """Say what was added and the one command that serves it."""
        lines = [f"Added {result.model.name} ({result.model.runtime.value})."]
        lines += list(result.notes)
        if result.serve:
            lines.append(f"Serve it: {result.serve} (or Swap screen, w).")
        self.app.notify(esc("\n".join(lines)), title="Models", timeout=15)
        self.refresh_data()

    def _open_manual_form(self) -> None:
        """The field-by-field form, for records llmctl cannot fetch."""

        def _on_close(payload: ModelCreate | ModelUpdate | None) -> None:
            if isinstance(payload, ModelCreate):
                self.run_action_worker(
                    lambda: _data.add_model(payload),
                    self._after(f"Added model {payload.name}."),
                )

        self.app.push_screen(ModelFormModal(), _on_close)

    def action_edit_model(self) -> None:
        """Open the edit-model form for the cursor row."""
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return

        def _on_close(payload: ModelCreate | ModelUpdate | None) -> None:
            if isinstance(payload, ModelUpdate):
                self.run_action_worker(
                    lambda: _data.update_model(model.id, payload),
                    self._after(f"Saved model {payload.name or model.name}."),
                )

        self.app.push_screen(ModelFormModal(model), _on_close)

    def action_clone_model(self) -> None:
        """Clone the cursor-row model under a new name."""
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return

        def _on_close(payload: CloneRequest | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.clone_model(payload.source_id, payload.new_name),
                self._after(f"Cloned {model.name} as {payload.new_name}."),
            )

        self.app.push_screen(CloneModal(model.id, model.name), _on_close)

    def action_delete_model(self) -> None:
        """Confirm and remove the cursor-row model, saying what each choice does.

        The checkbox is engine-specific: "remove it from Ollama" for an Ollama
        tag, "delete the files (N GB)" for a checkpoint. When llmctl cannot
        remove the model itself (outside the model roots, LM Studio did not
        report a folder, ...) the dialog says so and offers only the row.
        """
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return
        plan = self.query_data(
            lambda: _data.plan_model_removal(model.id), what="Removal check"
        )
        if plan is None:
            return
        consequence = (
            "Hides it from llmctl. Without the box ticked the model stays in "
            f"{model.runtime.value} and on disk; Add brings the row back."
        )
        if not plan.can_purge:
            consequence += f"\n\n{plan.purge_summary}"
        elif plan.blockers:
            consequence += "\n\nCan't remove the model itself right now: " + " ".join(
                plan.blockers
            )

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.remove_model(model.id, purge=payload.delete_files),
                self._after_remove,
            )

        self.app.push_screen(
            DeleteModal(
                f"model '{model.name}'",
                consequence,
                allow_file_delete=plan.can_purge and not plan.blockers,
                file_delete_label=plan.purge_label,
                verb="Remove",
            ),
            _on_close,
        )

    def _after_remove(self, result: Any) -> None:
        """Report the removal as the service stated it, then refresh."""
        if result is None or not result.removed_row:
            self.app.notify(
                "Nothing changed: the model was not found (deleted elsewhere?).",
                severity="warning",
            )
        else:
            self.app.notify(esc(result.message), title="Models")
        self.refresh_data()

    def _after(self, success: str) -> Any:
        """Callback for add/edit/clone/delete: say what happened, then refresh.

        Presets always confirmed a mutation; Models refreshed silently, so a
        save looked the same as nothing happening. A ``None``/``False``
        result is the service reporting the row was not there.
        """

        def _done(result: Any) -> None:
            if result is None or result is False:
                self.app.notify(
                    "Nothing changed: the model was not found (deleted elsewhere?).",
                    severity="warning",
                )
            else:
                self.app.notify(esc(success), title="Models")
            self.refresh_data()

        return _done
