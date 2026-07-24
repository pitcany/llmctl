"""Modal dialogs for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from llmctl.schemas import LaunchPlan
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, esc


class LaunchPlanModal(ModalScreen[str | None]):
    """Preview a launch plan, then plan it or actually launch it.

    Dismisses with ``"launch"`` (real start), ``"plan"`` (record a dry-run
    PLANNED session), or ``None`` (cancel). A refused plan only offers
    "plan" — forcing past a refusal is CLI-only (``llmctl start --force``).
    """

    BINDINGS = [("escape", "dismiss_cancel", "Cancel")]

    def __init__(self, plan: LaunchPlan) -> None:
        super().__init__()
        self._plan = plan

    def compose(self) -> ComposeResult:
        """Compose the plan preview dialog."""
        plan = self._plan
        gpus = ",".join(str(g) for g in plan.gpu_ids) or "cpu"
        est = "unknown" if plan.estimated_vram_gb is None else f"{plan.estimated_vram_gb:.1f} GB"
        free = "n/a" if plan.free_vram_gb is None else f"{plan.free_vram_gb:.1f} GB"
        lines = [
            "[b]Launch Plan[/b]",
            f"[{C_MUTED}]Model[/]      {esc(plan.model_name or plan.model_id)}",
            f"[{C_MUTED}]Backend[/]    {plan.runtime.value}",
            f"[{C_MUTED}]Profile[/]    {esc(plan.profile_name or '-')}",
            f"[{C_MUTED}]GPU mode[/]   {plan.gpu_selection_mode}  ->  {gpus}",
            f"[{C_MUTED}]Tensor par[/] {plan.tensor_parallel_size}",
            f"[{C_MUTED}]Port[/]       {plan.port or '-'}",
            f"[{C_MUTED}]VRAM[/]       est {est} / free {free}",
            f"[{C_MUTED}]Command[/]    {esc(plan.command_preview)}",
        ]
        for warning in plan.warnings:
            lines.append(f"[{C_WARN}]warning:[/] {esc(warning)}")
        for reason in plan.refusal_reasons:
            lines.append(f"[{C_ERR}]refusal:[/] {esc(reason)}")
        if plan.refusal_reasons:
            lines.append(
                f"\n[{C_WARN}]This launch is refused; only a planned session can be "
                "recorded here. Use `llmctl start --force` to override.[/]"
            )
        else:
            lines.append(f"\n[{C_OK}]This launch is allowed.[/]")

        with Vertical(id="plan-dialog", classes="panel"):
            yield Static("\n".join(lines), id="plan-body")
            with Vertical(id="plan-buttons"):
                if not plan.refusal_reasons:
                    yield Button("Launch now", variant="success", id="plan-launch")
                yield Button("Plan only (no process)", variant="primary", id="plan-confirm")
                yield Button("Cancel", variant="error", id="plan-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Resolve the modal based on the pressed button."""
        outcome = {"plan-launch": "launch", "plan-confirm": "plan"}.get(event.button.id or "")
        self.dismiss(outcome)

    def action_dismiss_cancel(self) -> None:
        """Dismiss the modal as cancelled."""
        self.dismiss(None)
