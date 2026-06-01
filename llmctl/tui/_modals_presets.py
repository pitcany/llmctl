"""Modal: pick where to launch a preset (TP fleet / coder slot / reasoner slot).

Kept in its own module from :mod:`llmctl.tui._modals` because the latter
is tied to :class:`~llmctl.schemas.LaunchPlan` (the original scheduler
abstraction), whereas the preset launcher works at the
``llmctl.services.vllm_orchestrator`` level. Separating them avoids
muddying either flow.
"""

from __future__ import annotations

from enum import StrEnum

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from llmctl.services.preset_loader import PresetView
from llmctl.tui._base import C_ACCENT, C_MUTED


class PresetLaunchTarget(StrEnum):
    """Which unit to apply the preset to."""

    TP = "tp"
    CODER = "coder"
    REASONER = "reasoner"


class PresetLaunchModal(ModalScreen[PresetLaunchTarget | None]):
    """Three-button picker: TP fleet / coder slot / reasoner slot.

    Resolves to the chosen :class:`PresetLaunchTarget`, or ``None`` if
    the user dismisses with Escape.
    """

    BINDINGS = [
        ("escape", "dismiss_cancel", "Cancel"),
        ("t", "pick_tp", "TP fleet"),
        ("c", "pick_coder", "Coder slot"),
        ("r", "pick_reasoner", "Reasoner slot"),
    ]

    def __init__(self, view: PresetView) -> None:
        super().__init__()
        self._view = view

    def compose(self) -> ComposeResult:
        """Compose the picker dialog."""
        v = self._view
        size = f"{v.param_count_b:.0f}B" if v.param_count_b else "?"
        lines = [
            f"[b]Launch preset[/b] [{C_ACCENT}]{v.alias}[/]",
            "",
            f"[{C_MUTED}]Model[/]    {v.model_id}",
            f"[{C_MUTED}]Served[/]   {v.served_name}",
            f"[{C_MUTED}]Family[/]   {v.family or '-'} ({size})",
            f"[{C_MUTED}]TP[/]       {v.tensor_parallel}",
            f"[{C_MUTED}]Quant[/]    {v.quantization}",
            "",
            "Pick a target. Slot launches keep served_name=<slot> so",
            "downstream client configs don't change.",
        ]
        with Vertical(id="plan-dialog", classes="panel"):
            yield Static("\n".join(lines))
            with Vertical(id="plan-buttons"):
                yield Button(
                    "TP fleet (vllm-tp, both GPUs) — t",
                    variant="primary",
                    id="pick-tp",
                )
                yield Button(
                    "Coder slot (GPU 0, served as 'coder') — c",
                    variant="success",
                    id="pick-coder",
                )
                yield Button(
                    "Reasoner slot (GPU 1, served as 'reasoner') — r",
                    variant="success",
                    id="pick-reasoner",
                )
                yield Button("Cancel — esc", variant="error", id="pick-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Map button id to a target, dismissing the modal with the result."""
        mapping: dict[str, PresetLaunchTarget | None] = {
            "pick-tp": PresetLaunchTarget.TP,
            "pick-coder": PresetLaunchTarget.CODER,
            "pick-reasoner": PresetLaunchTarget.REASONER,
            "pick-cancel": None,
        }
        self.dismiss(mapping.get(event.button.id))

    def action_pick_tp(self) -> None:
        """Keyboard shortcut for TP."""
        self.dismiss(PresetLaunchTarget.TP)

    def action_pick_coder(self) -> None:
        """Keyboard shortcut for coder slot."""
        self.dismiss(PresetLaunchTarget.CODER)

    def action_pick_reasoner(self) -> None:
        """Keyboard shortcut for reasoner slot."""
        self.dismiss(PresetLaunchTarget.REASONER)

    def action_dismiss_cancel(self) -> None:
        """Escape -> dismiss with None."""
        self.dismiss(None)
