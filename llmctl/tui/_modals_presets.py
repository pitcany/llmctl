"""Modal: confirm launching a preset onto the TP fleet unit.

Kept in its own module from :mod:`llmctl.tui._modals` because the latter
is tied to :class:`~llmctl.schemas.LaunchPlan` (the original scheduler
abstraction), whereas the preset launcher works at the
``llmctl.services.vllm_orchestrator`` level. Separating them avoids
muddying either flow.
"""

from __future__ import annotations

from enum import StrEnum

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from llmctl.presets import CANONICAL_QUANTIZATIONS, Model, PresetSchemaError
from llmctl.services.preset_loader import PresetView
from llmctl.tui._base import C_ACCENT, C_ERR, C_MUTED, C_WARN, esc


class PresetLaunchTarget(StrEnum):
    """Which unit to apply the preset to."""

    TP = "tp"


class PresetLaunchModal(ModalScreen[PresetLaunchTarget | None]):
    """Launch confirmation for the TP fleet unit, with its blast radius.

    Resolves to :attr:`PresetLaunchTarget.TP`, or ``None`` if the user
    dismisses with Escape.

    This is the widest-reaching action in the TUI: confirming it takes down
    every inference backend on the machine before bringing one back. The
    dialog therefore enumerates what the orchestrator will do rather than
    describing itself as a picker — and ``AUTO_FOCUS`` targets Cancel, so a
    reflexive Enter never restarts the shared unit.
    """

    AUTO_FOCUS = "#pick-cancel"
    BINDINGS = [
        ("escape", "dismiss_cancel", "Cancel"),
        ("t", "pick_tp", "TP fleet"),
    ]

    def __init__(
        self,
        view: PresetView,
        *,
        currently_served: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._view = view
        #: What the unit is serving right now, so the operator knows what
        #: they are terminating. ``None`` when the probe was not attempted
        #: or failed; ``[]`` when the unit answered with nothing loaded.
        self._currently_served = currently_served

    def _replacing_line(self) -> str:
        """Describe what the restart will terminate."""
        if self._currently_served is None:
            return f"[{C_MUTED}]Replaces[/] (could not reach the unit to check)"
        if not self._currently_served:
            return f"[{C_MUTED}]Replaces[/] nothing — the unit is not serving now"
        return (
            f"[{C_MUTED}]Replaces[/] [{C_WARN}]"
            f"{esc(', '.join(self._currently_served))}[/] (terminated)"
        )

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog with the full effect list."""
        v = self._view
        size = f"{v.param_count_b:.0f}B" if v.param_count_b else "?"
        lines = [
            f"[b]Launch preset[/b] [{C_ACCENT}]{esc(v.alias)}[/]",
            "",
            f"[{C_MUTED}]Model[/]    {esc(v.model_id)}",
            f"[{C_MUTED}]Served[/]   {esc(v.served_name)}",
            f"[{C_MUTED}]Family[/]   {esc(v.family or '-')} ({size})",
            f"[{C_MUTED}]TP[/]       {v.tensor_parallel}  (both GPUs)",
            f"[{C_MUTED}]Quant[/]    {esc(v.quantization)}",
            "",
            self._replacing_line(),
            "",
            f"[{C_WARN}]This affects the whole machine:[/]",
            "  1. stops the ollama service",
            "  2. stops the Harbor ollama container",
            "  3. rewrites vllm-tp.env and restarts vllm-tp",
            "  4. waits up to 5 min for the model to load",
            "  5. repoints the Hermes 'vllm' provider",
            "",
            f"[{C_MUTED}]In-flight requests fail. If step 3 or 4 fails, ollama "
            f"and Harbor stay stopped.[/]",
        ]
        with Vertical(id="plan-dialog", classes="panel"):
            yield Static("\n".join(lines))
            with Vertical(id="plan-buttons"):
                yield Button(
                    "Launch on vllm-tp — t",
                    variant="error",
                    id="pick-tp",
                )
                yield Button("Cancel — esc", variant="primary", id="pick-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Map button id to a target, dismissing the modal with the result."""
        mapping: dict[str, PresetLaunchTarget | None] = {
            "pick-tp": PresetLaunchTarget.TP,
            "pick-cancel": None,
        }
        self.dismiss(mapping.get(event.button.id))

    def action_pick_tp(self) -> None:
        """Keyboard shortcut for TP."""
        self.dismiss(PresetLaunchTarget.TP)

    def action_dismiss_cancel(self) -> None:
        """Escape -> dismiss with None."""
        self.dismiss(None)


def _opt_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _opt_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _opt_str(text: str) -> str | None:
    text = text.strip()
    return text or None


def _opt_bool(value: object) -> bool | None:
    """Tri-state select: '' -> None, 'true'/'false' -> bool."""
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


_QUANT_OPTIONS = [(q, q) for q in sorted(CANONICAL_QUANTIZATIONS)]
_BOOL_OPTIONS = [("(unset)", ""), ("true", "true"), ("false", "false")]


class PresetFormModal(ModalScreen[Model | None]):
    """Add a new preset.

    Dismisses with ``None`` on cancel or a validated
    :class:`llmctl.presets.Model` on submit. Used for ``a`` only —
    editing a preset shells out to ``$EDITOR`` against the file
    directly, since the schema's 19 fields plus inline YAML comments
    are easier to navigate in a real editor than in a form.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="preset-form-dialog", classes="panel"):
            yield Static("[b]Add preset[/b]", id="preset-form-title")

            yield Label("Alias (filename key)")
            yield Input(value="", id="preset-alias")

            yield Label("Served name (OpenAI 'model' identifier)")
            yield Input(value="", id="preset-served")

            yield Label("Model id (HF repo or local path)")
            yield Input(value="", id="preset-model-id")

            yield Label("Quantization")
            yield Select(
                _QUANT_OPTIONS, value="awq", id="preset-quant", allow_blank=False
            )

            yield Label("vLLM --quantization flag (e.g. awq_marlin, fp8)")
            yield Input(value="", id="preset-vquant")

            yield Label("tensor_parallel_size (1..8)")
            yield Input(value="2", id="preset-tp")

            yield Label("max_model_len")
            yield Input(value="32768", id="preset-maxlen")

            yield Label("family (optional)")
            yield Input(value="", id="preset-family")

            yield Label("param_count_b (optional)")
            yield Input(value="", id="preset-params")

            yield Label("max_num_seqs (optional)")
            yield Input(value="", id="preset-maxseqs")

            yield Label("gpu_memory_utilization (0, 1])")
            yield Input(value="", id="preset-gpuutil")

            yield Label("kv_cache_dtype (e.g. fp8, fp16; blank = auto)")
            yield Input(value="", id="preset-kvdtype")

            yield Label("dtype (optional)")
            yield Input(value="", id="preset-dtype")

            yield Label("trust_remote_code")
            yield Select(
                _BOOL_OPTIONS, value="", id="preset-trc", allow_blank=False
            )

            yield Label("host (optional, blank = default)")
            yield Input(value="", id="preset-host")

            yield Label("port (optional, blank = default)")
            yield Input(value="", id="preset-port")

            yield Label("tool_parser (optional)")
            yield Input(value="", id="preset-tool")

            yield Label("reasoning_parser (optional)")
            yield Input(value="", id="preset-reasoning")

            yield Label("turboquant (tri-state)")
            yield Select(
                _BOOL_OPTIONS, value="", id="preset-tq", allow_blank=False
            )

            yield Static("", id="preset-form-error")

            with Horizontal(id="preset-form-buttons"):
                yield Button("Add", variant="success", id="preset-submit")
                yield Button("Cancel", variant="error", id="preset-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "preset-cancel":
            self.dismiss(None)
            return
        if event.button.id != "preset-submit":
            return
        try:
            model = self._build_model()
        except PresetSchemaError as exc:
            self._show_error(str(exc))
            return
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.dismiss(model)

    def _build_model(self) -> Model:
        alias = self.query_one("#preset-alias", Input).value.strip()
        served = self.query_one("#preset-served", Input).value.strip()
        model_id = self.query_one("#preset-model-id", Input).value.strip()
        if not alias or not served or not model_id:
            raise ValueError("alias, served_name and model_id are required")
        tp_text = self.query_one("#preset-tp", Input).value.strip()
        maxlen_text = self.query_one("#preset-maxlen", Input).value.strip()
        if not tp_text or not maxlen_text:
            raise ValueError("tensor_parallel_size and max_model_len are required")
        try:
            tp = int(tp_text)
            max_len = int(maxlen_text)
        except ValueError as exc:
            raise ValueError(
                "tensor_parallel_size and max_model_len must be integers"
            ) from exc

        quant_value = self.query_one("#preset-quant", Select).value
        vquant = self.query_one("#preset-vquant", Input).value.strip()
        if not vquant:
            raise ValueError("vllm_quantization_flag is required")

        return Model(
            alias=alias,
            served_name=served,
            model_id=model_id,
            quantization=str(quant_value),
            vllm_quantization_flag=vquant,
            tensor_parallel_size=tp,
            max_model_len=max_len,
            family=_opt_str(self.query_one("#preset-family", Input).value),
            param_count_b=_opt_float(self.query_one("#preset-params", Input).value),
            max_num_seqs=_opt_int(self.query_one("#preset-maxseqs", Input).value),
            gpu_memory_utilization=_opt_float(
                self.query_one("#preset-gpuutil", Input).value
            ),
            kv_cache_dtype=_opt_str(self.query_one("#preset-kvdtype", Input).value),
            dtype=_opt_str(self.query_one("#preset-dtype", Input).value),
            trust_remote_code=_opt_bool(self.query_one("#preset-trc", Select).value),
            host=_opt_str(self.query_one("#preset-host", Input).value),
            port=_opt_int(self.query_one("#preset-port", Input).value),
            tool_parser=_opt_str(self.query_one("#preset-tool", Input).value),
            reasoning_parser=_opt_str(
                self.query_one("#preset-reasoning", Input).value
            ),
            tq=_opt_bool(self.query_one("#preset-tq", Select).value),
        )

    def _show_error(self, message: str) -> None:
        widget = self.query_one("#preset-form-error", Static)
        widget.update(f"[{C_ERR}]{message}[/]")
        self.app.notify(message, severity="error", title="Preset invalid")

    def action_cancel(self) -> None:
        self.dismiss(None)
