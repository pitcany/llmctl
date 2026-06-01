"""Launch modal for the Benchmarks screen.

Lets the operator pick a target model, benchmark kind, and a couple of
parameters, then resolves to a :class:`BenchmarkLaunch` value object that
the screen dispatches off-thread.

Editing the prompt itself is intentionally out of scope; for that, drop
to the CLI (``llmctl bench MODEL_ID --prompt "..."``) where the spec's
default prompt and arbitrary text both work without quoting hell inside
a TUI input.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from llmctl.db import BenchmarkKind
from llmctl.schemas import Model
from llmctl.tui._base import C_ERR, C_MUTED


@dataclass(frozen=True)
class BenchmarkLaunch:
    """The validated launch payload returned by :class:`BenchmarkLaunchModal`."""

    model_id: str
    kind: BenchmarkKind
    name: str
    max_tokens: int
    context_length: int | None
    dry_run: bool
    #: True when the operator explicitly picked "live" (vs. left blank).
    #: The screen forwards this to the service as ``require_live`` so a
    #: missing endpoint records a failure rather than a silent mock.
    require_live: bool


_KIND_OPTIONS: list[tuple[str, str]] = [
    ("chat (streaming chat completions)", BenchmarkKind.CHAT.value),
    ("completion (/v1/completions)", BenchmarkKind.COMPLETION.value),
    ("health (GET /v1/models)", BenchmarkKind.HEALTH.value),
    ("long_context (large prompt smoke)", BenchmarkKind.LONG_CONTEXT.value),
]


class BenchmarkLaunchModal(ModalScreen[BenchmarkLaunch | None]):
    """Pick model + kind + a few knobs and dispatch a benchmark."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        models: list[Model],
        *,
        preselect_model_id: str | None = None,
    ) -> None:
        super().__init__()
        self._models = [m for m in models if m.id]
        self._preselect = preselect_model_id

    def compose(self) -> ComposeResult:
        """Compose the launch dialog with sensible defaults for the form."""
        with Vertical(id="bench-launch-dialog", classes="panel"):
            yield Static("[b]Run benchmark[/b]", id="bench-launch-title")

            if not self._models:
                yield Static(
                    f"[{C_ERR}]No models registered.[/]\n"
                    f"[{C_MUTED}]Run [b]llmctl scan[/b] or add one from the "
                    f"Models screen first, then come back.[/]",
                    id="bench-launch-empty",
                )
                with Horizontal(id="bench-launch-buttons"):
                    yield Button("Close", variant="error", id="bench-cancel")
                return

            model_options: list[tuple[str, str]] = [
                (
                    f"{m.name}  ({m.runtime.value}"
                    + (f" · {m.source}" if m.source else "")
                    + ")",
                    m.id or "",
                )
                for m in self._models
            ]
            default_model = self._preselect or model_options[0][1]

            yield Label("Model")
            yield Select(
                model_options,
                value=default_model,
                id="bench-model",
                allow_blank=False,
            )

            yield Label("Kind")
            yield Select(
                _KIND_OPTIONS,
                value=BenchmarkKind.CHAT.value,
                id="bench-kind",
                allow_blank=False,
            )

            yield Label("Name (free-form label saved on the record)")
            yield Input(value="smoke", id="bench-name")

            yield Label("max_tokens (per prompt; ignored for kind=health)")
            yield Input(value="256", id="bench-max-tokens")

            yield Label("context_length (long_context only; blank = ~8K)")
            yield Input(value="", id="bench-context-length")

            yield Label(
                "Mode: blank = live (best-effort, mocks if no endpoint); "
                "'live' = strict (fails if no endpoint); "
                "'mock' = synthetic dry-run"
            )
            yield Input(value="", id="bench-mode")

            yield Static("", id="bench-launch-error")

            with Horizontal(id="bench-launch-buttons"):
                yield Button("Run", variant="success", id="bench-submit")
                yield Button("Cancel", variant="error", id="bench-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Resolve the modal based on the pressed button."""
        if event.button.id == "bench-cancel":
            self.dismiss(None)
            return
        if event.button.id != "bench-submit":
            return
        try:
            launch = self._build_launch()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.dismiss(launch)

    def action_cancel(self) -> None:
        """Dismiss with ``None`` (Esc binding)."""
        self.dismiss(None)

    def _build_launch(self) -> BenchmarkLaunch:
        """Validate the form fields and assemble a :class:`BenchmarkLaunch`.

        Raises :class:`ValueError` with a user-readable message if any field
        is malformed; the message is rendered in the in-modal error slot.
        """
        model_select = self.query_one("#bench-model", Select)
        kind_select = self.query_one("#bench-kind", Select)
        name_input = self.query_one("#bench-name", Input)
        max_tokens_input = self.query_one("#bench-max-tokens", Input)
        ctx_input = self.query_one("#bench-context-length", Input)
        mode_input = self.query_one("#bench-mode", Input)

        model_id = str(model_select.value or "").strip()
        if not model_id:
            raise ValueError("Pick a model.")
        try:
            kind = BenchmarkKind(str(kind_select.value))
        except ValueError as exc:
            raise ValueError(f"Unknown kind: {kind_select.value}") from exc
        name = (name_input.value or "smoke").strip() or "smoke"
        try:
            max_tokens = int((max_tokens_input.value or "256").strip())
            if max_tokens <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("max_tokens must be a positive integer.") from exc
        ctx_text = (ctx_input.value or "").strip()
        context_length: int | None
        if not ctx_text:
            context_length = None
        else:
            try:
                context_length = int(ctx_text)
                if context_length <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    "context_length must be a positive integer or blank."
                ) from exc
        mode = (mode_input.value or "").strip().lower()
        if mode not in {"", "live", "mock"}:
            raise ValueError("Mode must be blank, 'live', or 'mock'.")
        dry_run = mode == "mock"
        # Blank defaults to live but tolerates an endpoint miss; an explicit
        # "live" is a contract that we should fail loudly if no endpoint
        # resolves, instead of silently mocking.
        require_live = mode == "live"
        return BenchmarkLaunch(
            model_id=model_id,
            kind=kind,
            name=name,
            max_tokens=max_tokens,
            context_length=context_length,
            dry_run=dry_run,
            require_live=require_live,
        )

    def _show_error(self, message: str) -> None:
        """Render a validation error in the dialog's error slot."""
        slot = self.query_one("#bench-launch-error", Static)
        slot.update(f"[{C_ERR}]{message}[/]")
