"""Textual modal forms for model and profile management.

Returns ``None`` on cancel, or a typed payload (ModelCreate/ModelUpdate/
ProfileCreate/ProfileUpdate/CloneRequest/ConfirmDelete) on submit. Screens
own the side effect of calling the service layer with the returned payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from llmctl.db import RuntimeName
from llmctl.schemas import (
    Model,
    ModelCreate,
    ModelUpdate,
    Profile,
    ProfileCreate,
    ProfileUpdate,
)
from llmctl.services.model_ops import ADDABLE_ENGINES, AddPlan, ModelOpsError, plan_add
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, esc

_RUNTIME_OPTIONS = [(r.value, r.value) for r in RuntimeName]


@dataclass(frozen=True)
class CloneRequest:
    """Result of the clone modal: original id + new name."""

    source_id: str
    new_name: str


@dataclass(frozen=True)
class ConfirmDelete:
    """Result of the delete modal: optionally request file deletion."""

    delete_files: bool


def _optional_int(text: str, label: str = "value") -> int | None:
    """Blank -> None; anything else must be a whole number or the form refuses.

    ``abc`` in "Max context" used to be saved as ``None`` without a word.
    """
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number (got {text!r})") from exc


def _optional_float(text: str, label: str = "value") -> float | None:
    """Blank -> None; anything else must be a number (see :func:`_optional_int`)."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number (got {text!r})") from exc


def _form_error_text(exc: Exception) -> str:
    """One line per problem; pydantic's bracketed detail and URL dropped."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", ())) or "form"
            msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
            parts.append(f"{field}: {msg}")
        return "\n".join(parts)
    return str(exc)


def _parse_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _show_form_error(screen: ModalScreen, selector: str, exc: Exception) -> None:
    """Inline red error under the form; the modal stays open, nothing saved."""
    text = _form_error_text(exc)
    screen.query_one(selector, Static).update(f"[{C_ERR}]{esc(text)}[/]")
    screen.app.notify(esc(text), severity="error", title="Not saved")


class ModelFormModal(ModalScreen[ModelCreate | ModelUpdate | None]):
    """Add or edit a model. When ``model`` is None this is an Add form."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, model: Model | None = None) -> None:
        super().__init__()
        self._model = model

    @property
    def _is_edit(self) -> bool:
        return self._model is not None

    def compose(self) -> ComposeResult:
        m = self._model
        title = "Edit model" if self._is_edit else "Add model"
        with Vertical(id="model-form-dialog", classes="panel"):
            yield Static(f"[b]{title}[/b]", id="model-form-title")
            yield Label("Name")
            yield Input(value=m.name if m else "", id="m-name")
            yield Label("Backend")
            yield Select(
                _RUNTIME_OPTIONS,
                value=m.runtime.value if m else RuntimeName.VLLM.value,
                id="m-runtime",
                allow_blank=False,
            )
            yield Label("Path")
            yield Input(value=(m.path if m and m.path else ""), id="m-path")
            yield Label("Quantization")
            yield Input(
                value=(m.quantization if m and m.quantization else ""), id="m-quant"
            )
            yield Label("Max context")
            yield Input(
                value=str(m.max_context) if m and m.max_context else "",
                id="m-maxctx",
            )
            yield Label("Estimated VRAM (GB)")
            yield Input(
                value=(
                    f"{m.estimated_vram_gb:.1f}"
                    if m and m.estimated_vram_gb is not None
                    else ""
                ),
                id="m-vram",
            )
            yield Label("Tags (comma-separated)")
            yield Input(value=", ".join(m.tags) if m else "", id="m-tags")
            yield Label("Notes")
            yield Input(value=(m.notes if m and m.notes else ""), id="m-notes")
            yield Static("", id="m-error")
            with Horizontal(id="model-form-buttons"):
                label = "Save" if self._is_edit else "Add"
                yield Button(label, variant="success", id="m-submit")
                yield Button("Cancel", variant="error", id="m-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "m-cancel":
            self.dismiss(None)
            return
        name = self.query_one("#m-name", Input).value.strip()
        runtime_value = self.query_one("#m-runtime", Select).value
        if not name or runtime_value is Select.BLANK:
            self.app.notify("Name and backend are required.", severity="error")
            return
        try:
            self.dismiss(self._build(name, runtime_value))
        except (ValueError, ValidationError) as exc:
            _show_form_error(self, "#m-error", exc)

    def _build(self, name: str, runtime_value: object) -> ModelCreate | ModelUpdate:
        """Read the form; raises ValueError/ValidationError on bad input."""
        path = self.query_one("#m-path", Input).value.strip() or None
        fields = {
            "name": name,
            "runtime": RuntimeName(str(runtime_value)),
            "path": path,
            "quantization": self.query_one("#m-quant", Input).value.strip() or None,
            "max_context": _optional_int(
                self.query_one("#m-maxctx", Input).value, "Max context"
            ),
            "estimated_vram_gb": _optional_float(
                self.query_one("#m-vram", Input).value, "Estimated VRAM"
            ),
            "tags": _parse_tags(self.query_one("#m-tags", Input).value),
            "notes": self.query_one("#m-notes", Input).value.strip() or None,
        }
        if self._is_edit:
            return ModelUpdate(**fields)
        return ModelCreate(**fields, source=path)

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass(frozen=True)
class AddRequest:
    """Result of the add modal: a checked plan, or a request for the manual form."""

    plan: AddPlan | None = None
    manual: bool = False


_ENGINE_LABELS = {
    RuntimeName.OLLAMA: "Ollama",
    RuntimeName.VLLM: "vLLM",
    RuntimeName.SGLANG: "SGLang",
    RuntimeName.FREETOKEN: "FreeToken",
    RuntimeName.STRATA: "Strata",
    RuntimeName.LLAMA_CPP: "llama.cpp",
    RuntimeName.LMSTUDIO: "LM Studio",
}
_ENGINE_OPTIONS = [("auto (from source)", "auto")] + [
    (_ENGINE_LABELS[e], e.value) for e in ADDABLE_ENGINES
]

_ADD_EXAMPLES = (
    "[b]qwen3:8b[/b]                        Ollama\n"
    "[b]Qwen/Qwen3-8B[/b]                   vLLM (or pick SGLang / FreeToken)\n"
    "[b]unsloth/Qwen3-8B-GGUF:Q4_K_M[/b]    llama.cpp, one quant\n"
    "[b]lmstudio:qwen/qwen3-8b[/b]          LM Studio\n"
    "[b]~/models/my-model[/b]               register a local folder or .gguf"
)


class AddModelModal(ModalScreen[AddRequest | None]):
    """Add a model from one SOURCE string; the engine follows from its shape.

    The preview line is :func:`plan_add` on every keystroke, so what the
    operator reads ("Pull qwen3:8b into the Ollama library, same as: ollama
    pull qwen3:8b") is exactly what Add runs. Add stays refused while the
    source does not plan. Nothing touches the network until the screen runs
    the plan.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._plan: AddPlan | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="add-dialog", classes="panel"):
            yield Static("[b]Add model[/b]", id="add-title")
            yield Static(
                "Paste a model name, repo, or path. The engine is picked from its shape.",
                classes="muted",
            )
            yield Label("Source")
            yield Input(placeholder="e.g. qwen3:8b or Qwen/Qwen3-8B", id="add-source")
            yield Label("Engine")
            yield Select(_ENGINE_OPTIONS, value="auto", id="add-engine", allow_blank=False)
            yield Label("Name (optional)")
            yield Input(placeholder="defaults to the source", id="add-name")
            yield Static("", id="add-preview")
            yield Static(_ADD_EXAMPLES, id="add-examples", classes="muted")
            with Horizontal(id="add-buttons"):
                yield Button("Add", variant="success", id="add-submit")
                yield Button("Manual entry", id="add-manual")
                yield Button("Cancel", variant="error", id="add-cancel")

    def _replan(self) -> None:
        source = self.query_one("#add-source", Input).value
        engine = self.query_one("#add-engine", Select).value
        name = self.query_one("#add-name", Input).value.strip() or None
        preview = self.query_one("#add-preview", Static)
        if not source.strip():
            self._plan = None
            preview.update("")
            return
        try:
            self._plan = plan_add(
                source, None if engine in ("auto", Select.BLANK) else str(engine), name=name
            )
        except ModelOpsError as exc:
            self._plan = None
            preview.update(f"[{C_ERR}]{esc(str(exc))}[/]")
            return
        label = _ENGINE_LABELS.get(self._plan.engine, self._plan.engine.value)
        preview.update(
            f"[{C_OK}]→ {esc(label)}:[/] {esc(self._plan.summary)}\n"
            f"[{C_MUTED}]same as: {esc(self._plan.command)}[/]"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        self._replan()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._replan()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-cancel":
            self.dismiss(None)
        elif event.button.id == "add-manual":
            self.dismiss(AddRequest(manual=True))
        else:
            self._submit()

    def _submit(self) -> None:
        self._replan()
        if self._plan is None:
            self.app.notify(
                "Enter a source that shows a green preview first.", severity="error"
            )
            return
        self.dismiss(AddRequest(plan=self._plan))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProfileFormModal(ModalScreen[ProfileCreate | ProfileUpdate | None]):
    """Add or edit a profile. When ``profile`` is None this is a Create form."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, profile: Profile | None = None) -> None:
        super().__init__()
        self._profile = profile

    @property
    def _is_edit(self) -> bool:
        return self._profile is not None

    def compose(self) -> ComposeResult:
        p = self._profile
        title = "Edit profile" if self._is_edit else "Create profile"
        with Vertical(id="profile-form-dialog", classes="panel"):
            yield Static(f"[b]{title}[/b]", id="profile-form-title")
            yield Label("Name")
            yield Input(value=p.name if p else "", id="p-name")
            yield Label("Backend")
            yield Select(
                _RUNTIME_OPTIONS,
                value=p.runtime.value if p else RuntimeName.VLLM.value,
                id="p-runtime",
                allow_blank=False,
            )
            yield Label("Description")
            yield Input(value=(p.description if p and p.description else ""), id="p-desc")
            yield Label("Tensor parallel size (GPUs per replica)")
            yield Input(
                value=str(p.tensor_parallel_size) if p and p.tensor_parallel_size else "",
                id="p-tp",
            )
            yield Label("Max model length (tokens)")
            yield Input(
                value=str(p.max_model_len) if p and p.max_model_len else "",
                id="p-maxlen",
            )
            yield Label("GPU memory fraction, 0 < x <= 1")
            yield Input(
                value=(
                    f"{p.gpu_memory_utilization:.2f}"
                    if p and p.gpu_memory_utilization is not None
                    else ""
                ),
                id="p-gpuutil",
            )
            yield Label("dtype (e.g. bfloat16; blank = auto)")
            yield Input(value=(p.dtype if p and p.dtype else ""), id="p-dtype")
            yield Label("Quantization (e.g. awq, fp8; blank = none)")
            yield Input(
                value=(p.quantization if p and p.quantization else ""),
                id="p-quant",
            )
            yield Static("", id="p-error")
            with Horizontal(id="profile-form-buttons"):
                label = "Save" if self._is_edit else "Create"
                yield Button(label, variant="success", id="p-submit")
                yield Button("Cancel", variant="error", id="p-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "p-cancel":
            self.dismiss(None)
            return
        name = self.query_one("#p-name", Input).value.strip()
        runtime_value = self.query_one("#p-runtime", Select).value
        if not name or runtime_value is Select.BLANK:
            self.app.notify("Name and backend are required.", severity="error")
            return
        try:
            self.dismiss(self._build(name, runtime_value))
        except (ValueError, ValidationError) as exc:
            _show_form_error(self, "#p-error", exc)

    def _build(self, name: str, runtime_value: object) -> ProfileCreate | ProfileUpdate:
        """Read the form; raises ValueError/ValidationError on bad input."""
        fields = {
            "name": name,
            "runtime": RuntimeName(str(runtime_value)),
            "description": self.query_one("#p-desc", Input).value.strip() or None,
            "tensor_parallel_size": _optional_int(
                self.query_one("#p-tp", Input).value, "Tensor parallel size"
            ),
            "max_model_len": _optional_int(
                self.query_one("#p-maxlen", Input).value, "Max model length"
            ),
            "gpu_memory_utilization": _optional_float(
                self.query_one("#p-gpuutil", Input).value, "GPU memory fraction"
            ),
            "dtype": self.query_one("#p-dtype", Input).value.strip() or None,
            "quantization": self.query_one("#p-quant", Input).value.strip() or None,
        }
        if self._is_edit:
            return ProfileUpdate(**fields)
        return ProfileCreate(**fields)

    def action_cancel(self) -> None:
        self.dismiss(None)


class CloneModal(ModalScreen[CloneRequest | None]):
    """Prompt the user for a clone target name."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, source_id: str, current_name: str) -> None:
        super().__init__()
        self._source_id = source_id
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        with Vertical(id="clone-dialog", classes="panel"):
            yield Static(f"[b]Clone {esc(self._current_name)}[/b]", id="clone-title")
            yield Label("New name")
            yield Input(value=f"{self._current_name}-copy", id="clone-name")
            with Horizontal(id="clone-buttons"):
                yield Button("Clone", variant="success", id="clone-confirm")
                yield Button("Cancel", variant="error", id="clone-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clone-cancel":
            self.dismiss(None)
            return
        new_name = self.query_one("#clone-name", Input).value.strip()
        if not new_name:
            self.app.notify("New name is required.", severity="error")
            return
        self.dismiss(CloneRequest(source_id=self._source_id, new_name=new_name))

    def action_cancel(self) -> None:
        self.dismiss(None)


class DeleteModal(ModalScreen[ConfirmDelete | None]):
    """Confirm a deletion, stating what the deletion actually does.

    ``consequence`` is supplied by the caller and is the only description the
    operator sees. It is not defaulted: the dialog used to hard-code a
    "soft-delete only, files preserved" explainer that was false on the
    Presets screen (which unlinks YAML) and on Benchmarks (which hard-deletes
    the row), and that referred to a checkbox those call sites never render.

    ``AUTO_FOCUS`` targets Cancel. Textual otherwise focuses the first control
    in DOM order — the file-deletion checkbox when it exists — so the same
    ``d``, ``Enter`` reflex that completes a delete elsewhere silently armed
    recursive on-disk deletion here.
    """

    AUTO_FOCUS = "#delete-cancel"
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        label: str,
        consequence: str,
        *,
        allow_file_delete: bool = False,
        file_delete_target: str | None = None,
        file_delete_label: str | None = None,
        verb: str = "Delete",
    ) -> None:
        super().__init__()
        self._label = label
        self._consequence = consequence
        self._allow_file_delete = allow_file_delete
        self._file_delete_target = file_delete_target
        # Engine-specific wording ("Also remove it from Ollama (ollama rm
        # ...)"); the generic file wording is wrong for a daemon-held model.
        self._file_delete_label = file_delete_label
        # The modal is reused for any confirmed action, not just deletion.
        # Asking "Delete unit 'gpt-oss-120b'?" before a *stop* reads as though
        # the unit file is about to be removed, which is far more alarming
        # than what actually happens.
        self._verb = verb

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog", classes="panel"):
            yield Static(
                f"[b]{esc(self._verb)} {esc(self._label)}?[/b]", id="delete-title"
            )
            yield Static(esc(self._consequence), id="delete-explainer")
            if self._allow_file_delete:
                target = self._file_delete_target or "(no path recorded)"
                yield Checkbox(
                    self._file_delete_label
                    or f"Also delete files on disk, irreversibly: {target}",
                    value=False,
                    id="delete-files-cb",
                )
            with Horizontal(id="delete-buttons"):
                yield Button(self._verb, variant="error", id="delete-confirm")
                yield Button("Cancel", variant="primary", id="delete-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete-cancel":
            self.dismiss(None)
            return
        delete_files = False
        if self._allow_file_delete:
            delete_files = bool(self.query_one("#delete-files-cb", Checkbox).value)
        self.dismiss(ConfirmDelete(delete_files=delete_files))

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = [
    "AddModelModal",
    "AddRequest",
    "CloneModal",
    "CloneRequest",
    "ConfirmDelete",
    "DeleteModal",
    "ModelFormModal",
    "ProfileFormModal",
    "_optional_float",
    "_optional_int",
    "_parse_tags",
]
