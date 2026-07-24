"""Textual modal forms for model and profile management.

Returns ``None`` on cancel, or a typed payload (ModelCreate/ModelUpdate/
ProfileCreate/ProfileUpdate/CloneRequest/ConfirmDelete) on submit. Screens
own the side effect of calling the service layer with the returned payload.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from llmctl.tui._base import esc

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


def _optional_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _optional_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


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
        path = self.query_one("#m-path", Input).value.strip() or None
        quant = self.query_one("#m-quant", Input).value.strip() or None
        max_ctx = _optional_int(self.query_one("#m-maxctx", Input).value)
        vram = _optional_float(self.query_one("#m-vram", Input).value)
        tags = _parse_tags(self.query_one("#m-tags", Input).value)
        notes = self.query_one("#m-notes", Input).value.strip() or None
        runtime = RuntimeName(str(runtime_value))
        if self._is_edit:
            self.dismiss(
                ModelUpdate(
                    name=name,
                    runtime=runtime,
                    path=path,
                    quantization=quant,
                    max_context=max_ctx,
                    estimated_vram_gb=vram,
                    tags=tags,
                    notes=notes,
                )
            )
        else:
            self.dismiss(
                ModelCreate(
                    name=name,
                    runtime=runtime,
                    path=path,
                    source=path,
                    quantization=quant,
                    max_context=max_ctx,
                    estimated_vram_gb=vram,
                    tags=tags,
                    notes=notes,
                )
            )

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
            yield Label("tensor_parallel_size")
            yield Input(
                value=str(p.tensor_parallel_size) if p and p.tensor_parallel_size else "",
                id="p-tp",
            )
            yield Label("max_model_len")
            yield Input(
                value=str(p.max_model_len) if p and p.max_model_len else "",
                id="p-maxlen",
            )
            yield Label("gpu_memory_utilization (0..1]")
            yield Input(
                value=(
                    f"{p.gpu_memory_utilization:.2f}"
                    if p and p.gpu_memory_utilization is not None
                    else ""
                ),
                id="p-gpuutil",
            )
            yield Label("dtype")
            yield Input(value=(p.dtype if p and p.dtype else ""), id="p-dtype")
            yield Label("quantization")
            yield Input(
                value=(p.quantization if p and p.quantization else ""),
                id="p-quant",
            )
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
        description = self.query_one("#p-desc", Input).value.strip() or None
        tp = _optional_int(self.query_one("#p-tp", Input).value)
        max_len = _optional_int(self.query_one("#p-maxlen", Input).value)
        gpu_util = _optional_float(self.query_one("#p-gpuutil", Input).value)
        dtype = self.query_one("#p-dtype", Input).value.strip() or None
        quant = self.query_one("#p-quant", Input).value.strip() or None
        runtime = RuntimeName(str(runtime_value))
        if self._is_edit:
            self.dismiss(
                ProfileUpdate(
                    name=name,
                    runtime=runtime,
                    description=description,
                    tensor_parallel_size=tp,
                    max_model_len=max_len,
                    gpu_memory_utilization=gpu_util,
                    dtype=dtype,
                    quantization=quant,
                )
            )
        else:
            self.dismiss(
                ProfileCreate(
                    name=name,
                    runtime=runtime,
                    description=description,
                    tensor_parallel_size=tp,
                    max_model_len=max_len,
                    gpu_memory_utilization=gpu_util,
                    dtype=dtype,
                    quantization=quant,
                )
            )

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
    ) -> None:
        super().__init__()
        self._label = label
        self._consequence = consequence
        self._allow_file_delete = allow_file_delete
        self._file_delete_target = file_delete_target

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog", classes="panel"):
            yield Static(f"[b]Delete {esc(self._label)}?[/b]", id="delete-title")
            yield Static(esc(self._consequence), id="delete-explainer")
            if self._allow_file_delete:
                target = self._file_delete_target or "(no path recorded)"
                yield Checkbox(
                    f"Also delete files on disk, irreversibly: {target}",
                    value=False,
                    id="delete-files-cb",
                )
            with Horizontal(id="delete-buttons"):
                yield Button("Delete", variant="error", id="delete-confirm")
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
