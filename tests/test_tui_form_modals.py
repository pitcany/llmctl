"""The add/edit/clone form modals — the least-covered code in the package.

These forms are the write path into the registry: every field an operator can
set flows through ``on_button_pressed`` here. Nothing exercised their submit or
validation branches, so the silent behaviours the audit flagged (Add couples
``source`` to ``path``; a non-numeric max-context is dropped to ``None`` rather
than rejected) had no protection at all.
"""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Button, Input, Select

from llmctl.db import RuntimeName
from llmctl.schemas import (
    Model,
    ModelCreate,
    ModelUpdate,
    Profile,
    ProfileCreate,
    ProfileUpdate,
)
from llmctl.tui._modals_registry import (
    CloneModal,
    CloneRequest,
    ModelFormModal,
    ProfileFormModal,
    _optional_float,
    _optional_int,
    _parse_tags,
)

# --------------------------------------------------------------------------- #
# Pure parsers — the silent-coercion behaviour, pinned directly.
# --------------------------------------------------------------------------- #


def test_optional_int_drops_garbage_to_none() -> None:
    assert _optional_int("42") == 42
    assert _optional_int("   ") is None
    # The audit's concern: "abc" in a number field is swallowed, not rejected.
    assert _optional_int("abc") is None
    assert _optional_int("3.5") is None  # not an int


def test_optional_float_drops_garbage_to_none() -> None:
    assert _optional_float("0.85") == 0.85
    assert _optional_float("") is None
    assert _optional_float("high") is None


def test_parse_tags_trims_and_drops_empties() -> None:
    assert _parse_tags("a, ,b") == ["a", "b"]
    assert _parse_tags("  one ,two,  ") == ["one", "two"]
    assert _parse_tags("") == []


# --------------------------------------------------------------------------- #
# Modal drivers.
# --------------------------------------------------------------------------- #


class _Host(App[None]):
    """Bare host: push one modal and capture its dismissal result."""

    def __init__(self, modal, sink: list) -> None:
        super().__init__()
        self._modal = modal
        self._sink = sink

    def on_mount(self) -> None:
        self.push_screen(self._modal, self._sink.append)


def _drive(modal, steps):
    """Push ``modal``, run ``steps(app, pilot)``, return the dismissal result."""
    sink: list = []

    async def _main() -> None:
        app = _Host(modal, sink)
        async with app.run_test() as pilot:
            await pilot.pause()
            await steps(app, pilot)

    asyncio.run(_main())
    return sink


def _set(app, wid: str, value: str) -> None:
    app.screen.query_one(f"#{wid}", Input).value = value


# --------------------------------------------------------------------------- #
# ModelFormModal.
# --------------------------------------------------------------------------- #


def test_model_add_returns_create_and_couples_source_to_path() -> None:
    """Add mode returns ModelCreate; source is silently set to path.

    That coupling is a real behaviour (registry keys off ``source``), and
    pinning it means a future change to it is a conscious one, not a surprise.
    """

    async def steps(app, pilot) -> None:
        _set(app, "m-name", "my-model")
        _set(app, "m-path", "/models/my-model")
        _set(app, "m-quant", "fp8")
        _set(app, "m-maxctx", "4096")
        _set(app, "m-vram", "40.5")
        _set(app, "m-tags", "local, fast")
        app.screen.query_one("#m-submit", Button).press()
        await pilot.pause()

    (result,) = _drive(ModelFormModal(), steps)
    assert isinstance(result, ModelCreate)
    assert result.name == "my-model"
    assert result.path == "/models/my-model"
    assert result.source == "/models/my-model"  # the coupling
    assert result.runtime is RuntimeName.VLLM
    assert result.quantization == "fp8"
    assert result.max_context == 4096
    assert result.estimated_vram_gb == 40.5
    assert result.tags == ["local", "fast"]


def test_model_edit_returns_update_without_touching_source() -> None:
    """Edit mode returns ModelUpdate and does NOT re-couple source to path."""
    existing = Model(
        id="mid",
        name="old",
        runtime=RuntimeName.VLLM,
        source="original-source",
        path="/old/path",
    )

    async def steps(app, pilot) -> None:
        _set(app, "m-name", "renamed")
        _set(app, "m-path", "/new/path")
        app.screen.query_one("#m-submit", Button).press()
        await pilot.pause()

    (result,) = _drive(ModelFormModal(existing), steps)
    assert isinstance(result, ModelUpdate)
    assert result.name == "renamed"
    assert result.path == "/new/path"
    # Edit leaves source at its default — it is not derived from the new path.
    assert result.source != "/new/path"


def test_model_form_requires_a_name() -> None:
    """Empty name must not dismiss; the form stays open with an error."""
    notes: list[str] = []

    async def steps(app, pilot) -> None:
        app.notify = lambda msg, *a, **k: notes.append(str(msg))  # type: ignore[method-assign]
        _set(app, "m-name", "   ")
        _set(app, "m-path", "/x")
        app.screen.query_one("#m-submit", Button).press()
        await pilot.pause()
        # Still on the modal.
        assert isinstance(app.screen, ModelFormModal)

    result = _drive(ModelFormModal(), steps)
    assert result == []  # never dismissed
    assert any("required" in n.lower() for n in notes)


def test_model_form_non_numeric_context_is_dropped_not_rejected() -> None:
    """A typo in max-context yields None, not a validation error.

    This is a footgun worth having on record: the operator gets a model with
    no context limit instead of a "that's not a number" message.
    """

    async def steps(app, pilot) -> None:
        _set(app, "m-name", "m")
        _set(app, "m-maxctx", "lots")
        app.screen.query_one("#m-submit", Button).press()
        await pilot.pause()

    (result,) = _drive(ModelFormModal(), steps)
    assert isinstance(result, ModelCreate)
    assert result.max_context is None


def test_model_form_cancel_returns_none() -> None:
    async def steps(app, pilot) -> None:
        _set(app, "m-name", "discard-me")
        app.screen.query_one("#m-cancel", Button).press()
        await pilot.pause()

    (result,) = _drive(ModelFormModal(), steps)
    assert result is None


# --------------------------------------------------------------------------- #
# ProfileFormModal.
# --------------------------------------------------------------------------- #


def test_profile_create_returns_create_with_parsed_numbers() -> None:
    async def steps(app, pilot) -> None:
        _set(app, "p-name", "tp2")
        _set(app, "p-tp", "2")
        _set(app, "p-maxlen", "131072")
        _set(app, "p-gpuutil", "0.92")
        _set(app, "p-quant", "fp8")
        app.screen.query_one("#p-submit", Button).press()
        await pilot.pause()

    (result,) = _drive(ProfileFormModal(), steps)
    assert isinstance(result, ProfileCreate)
    assert result.name == "tp2"
    assert result.tensor_parallel_size == 2
    assert result.max_model_len == 131072
    assert result.gpu_memory_utilization == 0.92
    assert result.quantization == "fp8"


def test_profile_edit_returns_update() -> None:
    existing = Profile(id="pid", name="p", runtime=RuntimeName.VLLM)

    async def steps(app, pilot) -> None:
        _set(app, "p-name", "p-renamed")
        app.screen.query_one("#p-submit", Button).press()
        await pilot.pause()

    (result,) = _drive(ProfileFormModal(existing), steps)
    assert isinstance(result, ProfileUpdate)
    assert result.name == "p-renamed"


def test_profile_form_requires_a_name() -> None:
    notes: list[str] = []

    async def steps(app, pilot) -> None:
        app.notify = lambda msg, *a, **k: notes.append(str(msg))  # type: ignore[method-assign]
        app.screen.query_one("#p-submit", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, ProfileFormModal)

    result = _drive(ProfileFormModal(), steps)
    assert result == []
    assert any("required" in n.lower() for n in notes)


# --------------------------------------------------------------------------- #
# CloneModal.
# --------------------------------------------------------------------------- #


def test_clone_prefills_copy_name_and_returns_request() -> None:
    async def steps(app, pilot) -> None:
        # Default value is "<current>-copy"; accept it as-is.
        assert app.screen.query_one("#clone-name", Input).value == "ornith-35b-copy"
        app.screen.query_one("#clone-confirm", Button).press()
        await pilot.pause()

    (result,) = _drive(CloneModal("src-id", "ornith-35b"), steps)
    assert isinstance(result, CloneRequest)
    assert result.source_id == "src-id"
    assert result.new_name == "ornith-35b-copy"


def test_clone_requires_a_new_name() -> None:
    notes: list[str] = []

    async def steps(app, pilot) -> None:
        app.notify = lambda msg, *a, **k: notes.append(str(msg))  # type: ignore[method-assign]
        app.screen.query_one("#clone-name", Input).value = "   "
        app.screen.query_one("#clone-confirm", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, CloneModal)

    result = _drive(CloneModal("src", "orig"), steps)
    assert result == []
    assert any("required" in n.lower() for n in notes)


def test_runtime_select_is_not_blank_by_default() -> None:
    """The backend Select defaults to vLLM, so the name-and-backend guard
    never trips on the backend alone."""

    async def steps(app, pilot) -> None:
        sel = app.screen.query_one("#m-runtime", Select)
        assert sel.value is not Select.BLANK
        assert sel.value == RuntimeName.VLLM.value

    _drive(ModelFormModal(), steps)
