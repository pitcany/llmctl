"""The preset launch is the widest-blast-radius action in the TUI.

Confirming it stops ollama, stops the Harbor ollama container, rewrites
vllm-tp.env, restarts the single machine-wide vLLM unit, blocks up to five
minutes on readiness, and rewrites the Hermes provider. The dialog used to say
only "Confirm to launch on the TP fleet unit (both GPUs)" — it disclosed none
of that, and never showed which model was about to be terminated.
"""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Static

from llmctl.services.preset_loader import PresetView
from llmctl.tui._modals_presets import PresetLaunchModal


def _view() -> PresetView:
    return PresetView(
        alias="ornith-35b",
        served_name="ornith-35b",
        model_id="deepreinforce-ai/Ornith-1.0-35B-FP8",
        family="qwen3_5_moe",
        param_count_b=35.0,
        tensor_parallel=2,
        quantization="fp8",
        source_path=None,
    )


class _Host(App[None]):
    def __init__(self, modal) -> None:
        super().__init__()
        self._modal = modal

    def on_mount(self) -> None:
        self.push_screen(self._modal)


def _texts(screen) -> str:
    return " ".join(str(w.content) for w in screen.query(Static))


def _run(modal, body):
    async def _main() -> None:
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(_main())


def test_launch_modal_discloses_the_full_blast_radius() -> None:
    """Every side effect the orchestrator performs must be stated."""
    modal = PresetLaunchModal(_view())

    async def body(app, pilot) -> None:
        blob = _texts(app.screen).lower()
        for expected in ("ollama", "harbor", "vllm-tp", "hermes"):
            assert expected in blob, (
                f"launch dialog never mentions {expected!r}: {blob}"
            )
        # The readiness wait is why the terminal appears to hang.
        assert "5 min" in blob or "300" in blob or "minute" in blob

    _run(modal, body)


def test_launch_modal_names_the_model_being_replaced() -> None:
    """The operator must be told what they are terminating."""
    modal = PresetLaunchModal(_view(), currently_served=["qwen3.6-27b"])

    async def body(app, pilot) -> None:
        assert "qwen3.6-27b" in _texts(app.screen)

    _run(modal, body)


def test_launch_modal_says_so_when_nothing_is_served() -> None:
    """An idle unit is worth stating too, so the field is never ambiguous."""
    modal = PresetLaunchModal(_view(), currently_served=[])

    async def body(app, pilot) -> None:
        blob = _texts(app.screen).lower()
        assert "not serving" in blob or "nothing" in blob or "idle" in blob

    _run(modal, body)


def test_launch_modal_defaults_to_cancel() -> None:
    """A reflexive Enter must not restart the machine-wide unit."""
    modal = PresetLaunchModal(_view())

    async def body(app, pilot) -> None:
        focused = app.screen.focused
        assert focused is not None and focused.id == "pick-cancel", (
            f"default focus is {focused.id if focused else None!r}; "
            "Enter would launch"
        )

    _run(modal, body)


def test_failed_launch_reports_what_was_left_stopped(monkeypatch) -> None:
    """A failure mid-sequence leaves the box with no inference backend.

    Fleet preflight stops ollama and the Harbor container before the vLLM
    restart is attempted. Nothing rolls that back, so if the restart then
    fails the operator must be told what is now down — result.fleet_stopped
    carried the answer and was never used.
    """
    from llmctl.services.vllm_orchestrator import OrchestratorResult
    from llmctl.tui import _data
    from llmctl.tui.app import MissionControlApp

    monkeypatch.setattr(_data, "get_preset_views_with_links", lambda: [])
    notes: list[str] = []

    class _Restart:
        ready = False
        error = "unit entered failed state"

    result = OrchestratorResult(spec=type("S", (), {"port": 8003})())
    result.fleet_stopped = ["ollama", "vllm-tp"]
    result.restart = _Restart()

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_presets()
            for _ in range(40):
                await pilot.pause(0.02)
                if app.screen.__class__.__name__.startswith("Presets"):
                    break
            monkeypatch.setattr(
                app, "notify", lambda msg, *a, **k: notes.append(str(msg))
            )
            app.screen._after_launch("vLLM TP: x", result)
            await pilot.pause()

    asyncio.run(_run())
    blob = " ".join(notes).lower()
    assert "ollama" in blob, (
        f"failed launch did not report that ollama was left stopped: {notes!r}"
    )
