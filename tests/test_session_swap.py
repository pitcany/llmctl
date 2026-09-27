"""`llmctl swap` onto a llama.cpp model is a real swap, with rollback.

The costly mistake here is stopping the tutor's vLLM for a model that can
never load. These tests pin that the fit is decided against card sizes
*before* anything stops, the order things are stopped in, that a failure
puts everything back (vLLM, lanes, other llama-servers), and that swapping
to vLLM or a lane first stops the llama-servers llmctl started.
"""

from __future__ import annotations

from types import SimpleNamespace

from llmctl.services.lane_swap import LaneSwapDeps
from llmctl.services.lanes import Lane, SwapState, SwapTarget, build_catalog
from llmctl.services.session_swap import (
    GpuSession,
    Placement,
    place,
    start_request_fields,
    swap_to_session,
)

# -- placement -----------------------------------------------------------------


def test_place_uses_one_card_when_it_fits() -> None:
    assert place("m", 20.0, [32.0, 32.0]).cards == 1


def test_place_splits_across_cards_when_one_is_too_small() -> None:
    # gpt-oss-120b: ~59 GB does not fit one 32 GB card, does fit two.
    assert place("gpt-oss", 59.0, [31.8, 31.8]).cards == 2


def test_place_refuses_what_no_combination_holds() -> None:
    result = place("huge", 90.0, [32.0, 32.0])
    assert result.cards == 0 and "Nothing was stopped" in (result.problem or "")


def test_place_without_an_estimate_says_so() -> None:
    result = place("m", None, [32.0])
    assert result.cards == 1 and "no VRAM estimate" in (result.note or "")


# -- the swap --------------------------------------------------------------------


LANE = Lane(
    name="coder1",
    served_model="coder",
    upstream="http://127.0.0.1:11435",
    start_cmd=("coder1-start",),
    unit="coder1.service",
    runtime="ollama",
)
OTHER = GpuSession(id="s-other", name="other", runtime="llama_cpp", port=8001, gpu_ids=(0,))


class _Recorder:
    """Every side effect, in order, as ``(verb, what)``."""

    def __init__(self, *, quiet_rc: int = 0) -> None:
        self.events: list[tuple[str, str]] = []
        self.quiet_rc = quiet_rc

    def deps(self) -> LaneSwapDeps:
        def run_quiet(argv):
            self.events.append(("systemctl", " ".join(argv[-2:])))
            return self.quiet_rc

        def run_script(argv, timeout):
            self.events.append(("script", argv[0]))
            return 0

        return LaneSwapDeps(
            run_quiet=run_quiet,
            run_script=run_script,
            probe=lambda upstream: ["coder"],
            sleep=lambda s: self.events.append(("sleep", str(s))),
            log=lambda m: None,
            record=lambda k, s: self.events.append(("record", k)),
        )

    def stop(self, sid):
        self.events.append(("stop-session", sid))

    def restart(self, sid):
        self.events.append(("restart-session", sid))


def _session(status="running", sid="s-new"):
    return SimpleNamespace(id=sid, status=SimpleNamespace(value=status), error=None)


def _swap(rec: _Recorder, start, *, placement=None, vllm_active=True):
    return swap_to_session(
        "gpt-oss",
        placement=placement or Placement(cards=2),
        start=start,
        stop=rec.stop,
        restart=rec.restart,
        gpu_sessions=[OTHER],
        active_lanes=[LANE],
        vllm_active=vllm_active,
        vllm_unit="vllm-tp",
        deps=rec.deps(),
    )


def test_swap_stops_everything_on_the_gpus_then_starts() -> None:
    rec = _Recorder()
    cards: list[int] = []
    result = _swap(rec, lambda n: cards.append(n) or _session())
    assert result.ok
    assert rec.events[:3] == [
        ("stop-session", "s-other"),
        ("systemctl", "stop coder1.service"),
        ("systemctl", "stop vllm-tp"),
    ]
    assert cards == [2]  # the placement's card count reaches the scheduler
    assert ("record", "gpt-oss") in rec.events


def test_a_model_that_cannot_fit_stops_nothing() -> None:
    rec = _Recorder()
    result = _swap(
        rec,
        lambda n: _session(),
        placement=Placement(cards=0, problem="needs 90 GB. Nothing was stopped."),
    )
    assert not result.ok and "Nothing was stopped" in (result.error or "")
    assert rec.events == []


def test_failed_start_puts_everything_back() -> None:
    rec = _Recorder()
    result = _swap(rec, lambda n: _session(status="failed"))
    assert not result.ok and "did not come up" in (result.error or "")
    after = rec.events[3:]
    assert ("stop-session", "s-new") in after  # the half-started server
    assert ("systemctl", "start vllm-tp") in after  # vLLM back on its old preset
    assert ("script", "coder1-start") in after  # the lane back
    assert ("restart-session", "s-other") in after  # the other llama-server back
    assert "vllm-tp" in result.restored


def test_scheduler_refusal_from_lagging_vram_is_retried() -> None:
    rec = _Recorder()
    attempts = {"n": 0}

    def start(n):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Estimated 30 GB on GPU 0 exceeds free 2 GB")
        return _session()

    assert _swap(rec, start).ok
    assert attempts["n"] == 3 and rec.events.count(("sleep", "3.0")) == 2


def test_vllm_that_will_not_stop_aborts_before_starting() -> None:
    rec = _Recorder(quiet_rc=1)
    started: list[int] = []
    result = _swap(rec, lambda n: started.append(n) or _session())
    assert not result.ok and started == []


# -- preview + the reverse direction ------------------------------------------


def _state(gpu_sessions) -> SwapState:
    class _Spec:
        served_name = "qwen3.8-27b"

    return SwapState(
        catalog=build_catalog({"qwen3.8-27b": _Spec()}, [LANE], vllm_port=8003),
        serving={"qwen3.8-27b"},
        active_units={"coder1.service"},
        vllm_unit="vllm-tp",
        vllm_active=True,
        gpu_sessions=gpu_sessions,
    )


def test_preview_of_a_session_swap_names_what_stops() -> None:
    target = SwapTarget(
        key="gpt-oss", kind="session", model="gpt-oss", engine="llama.cpp", upstream="",
        placement=Placement(cards=2),
    )
    lines = "\n".join(_state([OTHER]).preview(target).lines())
    assert "vllm-tp (qwen3.8-27b)" in lines
    assert "coder1.service" in lines
    assert "llama-server for other, port 8001" in lines
    assert "llama-server for gpt-oss on 2 GPUs" in lines
    assert "llmctl picks the port" in lines


def test_preview_of_a_session_swap_shows_a_pinned_port() -> None:
    target = SwapTarget(
        key="gpt-oss", kind="session", model="gpt-oss", engine="llama.cpp", upstream="",
        placement=Placement(cards=2), port=8005,
    )
    lines = "\n".join(_state([]).preview(target).lines())
    assert "gpt-oss on 2 GPUs (split); port 8005" in lines


def test_vllm_swap_preview_lists_llama_servers_in_the_way() -> None:
    state = _state([OTHER])
    target = next(t for t in state.catalog if t.kind == "vllm")
    assert "llama-server for other, port 8001" in "\n".join(state.preview(target).lines())
    assert state.sessions_in_the_way(target) == [OTHER]


def test_cli_vllm_swap_pauses_llama_servers_and_restores_them(monkeypatch) -> None:
    import llmctl.cli_swap as cli_swap

    seen: list[str] = []
    monkeypatch.setattr(cli_swap, "load_settings", lambda: object())
    monkeypatch.setattr(
        cli_swap,
        "pause_for_swap",
        lambda state, target, **kw: (seen.append("pause") or list(state.gpu_sessions), None),
    )
    monkeypatch.setattr(
        cli_swap,
        "resume_after_failed_swap",
        lambda sessions, **kw: seen.append(f"resume {[s.id for s in sessions]}")
        or SimpleNamespace(restored=[], restore_failed=[]),
    )
    monkeypatch.setattr(cli_swap, "stop_user_unit", lambda unit, deps: True)
    monkeypatch.setattr(cli_swap, "restore_previous", lambda *a, **k: None)
    from rich.console import Console

    paused = cli_swap.stop_lanes_for_vllm(_state([OTHER]), Console(quiet=True))
    assert seen == ["pause"] and paused.sessions == [OTHER]
    cli_swap.restore_lanes(paused, Console(quiet=True))
    assert seen[-1] == "resume ['s-other']"


# -- start request ---------------------------------------------------------------


def test_swap_start_uses_the_models_default_profile() -> None:
    record = SimpleNamespace(id="m1", runtime="llama_cpp", default_profile_id="p1")
    fields = start_request_fields(record, 2)
    assert fields["profile_id"] == "p1"
    assert fields["parameters"] == {"tensor_parallel_size": 2}


def test_swap_start_card_count_overrides_the_profile() -> None:
    record = SimpleNamespace(id="m1", runtime="llama_cpp", default_profile_id=None)
    fields = start_request_fields(record, 1)
    assert fields["profile_id"] is None
    # Explicit even for one card, so a profile's tensor_parallel_size cannot win.
    assert fields["parameters"] == {"tensor_parallel_size": 1}
