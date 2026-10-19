"""`llmctl stop` with no session ID, or with a name from `llmctl ls`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmctl import cli
from llmctl.services import stop_serving
from llmctl.services.lanes import Lane, SwapState, build_catalog
from llmctl.services.session_swap import GpuSession
from llmctl.services.stop_serving import StopDeps, pick, serving_now, stop_items

STRATA = Lane(
    name="strata",
    served_model="flash-strata",
    upstream="http://127.0.0.1:8007",
    start_cmd=("strata-start",),
    unit="strata.service",
    runtime="strata",
)
CODER = Lane(
    name="coder1",
    served_model="qwen-coder",
    upstream="http://127.0.0.1:11435",
    start_cmd=("coder-start",),
    unit="coder1.service",
    runtime="ollama",
)
GLM = GpuSession(id="0123abcd-ffff", name="glm", runtime="llama_cpp", port=8006, gpu_ids=(0, 1))


class _Spec:
    served_name = "qwen3.8-27b"


def _state(*, vllm: bool = False, lanes=(), sessions=()) -> SwapState:
    serving = {"qwen3.8-27b"} if vllm else set()
    serving |= {lane.name for lane in lanes}
    return SwapState(
        catalog=build_catalog({"qwen3.8-27b": _Spec()}, [STRATA, CODER], vllm_port=8003),
        serving=serving,
        active_units={lane.unit for lane in lanes},
        vllm_unit="vllm-tp",
        vllm_active=vllm,
        gpu_sessions=list(sessions),
    )


def test_serving_now_lists_vllm_lanes_and_sessions() -> None:
    items = serving_now(_state(vllm=True, lanes=[STRATA], sessions=[GLM]))
    assert [i.kind for i in items] == ["vllm", "lane", "session"]
    assert "qwen3.8-27b" in items[0].label
    assert items[1].ref == "strata.service"
    assert items[2].ref == GLM.id


def test_nothing_serving_is_an_empty_list() -> None:
    assert serving_now(_state()) == []


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("strata", "lane"),
        ("flash-strata", "lane"),
        ("strata.service", "lane"),
        ("qwen3.8-27b", "vllm"),
        ("vllm", "vllm"),
        ("glm", "session"),
        ("0123", "session"),  # a session-id prefix
    ],
)
def test_pick_by_any_name_ls_shows(name, kind) -> None:
    items = serving_now(_state(vllm=True, lanes=[STRATA], sessions=[GLM]))
    assert [i.kind for i in pick(items, name)] == [kind]


def test_stop_items_reports_each_result_and_survives_an_exception() -> None:
    items = serving_now(_state(vllm=True, lanes=[STRATA], sessions=[GLM]))

    def boom(ref: str) -> bool:
        raise RuntimeError("sudo said no")

    deps = StopDeps(stop_vllm=boom, stop_lane=lambda u: True, stop_session=lambda s: False)
    assert [ok for _, ok in stop_items(items, deps)] == [False, True, False]


# --- the CLI ---------------------------------------------------------------------------


@pytest.fixture
def stopped(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_is_live_session", lambda selector: False)
    monkeypatch.setattr(
        stop_serving,
        "real_deps",
        lambda settings: StopDeps(
            stop_vllm=lambda r: calls.append(("vllm", r)) or True,
            stop_lane=lambda r: calls.append(("lane", r)) or True,
            stop_session=lambda r: calls.append(("session", r)) or True,
        ),
    )
    return calls


def _stop(monkeypatch, state: SwapState, *args: str):
    monkeypatch.setattr(cli, "gather_state", lambda settings: state)
    return CliRunner().invoke(cli.app, ["stop", *args])


def test_stop_with_no_id_stops_the_one_thing_serving(monkeypatch, stopped) -> None:
    res = _stop(monkeypatch, _state(lanes=[STRATA]))
    assert res.exit_code == 0, res.output
    assert stopped == [("lane", "strata.service")]
    assert "Stopped lane strata" in " ".join(res.output.split())


def test_several_serving_need_a_name_or_all(monkeypatch, stopped) -> None:
    state = _state(lanes=[STRATA, CODER])
    res = _stop(monkeypatch, state)
    assert res.exit_code == 1
    assert "llmctl stop --all" in res.output and stopped == []
    res = _stop(monkeypatch, state, "coder1")
    assert res.exit_code == 0, res.output
    assert stopped == [("lane", "coder1.service")]
    res = _stop(monkeypatch, state, "--all")
    assert stopped[1:] == [("lane", "strata.service"), ("lane", "coder1.service")]


def test_nothing_serving_says_so(monkeypatch, stopped) -> None:
    res = _stop(monkeypatch, _state())
    assert res.exit_code == 0
    assert "Nothing llmctl can stop is serving." in res.output


def test_an_unknown_name_is_refused(monkeypatch, stopped) -> None:
    monkeypatch.setattr(cli.SessionService, "get_session", lambda self, s: None)
    res = _stop(monkeypatch, _state(lanes=[STRATA]), "nope")
    assert res.exit_code == 2
    assert "Nothing serving is called nope" in " ".join(res.output.split())
    assert stopped == []


def test_a_failed_stop_exits_1(monkeypatch, stopped) -> None:
    monkeypatch.setattr(
        stop_serving,
        "real_deps",
        lambda settings: StopDeps(
            stop_vllm=lambda r: False, stop_lane=lambda r: False, stop_session=lambda r: False
        ),
    )
    res = _stop(monkeypatch, _state(vllm=True))
    assert res.exit_code == 1
    assert "Could not stop vLLM unit vllm-tp" in " ".join(res.output.split())


def test_a_live_session_id_keeps_the_old_path(monkeypatch, stopped) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli, "_is_live_session", lambda selector: True)
    monkeypatch.setattr(cli, "_stop_session", lambda sid, **k: seen.append(sid))
    res = _stop(monkeypatch, _state(lanes=[STRATA]), "0123abcd")
    assert res.exit_code == 0, res.output
    assert seen == ["0123abcd"] and stopped == []


def test_an_ambiguous_id_prefix_picks_nothing() -> None:
    other = GpuSession(id="0999eeee", name="gemma", runtime="llama_cpp", port=8010, gpu_ids=(1,))
    items = serving_now(_state(sessions=[GLM, other]))
    assert pick(items, "0") == []
    assert [i.ref for i in pick(items, "01")] == [GLM.id]


def test_preview_line_names_the_cards_each_item_frees() -> None:
    from dataclasses import replace

    from llmctl.services.stop_serving import preview_line

    lane1 = replace(CODER, gpus=frozenset({1}))
    bare = GpuSession(id="9999abcd-ffff", name="bare", runtime="llama_cpp", port=8007, gpu_ids=())
    state = _state(vllm=True, lanes=[lane1], sessions=[GLM, bare])
    state.catalog = build_catalog({"qwen3.8-27b": _Spec()}, [STRATA, lane1], vllm_port=8003)
    lines = [preview_line(i) for i in serving_now(state)]
    assert lines[0].endswith("frees cards all")
    assert lines[1] == "  lane coder1 (coder1.service, qwen-coder) — frees cards 1"
    assert lines[2].endswith("frees cards 0+1")
    assert lines[3].endswith("frees cards all")


# --- --dry-run and the preview before a real stop ---------------------------------------


@pytest.fixture
def spies(monkeypatch, stopped):
    hits: list[str] = []
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: hits.append("confirm"))
    monkeypatch.setattr(stop_serving, "stop_items", lambda *a, **k: hits.append("stop") or [])
    monkeypatch.setattr(cli.SessionService, "stop", lambda *a, **k: hits.append("svc"))
    return hits


def test_dry_run_previews_and_changes_nothing(monkeypatch, spies) -> None:
    res = _stop(monkeypatch, _state(lanes=[STRATA, CODER]), "--all", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "Would stop:" in res.output
    assert "lane strata" in res.output and "lane coder1" in res.output
    assert spies == []


def test_dry_run_still_needs_a_choice_and_a_known_name(monkeypatch, spies) -> None:
    state = _state(lanes=[STRATA, CODER])
    res = _stop(monkeypatch, state, "--dry-run")
    assert res.exit_code == 1 and "llmctl stop --all" in res.output
    monkeypatch.setattr(cli.SessionService, "get_session", lambda self, s: None)
    assert _stop(monkeypatch, state, "nope", "--dry-run").exit_code == 2
    assert spies == []


def test_dry_run_with_nothing_serving(monkeypatch, spies) -> None:
    res = _stop(monkeypatch, _state(), "--dry-run")
    assert res.exit_code == 0
    assert "Nothing llmctl can stop is serving." in res.output


def test_dry_run_of_a_live_session(monkeypatch, spies) -> None:
    from llmctl.schemas import Session

    monkeypatch.setattr(cli, "_is_live_session", lambda selector: True)
    sess = Session(
        id="0123abcd-ffff", runtime="llama_cpp", port=8006, gpu_ids=[0, 1], served_name="glm"
    )
    monkeypatch.setattr(cli.SessionService, "get_session", lambda self, s: sess)
    res = _stop(monkeypatch, _state(), "0123abcd", "--dry-run")
    out = " ".join(res.output.split())
    assert res.exit_code == 0, res.output
    assert "Would stop:" in out and "glm" in out and "0123abcd" in out and "0, 1" in out
    assert spies == []


def test_a_real_stop_previews_before_confirming(monkeypatch, stopped) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: order.append("confirm"))
    monkeypatch.setattr(cli.console, "print", lambda *a, **k: order.append(str(a[0])))
    res = _stop(monkeypatch, _state(lanes=[STRATA]))
    assert res.exit_code == 0, res.output
    assert order[0] == "Would stop:" and "lane strata" in order[1] and order[2] == "confirm"


@pytest.mark.parametrize("extra", [(), ("--dry-run",)])
def test_an_ambiguous_session_prefix_is_refused_cleanly(monkeypatch, spies, extra) -> None:
    from llmctl.services.sessions import AmbiguousSessionError

    def boom(self, selector):
        raise AmbiguousSessionError("prefix 01 matches 2 sessions")

    monkeypatch.setattr(cli, "_is_live_session", lambda selector: True)
    monkeypatch.setattr(cli.SessionService, "get_session", boom)
    res = _stop(monkeypatch, _state(), "01", *extra)
    assert res.exit_code == 1, res.output
    assert "Stop refused" in res.output
    assert spies == []


def test_dry_run_of_a_known_but_not_serving_session(monkeypatch, spies) -> None:
    from llmctl.schemas import Session

    sess = Session(id="0123abcd-ffff", runtime="llama_cpp", port=8006, gpu_ids=[2], served_name="g")
    monkeypatch.setattr(cli, "_is_live_session", lambda selector: False)
    monkeypatch.setattr(cli.SessionService, "get_session", lambda self, s: sess)
    res = _stop(monkeypatch, _state(), "0123abcd", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "Would stop:" in res.output and "0123abcd" in res.output
    assert spies == []
