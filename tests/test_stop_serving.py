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
