from typing import Any

from llmctl.schemas import AdapterStatus, HealthState, RuntimeName
from llmctl.services.engines import EngineStatus
from llmctl.services.health import HealthService
from llmctl.services.lanes import Lane


class _Adapter:
    def __init__(self, runtime: RuntimeName, message: str, state=HealthState.UNAVAILABLE) -> None:
        self.runtime, self.message, self.state = runtime, message, state

    async def health_check(self) -> Any:
        return AdapterStatus(runtime=self.runtime, state=self.state, message=self.message)

    async def list_loaded_models(self) -> None:
        return None


class _Router:
    def __init__(self, adapter: _Adapter) -> None:
        self.adapter = adapter

    def list_runtimes(self) -> list[RuntimeName]:
        return [self.adapter.runtime]

    def get_adapter(self, runtime: RuntimeName) -> _Adapter:
        return self.adapter


def _run(runtime, message, rows, users, lanes=(), up=False, state=HealthState.UNAVAILABLE):
    async def probe(lane: Lane) -> bool:
        return up

    svc = HealthService(
        None,
        _Router(_Adapter(runtime, message, state)),
        None,
        lanes=list(lanes),
        engines=rows,
        model_users=users,
        lane_probe=probe,
    )
    return svc.get_health()["runtimes"][runtime.value]


_FT = RuntimeName.FREETOKEN
_HINT = "Run `llmctl setup freetoken`"


def test_rule_c_missing_unused() -> None:
    rt = _run(_FT, _HINT, [EngineStatus("freetoken", "missing", None)], {})
    assert rt["state"] == "not_used" and rt["message"] == "not used"


def test_rule_d_missing_used() -> None:
    rt = _run(
        _FT, _HINT, [EngineStatus("freetoken", "missing", None)], {"freetoken": list("abcd")}
    )
    assert rt["state"] == "unavailable"
    assert "a, b, c and 1 more" in rt["message"]
    assert rt["message"].endswith("`llmctl setup freetoken` installs it.")


def test_rule_e_never_says_installed() -> None:
    msg = "SGLang is installed (/usr/bin/python3) and not running..."
    rt = _run(
        RuntimeName.SGLANG, msg, [EngineStatus("sglang", "missing", None)], {"sglang": ["m"]}
    )
    assert "installed (" not in rt["message"] and "is installed" not in rt["message"]


def test_present_engine_and_unknown_row_unchanged() -> None:
    for rows in ([EngineStatus("freetoken", "path", "/x/ft")], []):
        rt = _run(_FT, "ok msg", rows, {}, state=HealthState.OK)
        assert rt["state"] == "ok" and rt["message"] == "ok msg"


def test_answering_lane_beats_not_used() -> None:
    lane = Lane(
        name="ft", served_model="m", upstream="http://127.0.0.1:1", start_cmd=("true",),
        runtime="freetoken",
    )
    rt = _run(_FT, _HINT, [EngineStatus("freetoken", "missing", None)], {}, [lane], up=True)
    assert rt["state"] == "ok"
