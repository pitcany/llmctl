import asyncio
import time
from typing import Any

from llmctl.schemas import AdapterStatus, HealthState, Model, ModelStatus, RuntimeName
from llmctl.services.health import HealthService
from llmctl.services.lanes import Lane


def test_not_used_state():
    assert HealthState.NOT_USED.value == "not_used"
    assert HealthState("not_used") is HealthState.NOT_USED


_NOT_RUNNING = "Strata is installed (/usr/bin/python3) and not running; start it."


class _Adapter:
    def __init__(
        self,
        message: str,
        loaded: list[Model] | None = None,
        runtime: RuntimeName = RuntimeName.STRATA,
    ) -> None:
        self.runtime = runtime
        self.message = message
        self.loaded = loaded

    async def health_check(self) -> Any:
        return AdapterStatus(runtime=self.runtime, state=HealthState.OK, message=self.message)

    async def list_loaded_models(self) -> list[Model] | None:
        return self.loaded


class _Router:
    def __init__(self, adapters: dict[RuntimeName, _Adapter]) -> None:
        self.adapters = adapters

    def list_runtimes(self) -> list[RuntimeName]:
        return list(self.adapters)

    def get_adapter(self, runtime: RuntimeName) -> _Adapter:
        return self.adapters[runtime]


def _lane(name: str = "strata", runtime: str = "strata", port: int = 8007) -> Lane:
    return Lane(
        name=name,
        served_model=f"qwen3.8-flash-next-{name}",
        upstream=f"http://127.0.0.1:{port}",
        start_cmd=("true",),
        runtime=runtime,
    )


def _health(adapters, lanes, probe) -> dict[str, Any]:
    svc = HealthService(None, _Router(adapters), None, lanes=lanes, lane_probe=probe)
    return svc.get_health()["runtimes"]


def _answer(value: bool):
    async def probe(lane: Lane) -> bool:
        return value

    return probe


def test_rule_a_answering_lane_is_serving() -> None:
    rt = _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING)}, [_lane()], _answer(True))["strata"]
    assert rt["state"] == "ok"
    assert "serving via lane strata at http://127.0.0.1:8007" in rt["message"]
    assert "not running" not in rt["message"] and "not installed" not in rt["message"]
    assert {"served_name": "qwen3.8-flash-next-strata", "managed": False}.items() <= (
        rt["sessions"][0].items()
    )
    assert rt["serving_count"] == 1 and rt["unmanaged_count"] == 1
    assert rt["lane"] == "strata"


def test_rule_a_does_not_duplicate_a_listed_session() -> None:
    loaded = [
        Model(
            name="qwen3.8-flash-next-strata",
            runtime=RuntimeName.STRATA,
            source="x",
            status=ModelStatus.DISCOVERED,
        )
    ]
    rt = _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING, loaded)}, [_lane()], _answer(True))[
        "strata"
    ]
    assert rt["serving_count"] == 1


def test_rule_b_lane_not_running() -> None:
    rt = _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING)}, [_lane()], _answer(False))["strata"]
    assert rt["state"] == "ok"
    assert rt["message"] == (
        "lane strata is not running (http://127.0.0.1:8007); `llmctl run strata` starts it."
    )


def test_rule_f_session_never_says_not_running() -> None:
    loaded = [
        Model(name="m", runtime=RuntimeName.STRATA, source="m", status=ModelStatus.DISCOVERED)
    ]
    rt = _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING, loaded)}, [_lane()], _answer(False))[
        "strata"
    ]
    assert "not running" not in rt["message"]


def test_raising_probe_counts_as_false() -> None:
    async def boom(lane: Lane) -> bool:
        raise RuntimeError("x")

    rt = _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING)}, [_lane()], boom)["strata"]
    assert "is not running" in rt["message"]


def test_non_engine_runtimes_are_untouched() -> None:
    rt = _health(
        {RuntimeName.LMSTUDIO: _Adapter("not running here", runtime=RuntimeName.LMSTUDIO)},
        [_lane("x", "lmstudio")],
        _answer(True),
    )["lmstudio"]
    assert rt["message"] == "not running here" and "lane" not in rt


def test_lanes_are_probed_concurrently() -> None:
    async def slow(lane: Lane) -> bool:
        await asyncio.sleep(0.5)
        return True

    lanes = [_lane(f"l{i}", "strata", 8000 + i) for i in range(3)]
    start = time.monotonic()
    _health({RuntimeName.STRATA: _Adapter(_NOT_RUNNING)}, lanes, slow)
    assert time.monotonic() - start < 1.2


def _engine_health(runtime: RuntimeName, message: str, users: dict[str, list[str]], status: str):
    from llmctl.services.engines import EngineStatus

    svc = HealthService(
        None,
        _Router({runtime: _Adapter(message, runtime=runtime)}),
        None,
        lanes=[],
        engines=[EngineStatus(runtime.value, status, None)],
        model_users=users,
        lane_probe=_answer(False),
    )
    return svc.get_health()["runtimes"][runtime.value]


def test_rule_c_missing_engine_nobody_uses_is_not_used() -> None:
    rt = _engine_health(RuntimeName.FREETOKEN, "Run `llmctl setup freetoken`", {}, "missing")
    assert rt["state"] == "not_used" and rt["message"] == "not used"


def test_rule_d_missing_engine_with_users_names_models() -> None:
    rt = _engine_health(
        RuntimeName.FREETOKEN,
        "Run `llmctl setup freetoken`",
        {"freetoken": ["m1", "m2"]},
        "missing",
    )
    assert rt["state"] == "unavailable"
    assert "m1" in rt["message"] and "m2" in rt["message"]


def test_rule_e_missing_engine_never_says_installed() -> None:
    rt = _engine_health(
        RuntimeName.SGLANG,
        "SGLang is installed (/usr/bin/python3) and not running.",
        {"sglang": ["m"]},
        "missing",
    )
    assert "is installed" not in rt["message"] and "installed (" not in rt["message"]
