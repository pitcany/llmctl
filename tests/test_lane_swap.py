"""Swapping onto a lane: order of operations, readiness, and rollback."""

from __future__ import annotations

from llmctl.services.lane_swap import LaneSwapDeps, swap_to_lane
from llmctl.services.lanes import Lane

FT = Lane(
    name="flash",
    served_model="ft-model",
    upstream="http://127.0.0.1:1919",
    start_cmd=("ft-start", "--take-gpus"),
    health_timeout_s=10,
    unit="ft.service",
)
LLAMA = Lane(
    name="dsv4",
    served_model="ds-model",
    upstream="http://127.0.0.1:8000",
    start_cmd=("dsv4-start", "--take-gpus"),
    health_timeout_s=10,
    unit="ds.service",
)


class Fake:
    """Records every side effect; knobs decide what fails."""

    def __init__(self, *, script_rc=None, ready=None, stop_fails=()):
        self.events: list[str] = []
        self.script_rc = script_rc or {}
        self.ready = ready if ready is not None else {"ft-model", "ds-model"}
        self.stop_fails = set(stop_fails)
        self.recorded: list[tuple[str, float]] = []
        self.now = 0.0

    def deps(self) -> LaneSwapDeps:
        return LaneSwapDeps(
            run_script=self.run_script,
            run_quiet=self.run_quiet,
            probe=self.probe,
            clock=lambda: self.now,
            sleep=self.sleep,
            log=lambda _msg: None,
            record=lambda key, s: self.recorded.append((key, s)),
            poll_interval_s=5,
        )

    def run_script(self, argv, _timeout):
        self.events.append("script " + argv[0])
        self.now += 60
        return self.script_rc.get(argv[0], 0)

    def run_quiet(self, argv):
        self.events.append(" ".join(argv))
        return 1 if argv[-1] in self.stop_fails else 0

    def probe(self, upstream):
        return [m for m in self.ready if (m == "ft-model") == upstream.endswith(":1919")]

    def sleep(self, seconds):
        self.now += seconds


def test_success_stops_other_lanes_then_starts_and_records():
    fake = Fake()
    result = swap_to_lane(
        FT, other_active=[LLAMA], vllm_active=False, vllm_unit="vllm-tp", deps=fake.deps()
    )
    assert result.ok
    assert fake.events == ["systemctl --user stop ds.service", "script ft-start"]
    assert fake.recorded == [("flash", 60.0)]


def test_failed_start_rolls_back_to_vllm_and_other_lanes():
    fake = Fake(script_rc={"ft-start": 1})
    result = swap_to_lane(
        FT, other_active=[LLAMA], vllm_active=True, vllm_unit="vllm-tp", deps=fake.deps()
    )
    assert not result.ok
    assert result.error == "start script exited 1"
    assert fake.events == [
        "systemctl --user stop ds.service",
        "script ft-start",
        "systemctl --user stop ft.service",  # half-started target
        "sudo -n systemctl start vllm-tp",  # env untouched -> previous preset
        "script dsv4-start",
    ]
    assert result.restored == ["vllm-tp", "ds.service"]
    assert fake.recorded == []


def test_readiness_timeout_is_a_failure():
    fake = Fake(ready=set())
    result = swap_to_lane(
        FT, other_active=[], vllm_active=False, vllm_unit="vllm-tp", deps=fake.deps()
    )
    assert not result.ok
    assert "not listed" in (result.error or "")
    assert "systemctl --user stop ft.service" in fake.events


def test_a_lane_that_will_not_stop_aborts_before_starting():
    fake = Fake(stop_fails={"ds.service"})
    result = swap_to_lane(
        FT, other_active=[LLAMA], vllm_active=False, vllm_unit="vllm-tp", deps=fake.deps()
    )
    assert not result.ok
    assert "script ft-start" not in fake.events
    assert result.error == "could not stop ds.service"


def test_swap_to_vllm_stops_lanes_and_restores_them_on_failure():
    from types import SimpleNamespace

    from llmctl.services.lane_swap import swap_to_vllm

    fake = Fake()
    result, lanes = swap_to_vllm([FT], lambda: SimpleNamespace(ok=False), deps=fake.deps())
    assert result.ok is False
    assert fake.events == ["systemctl --user stop ft.service", "script ft-start"]
    assert lanes.restored == ["ft.service"]


def test_swap_to_vllm_does_not_start_when_a_lane_will_not_stop():
    from llmctl.services.lane_swap import swap_to_vllm

    fake = Fake(stop_fails={"ft.service"})
    started: list[bool] = []
    result, lanes = swap_to_vllm([FT], lambda: started.append(True), deps=fake.deps())
    assert result is None and started == []
    assert lanes.error == "could not stop ft.service"
