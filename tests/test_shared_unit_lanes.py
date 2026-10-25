"""Two Ollama lanes on one unit: only the loaded model reads serving.

`coder1` and `gemma1` are two models on ollama-gpu1 (:11435). Ollama's
/v1/models lists every installed model, so both lanes read "serving" whenever
the unit was up, and `llmctl run gemma1` said "nothing to do" while Qwen was
loaded. Ollama lanes are now judged by /api/ps, and a lane whose sibling is
serving is neither "loading" nor something a swap stops.
"""

from __future__ import annotations

import io
import json

from llmctl.services.backends import probe_ollama_loaded
from llmctl.services.lanes import Lane, SwapState, SwapTarget, serving_keys

UPSTREAM = "http://127.0.0.1:11435"
UNIT = "ollama-gpu1.service"


def _row(key: str, model: str) -> SwapTarget:
    lane = Lane(
        name=key,
        served_model=model,
        upstream=UPSTREAM,
        start_cmd=("/bin/start", "--take-gpus"),
        unit=UNIT,
        runtime="ollama",
        gpus=frozenset({1}),
    )
    return SwapTarget(
        key=key, kind="lane", model=model, engine="Ollama", upstream=UPSTREAM, lane=lane
    )


CODER = _row("coder1", "qwen3.6:35b")
GEMMA = _row("gemma1", "gemma4:26b")


def _installed(url: str) -> list[str]:
    return ["qwen3.6:35b", "gemma4:26b"]  # /v1/models: everything installed


def _loaded(url: str) -> list[str]:
    return ["gemma4:26b"]  # /api/ps: what is in memory


def _state(serving: set[str]) -> SwapState:
    return SwapState(
        catalog=[CODER, GEMMA],
        serving=serving,
        active_units={UNIT},
        vllm_unit="vllm-tp.service",
        vllm_active=False,
    )


def test_ollama_lanes_are_judged_by_the_loaded_model() -> None:
    assert serving_keys([CODER, GEMMA], _installed) == {"coder1", "gemma1"}
    assert serving_keys([CODER, GEMMA], _installed, {"ollama": _loaded}) == {"gemma1"}


def test_the_sibling_of_a_serving_lane_is_idle_not_loading() -> None:
    state = _state({"gemma1"})
    assert state.state_of(GEMMA) == "serving"
    assert state.state_of(CODER) == ""
    assert state.active_lanes() == [GEMMA.lane]


def test_a_swap_to_the_sibling_stops_the_unit_once_and_names_the_loaded_model() -> None:
    state = _state({"coder1"})
    preview = state.preview(GEMMA)
    assert not preview.already_serving
    assert [s for s in preview.stops if s.startswith(UNIT)] == [f"{UNIT} (qwen3.6:35b)"]
    assert state.lanes_in_the_way(GEMMA) == [CODER.lane]


def test_with_nothing_loaded_yet_both_lanes_may_be_loading() -> None:
    state = _state(set())
    assert state.state_of(CODER) == state.state_of(GEMMA) == "loading"


def test_probe_ollama_loaded_reads_api_ps() -> None:
    seen: list[str] = []

    def http_get(url: str, timeout: float):
        seen.append(url)
        body = {"models": [{"name": "gemma4:26b", "model": "gemma4:26b"}]}
        return io.BytesIO(json.dumps(body).encode())

    assert probe_ollama_loaded(UPSTREAM, http_get=http_get) == ["gemma4:26b", "gemma4:26b"]
    assert seen == [f"{UPSTREAM}/api/ps"]

    def refused(url: str, timeout: float):
        raise OSError("connection refused")

    assert probe_ollama_loaded(UPSTREAM, http_get=refused) is None
