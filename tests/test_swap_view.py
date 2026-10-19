"""swap_view: the one builder behind `ls --json` and the API."""

from __future__ import annotations

from llmctl import cli_swap
from llmctl.config import RouterSettings
from llmctl.services import swap_view
from llmctl.services.lanes import SwapState, SwapTarget


def _state() -> SwapState:
    catalog = [
        SwapTarget(key="lane-a", kind="lane", model="a", engine="vllm", upstream="http://h:1"),
        SwapTarget(key="sess-b", kind="session", model="b", engine="llama.cpp", upstream=""),
    ]
    return SwapState(
        catalog=catalog,
        serving={"lane-a"},
        active_units=set(),
        vllm_unit="v",
        vllm_active=False,
    )


def test_gateway_url():
    r = RouterSettings()
    assert swap_view.gateway_url(r) == f"http://{r.host}:{r.port}/v1"


def test_endpoint_for_matches_rows_json():
    state = _state()
    rows = swap_view.rows_json(state)
    for target, row in zip(state.catalog, rows, strict=True):
        assert swap_view.endpoint_for(state, target) == row["endpoint"]
    assert rows[0]["endpoint"] == "http://h:1/v1"
    assert rows[1]["endpoint"] is None


def test_cli_swap_reexports():
    assert cli_swap.rows_json is swap_view.rows_json
    assert cli_swap.gateway_aliases is swap_view.gateway_aliases
    assert cli_swap._aliases_for is swap_view._aliases_for
