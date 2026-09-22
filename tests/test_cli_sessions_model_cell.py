"""`llmctl sessions` must name what each session serves.

The Model column printed ``model_id`` alone, which is None for adopted
sessions and for the vLLM unit — so the column was empty on every row of a
table whose whole subject is what is running.
"""

from __future__ import annotations

from llmctl.cli import _session_model_cell
from llmctl.db import RuntimeName
from llmctl.schemas import Session


def _session(**kw) -> Session:
    return Session(runtime=RuntimeName.VLLM, **kw)


def test_served_name_is_preferred():
    cell = _session_model_cell(_session(served_name="qwen3.8-27b"), {})
    assert cell == "qwen3.8-27b"


def test_model_id_is_resolved_to_a_registry_name():
    session = _session(model_id="abc-123")
    assert _session_model_cell(session, {"abc-123": "deepseek-v4-flash"}) == "deepseek-v4-flash"


def test_unknown_model_id_falls_back_to_the_id():
    assert _session_model_cell(_session(model_id="abc-123"), {}) == "abc-123"


def test_no_model_information_is_a_dash_not_a_blank():
    assert _session_model_cell(_session(), {}) == "—"


def test_served_name_wins_over_the_registry_lookup():
    session = _session(served_name="qwen3.8-27b", model_id="abc-123")
    assert _session_model_cell(session, {"abc-123": "something-else"}) == "qwen3.8-27b"
