"""`llmctl run N` takes an `ls` row number; unknown names get a helpful error."""

from __future__ import annotations

from typer.testing import CliRunner

from llmctl import cli
from llmctl.services.lanes import SwapTarget, close_matches
from llmctl.services.vllm_orchestrator import UnknownPresetError
from tests.test_cli_swap import _run, _state, calls  # noqa: F401


def _boom(*a, **k):
    raise AssertionError("start_vllm_tp must not run")


def test_row_number_matches_key(monkeypatch, calls):  # noqa: F811
    state = _state(set(), set())
    by_num = _run(monkeypatch, state, "2", "--dry-run")
    by_key = _run(monkeypatch, state, state.catalog[1].key, "--dry-run")
    assert by_num.exit_code == 0 and by_num.output == by_key.output
    assert calls["lane"] == [] and calls["stop"] == []


def test_row_number_out_of_range(monkeypatch, calls):  # noqa: F811
    monkeypatch.setattr(cli, "gather_state", lambda s: _state(set(), set()))
    monkeypatch.setattr(cli, "start_vllm_tp", _boom)
    res = CliRunner().invoke(cli.app, ["run", "99"])
    assert res.exit_code == 2
    assert "1-2" in res.output
    assert calls["stop"] == [] and calls["lane"] == []


def test_literal_key_beats_row_number(monkeypatch, calls):  # noqa: F811
    state = _state(set(), set())
    state.catalog.append(
        SwapTarget(key="1", kind="vllm", model="literal-row-one", engine="vLLM", upstream="")
    )
    res = _run(monkeypatch, state, "1", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "literal-row-one" in res.output


def test_unknown_name_suggests(monkeypatch, calls):  # noqa: F811
    monkeypatch.setattr(cli, "gather_state", lambda s: _state(set(), set()))

    def raises(*a, **k):
        # The real message, not the bare name: the handler must not echo it.
        raise UnknownPresetError("unknown preset 'qwen3.8-2'. Available: qwen3.8-27b")

    monkeypatch.setattr(cli, "start_vllm_tp", raises)
    res = CliRunner().invoke(cli.app, ["run", "qwen3.8-2"])
    assert res.exit_code == 2
    assert "Unknown preset" not in res.output
    assert "Models you can run" not in res.output
    for text in ("No model named 'qwen3.8-2'", "Did you mean", "qwen3.8-27b", "llmctl ls"):
        assert text in res.output
    assert "Available:" not in res.output
    assert calls["stop"] == []


def test_close_matches_limited_to_three():
    rows = [
        SwapTarget(key=f"model-{i}", kind="vllm", model=f"model-{i}x", engine="v", upstream="")
        for i in range(6)
    ]
    out = close_matches("model-1", rows)
    assert 0 < len(out) <= 3
    assert set(out) <= {n for r in rows for n in (r.key, r.model)}
