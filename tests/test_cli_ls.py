"""`llmctl ls`: whole names at 80 columns, and --json."""

from __future__ import annotations

import json

from rich.console import Console
from typer.testing import CliRunner

from llmctl import cli, cli_swap
from llmctl.services.lanes import SwapState, SwapTarget

KEYS = ("muse-glimmer-30b", "qwen3.8-27b-uncensored", "ollama-x")


def _state() -> SwapState:
    catalog = [
        SwapTarget(key=k, kind="session", model=m, engine="llama.cpp", upstream="")
        for k, m in zip(
            KEYS,
            ("a", "b", "hf.co/someorg/Some-Very-Long-Model-Name-GGUF:Q4_K_M-extra-long-tag"),
            strict=True,
        )
    ]
    return SwapState(
        catalog=catalog, serving={"muse-glimmer-30b"}, active_units=set(),
        vllm_unit="v", vllm_active=False,
    )


def test_names_whole_at_80_columns():
    c = Console(width=80, record=True)
    cli_swap.render_table(_state(), c)
    text = c.export_text()
    for k in KEYS:
        assert any(k in line for line in text.splitlines())
    assert "…" not in text
    assert "Models you can run" in text
    assert "llmctl run NAME" in text  # the caption stays


def test_ls_json(monkeypatch):
    monkeypatch.setattr(cli, "gather_state", lambda settings: _state())
    res = CliRunner().invoke(cli.app, ["ls", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert [r["number"] for r in data] == [1, 2, 3]
    documented = {"number", "name", "model", "engine", "kind", "cards", "state", "last_load_s"}
    assert all(documented <= set(r) for r in data)
    assert [r["name"] for r in data] == list(KEYS)
    assert {r["state"] for r in data} <= {"serving", "loading", ""}
    assert all(r["last_load_s"] is None or isinstance(r["last_load_s"], float) for r in data)
