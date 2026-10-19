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
        catalog=catalog,
        serving={"muse-glimmer-30b"},
        active_units=set(),
        vllm_unit="v",
        vllm_active=False,
    )


def test_names_whole_at_80_columns():
    c = Console(width=80, record=True)
    cli_swap.render_table(_state(), c)
    text = c.export_text()
    for k in KEYS:
        assert any(k in line for line in text.splitlines())
    # Model may be cut to fit; Name (the 4th cell) never is.
    names = [line.split("│")[4] for line in text.splitlines() if line.startswith("│")]
    assert names and not any("…" in cell for cell in names)
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


def test_one_line_per_row_and_no_wider_than_the_console():
    """Folding Model made each row several lines tall; cropping cut Last load off."""
    for width in (80, 100):
        c = Console(width=width, record=True)
        cli_swap.render_table(_state(), c)
        lines = c.export_text().splitlines()
        body = [line for line in lines if line.startswith("│")]
        assert len(body) == len(KEYS), "\n".join(lines)
        assert all(len(line) <= width for line in lines)
        assert all(line.rstrip().endswith("│") for line in body)  # right edge not cropped


# -- endpoints and gateway aliases ------------------------------------------


def _lane_state(serving: bool = True) -> SwapState:
    catalog = [
        SwapTarget(
            key="lane[x]",
            kind="lane",
            model="org/m[1]",
            engine="llama.cpp",
            upstream="http://127.0.0.1:8001",
        ),
        SwapTarget(key="sess", kind="session", model="s", engine="llama.cpp", upstream=""),
    ]
    return SwapState(
        catalog=catalog,
        serving={"lane[x]", "sess"} if serving else set(),
        active_units=set(),
        vllm_unit="v",
        vllm_active=False,
    )


def _aliases():
    from llmctl.services.gateway import AliasView

    return [
        AliasView("b", "t", "id", "org/m[1]", True),
        AliasView("a", "t", "id", "org/m[1]", True),
        AliasView("sick", "t", None, "org/m[1]", False),
        AliasView("other", "t", "id", "zzz", True),
    ]


def test_rows_json_endpoint_and_aliases():
    data = cli_swap.rows_json(_lane_state(), _aliases())
    assert data[0]["endpoint"] == "http://127.0.0.1:8001/v1"
    assert data[0]["aliases"] == ["local-a", "local-b"]
    assert data[1]["endpoint"] is None and data[1]["aliases"] == []
    idle = cli_swap.rows_json(_lane_state(False), _aliases())
    assert idle[0]["aliases"] == []


def test_render_endpoints_lines():
    from llmctl.config import RouterSettings

    c = Console(width=200, record=True)
    cli_swap.render_endpoints(_lane_state(), c, _aliases(), RouterSettings())
    lines = c.export_text().splitlines()
    assert lines[0] == "lane[x] → http://127.0.0.1:8001/v1  (model org/m[1])"
    r = RouterSettings()
    assert lines[1] == f"  gateway: http://{r.host}:{r.port}/v1 as local-a, local-b"
    assert len(lines) == 2


def test_render_endpoints_silent_when_nothing_serving():
    from llmctl.config import RouterSettings

    c = Console(record=True)
    cli_swap.render_endpoints(_lane_state(False), c, _aliases(), RouterSettings())
    assert c.export_text() == ""


def test_ls_survives_gateway_failure(monkeypatch):
    def boom(self):
        raise RuntimeError("db down")

    from llmctl.services.gateway import GatewayService

    monkeypatch.setattr(GatewayService, "alias_view", boom)
    monkeypatch.setattr(cli, "gather_state", lambda settings: _lane_state())
    assert cli_swap.gateway_aliases(cli.load_settings()) == []
    res = CliRunner().invoke(cli.app, ["ls", "--json"])
    assert res.exit_code == 0
    assert all(r["aliases"] == [] for r in json.loads(res.stdout))
    assert CliRunner().invoke(cli.app, ["ls"]).exit_code == 0
