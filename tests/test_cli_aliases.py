from __future__ import annotations

import io
import json
from types import SimpleNamespace

from rich.console import Console
from typer.testing import CliRunner

from llmctl import cli
from llmctl.services.gateway import GatewayService

UUID = "0123abcd-4567-89ef-0123-456789abcdef"


def _rows():
    return [
        SimpleNamespace(
            name="long-context",
            target="qwen3.8-flash-next-strata",
            resolved_session_id=UUID,
            resolved_served_name="qwen3.8-flash-next-strata",
            healthy=True,
        ),
        SimpleNamespace(
            name="coding",
            target="local-coding",
            resolved_session_id=None,
            resolved_served_name=None,
            healthy=False,
        ),
        SimpleNamespace(
            name="[red]x",
            target=None,
            resolved_session_id=None,
            resolved_served_name=None,
            healthy=False,
        ),
    ]


def _patch(monkeypatch):
    monkeypatch.setattr(GatewayService, "alias_view", lambda self: _rows())
    monkeypatch.setattr(cli, "load_settings", lambda: None)

    class _Db:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cli, "_session", lambda: _Db())
    monkeypatch.setattr(GatewayService, "__init__", lambda self, *a, **k: None)


def test_table_whole_ids(monkeypatch) -> None:
    _patch(monkeypatch)
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=120, color_system=None))
    result = CliRunner().invoke(cli.app, ["aliases"])
    assert result.exit_code == 0
    out = buf.getvalue()
    lines = out.splitlines()
    for text in ("long-context", "local-long-context", "qwen3.8-flash-next-strata", "local-coding"):
        assert any(text in ln for ln in lines)
    assert sum("long-context" in ln for ln in lines) == 1
    assert "0123abcd" in out and "0123abcd-" not in out
    assert "[red]x" in out
    assert len(lines) == 3 + 3 + 2  # title, borders/header, rows


def test_json(monkeypatch) -> None:
    _patch(monkeypatch)
    result = CliRunner().invoke(cli.app, ["aliases", "--json"])
    data = json.loads(result.output)
    assert [d["status"] for d in data] == ["healthy", "bound", "unbound"]
    assert data[0] == {
        "name": "long-context",
        "public_id": "local-long-context",
        "target": "qwen3.8-flash-next-strata",
        "session_id": UUID,
        "served_name": "qwen3.8-flash-next-strata",
        "status": "healthy",
    }
    assert data[2]["session_id"] is None
