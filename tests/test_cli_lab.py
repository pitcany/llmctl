"""`llmctl recommend` reads the model lab's recommendations.json (read-only)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from llmctl.cli import app

runner = CliRunner()


def _write_lab(tmp_path):
    (tmp_path / "targets.yaml").write_text(
        "targets:\n  gemma4-26b-gpu1:\n    base_url: http://x/v1\n    model: g\n    swap: gemma1\n"
    )
    (tmp_path / "recommendations.json").write_text(json.dumps({
        "generated": "2026-09-29T01:00:00",
        "roles": [
            {"role": "coding", "think": "on", "current": "qwen3.6-35b-a3b-gpu1",
             "best": "gemma4-26b-gpu1", "advice": "keep qwen3.6-35b-a3b-gpu1: not enough evidence"},
            {"role": "rag", "think": "off", "current": "qwen3.8-27b", "best": "qwen3.8-27b",
             "advice": "keep qwen3.8-27b (measured best)"},
        ],
        "redundant": [{"redundant": "a", "given": "b", "think": "off"}],
    }))


def test_recommend_shows_roles_and_swap_command(tmp_path, monkeypatch) -> None:
    _write_lab(tmp_path)
    monkeypatch.setenv("MODELLAB_DIR", str(tmp_path))
    res = runner.invoke(app, ["recommend"], env={"COLUMNS": "250"})
    assert res.exit_code == 0, res.output
    assert "llmctl swap gemma1" in res.output
    assert "coding" in res.output and "rag" in res.output
    assert "redundant?" in res.output


def test_recommend_filters_role_and_rejects_unknown(tmp_path, monkeypatch) -> None:
    _write_lab(tmp_path)
    monkeypatch.setenv("MODELLAB_DIR", str(tmp_path))
    res = runner.invoke(app, ["recommend", "rag"], env={"COLUMNS": "250"})
    assert res.exit_code == 0 and "coding" not in res.output
    assert runner.invoke(app, ["recommend", "nope"]).exit_code == 1


def test_recommend_without_lab_results_explains(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELLAB_DIR", str(tmp_path))
    res = runner.invoke(app, ["recommend"])
    assert res.exit_code == 1
    assert "modellab run" in res.output
