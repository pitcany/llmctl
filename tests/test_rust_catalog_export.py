"""llmctl-next's reverse catalog export goes back into Python (Rust M5c item 3).

``tests/fixtures/rust_parity/catalog_export.json`` is what llmctl-next's
``catalog export`` writes after importing ``catalog_bundle.json`` (pinned by
``rust/crates/llmctl-cli/tests/catalog_e2e.rs``). Imported into a fresh DB
with ``import-registry``, it must give the same rows as importing the
original bundle: catalog edits made during the pilot survive a reversal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.db import init_db

FIX = Path(__file__).resolve().parent / "fixtures" / "rust_parity"
runner = CliRunner()

#: What import-registry carries over (ids, timestamps, status and size are
#: assigned by the importing DB).
CARRIED = (
    "name", "runtime", "source", "path", "host", "format", "quantization",
    "estimated_vram_gb", "max_context", "parameter_count", "notes",
    "default_profile_id", "tags", "metadata", "active",
)


def _roundtrip(bundle: Path, tmp: Path, monkeypatch: pytest.MonkeyPatch, tag: str) -> dict:
    url = f"sqlite:///{tmp / f'{tag}.sqlite3'}"
    monkeypatch.setenv("LLMCTL_DB_URL", url)
    init_db(url)
    res = runner.invoke(app, ["import-registry", str(bundle)])
    assert res.exit_code == 0, res.output
    out = tmp / f"{tag}-export.json"
    res = runner.invoke(app, ["export-registry", str(out)])
    assert res.exit_code == 0, res.output
    return json.loads(out.read_text())


def _rows(bundle: dict) -> list[str]:
    return sorted(json.dumps({k: m[k] for k in CARRIED}, sort_keys=True) for m in bundle["models"])


def _profiles(bundle: dict) -> dict[str, dict]:
    return {p["name"]: {k: v for k, v in p.items() if k != "id"} for p in bundle["profiles"]}


def test_export_reimports_to_the_same_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _roundtrip(FIX / "catalog_bundle.json", tmp_path, monkeypatch, "original")
    reversed_ = _roundtrip(FIX / "catalog_export.json", tmp_path, monkeypatch, "reversed")
    assert len(original["models"]) == 12  # import-registry keeps the in-bundle duplicate too
    assert _rows(reversed_) == _rows(original)
    assert _profiles(reversed_) == _profiles(original)


def test_export_keeps_every_bundle_row_verbatim() -> None:
    bundle = json.loads((FIX / "catalog_bundle.json").read_text())
    export = json.loads((FIX / "catalog_export.json").read_text())
    by_id = {m["id"]: m for m in export["models"]}
    assert len(by_id) == len(bundle["models"])
    for row in bundle["models"]:
        assert by_id[row["id"]] == row, row["name"]
    assert export["profiles"] == bundle["profiles"]


def test_a_reversal_into_the_db_it_came_from_needs_update_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1/R2 import the agent's export into the DB that made the bundle: every row exists.

    Found by the M5f rehearsal: without ``--update-models`` an edit that the agent made
    to an existing row did not reach Python.
    """
    live = _roundtrip(FIX / "catalog_bundle.json", tmp_path, monkeypatch, "live")
    row = live["models"][0]
    changed = {**row, "notes": "edited by the agent", "active": False}
    edited = {**live, "models": [changed, *live["models"][1:]]}
    bundle = tmp_path / "agent-export.json"
    bundle.write_text(json.dumps(edited))

    def now() -> dict:
        out = tmp_path / "now.json"
        assert runner.invoke(app, ["export-registry", str(out)]).exit_code == 0
        return json.loads(out.read_text())

    assert runner.invoke(app, ["import-registry", str(bundle)]).exit_code == 0
    assert _rows(now()) == _rows(live)  # the default still skips existing rows

    res = runner.invoke(app, ["import-registry", str(bundle), "--update-models"])
    assert res.exit_code == 0 and "updated 1 models" in res.output, res.output
    assert _rows(now()) == _rows(edited)
    res = runner.invoke(app, ["import-registry", str(bundle), "--update-models"])
    # A second pass changes nothing.
    assert "Imported 0 models, updated 0 models" in res.output, res.output

