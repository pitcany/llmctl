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
