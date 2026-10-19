"""app.js T4 actions: run/stop confirmation, job panel, reattach (static scans)."""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib.resources import files

import pytest


def _js() -> str:
    return (files("llmctl.api") / "static" / "app.js").read_text(encoding="utf-8")


def test_app_js_has_action_literals() -> None:
    js = _js()
    for needle in (
        "/swap/run",
        "/swap/stop",
        "/swap/jobs/",
        "dry_run",
        "already_serving",
        "Already serving",
        "Ready",
        "Failed (exit",
        "slice(-200)",
        "frees cards",
        "stopped ",
    ):
        assert needle in js, needle


def test_app_js_line_cap() -> None:
    assert len(_js().splitlines()) <= 450


def test_app_js_polls_jobs_every_second() -> None:
    assert re.search(r"JOB_POLL_MS\s*=\s*1000", _js())


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses() -> None:
    path = files("llmctl.api") / "static" / "app.js"
    res = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
