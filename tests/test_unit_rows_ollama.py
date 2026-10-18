"""Ollama units report what is loaded (/api/ps), not what is installed."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path

from llmctl.config import ManagedUnitConfig, Settings
from llmctl.services.units import unit_rows


def _settings(tmp_path: Path, runtime: str) -> Settings:
    s = Settings()
    s.managed_units.vllm_engine = ManagedUnitConfig(
        unit_name="u",
        default_port=8003,
        env_file_path=tmp_path / "u.env",
        runtime=runtime,
    )
    return s


def _fake(ps=None, *, fail=False):
    urls: list[str] = []

    def get(url: str, timeout: float):
        urls.append(url)
        if fail:
            raise urllib.error.URLError("refused")
        if url.endswith("/api/ps"):
            return BytesIO(json.dumps(ps).encode())
        return BytesIO(json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode())

    return get, urls


def _row(tmp_path, runtime, get):
    rows = unit_rows(_settings(tmp_path, runtime), http_get=get)
    return next(r for r in rows if r.role == "vllm-engine")


def test_ollama_reports_loaded_once(tmp_path) -> None:
    get, urls = _fake({"models": [{"name": "m:1", "model": "m:1"}]})
    assert _row(tmp_path, "ollama", get).served == ["m:1"]
    assert urls[0].endswith("/api/ps")


def test_ollama_failure_is_none(tmp_path) -> None:
    get, _ = _fake(fail=True)
    assert _row(tmp_path, "ollama", get).served is None


def test_ollama_nothing_loaded_is_empty(tmp_path) -> None:
    get, _ = _fake({"models": []})
    assert _row(tmp_path, "ollama", get).served == []


def test_vllm_still_uses_v1_models(tmp_path) -> None:
    get, urls = _fake({"models": []})
    assert _row(tmp_path, "vllm", get).served == ["a", "b"]
    assert urls[0].endswith("/v1/models")
