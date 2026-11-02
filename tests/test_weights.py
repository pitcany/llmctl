"""Weights gone: a preset or lane whose model left the disk is marked and refused.

On a host where starting a model first stops what serves, offering a model
that cannot load is an outage waiting for a keystroke: the Muse preset
(hub snapshot moved away) and two Ollama lanes (``ollama rm``) were all
still listed and runnable on 2026-11-01. ``llmctl ls`` now says ``weights
gone`` and ``llmctl run`` refuses before stopping anything. Every probe here
is faked; nothing touches the network or the real caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llmctl import cli, cli_swap
from llmctl.services import weights
from llmctl.services.lanes import Lane, SwapState, SwapTarget, gather_state
from llmctl.services.weights import (
    Verdict,
    gone_targets,
    hub_cache_roots,
    ollama_installed_models,
    ollama_model_verdict,
    repo_in_hub_cache,
    vllm_model_verdict,
)


class _Resp:
    def __init__(self, body: str) -> None:
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body


def _tags(by_base: dict[str, list[str] | None]):
    """A fake ``http_get``: base URL -> installed names, ``None`` = server down."""
    calls: list[str] = []

    def get(url: str, timeout: float) -> Any:
        calls.append(url)
        for base, names in by_base.items():
            if url.startswith(base):
                if names is None:
                    raise OSError("down")
                return _Resp(json.dumps({"models": [{"name": n} for n in names]}))
        raise OSError("unknown host")

    get.calls = calls  # type: ignore[attr-defined]
    return get


def _snapshot(hub: Path, repo_id: str, *, config: bool = True) -> Path:
    snap = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    if config:
        (snap / "config.json").write_text("{}")
    return snap


@pytest.fixture
def no_env_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


def test_hub_cache_roots_order_and_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    roots = hub_cache_roots([tmp_path / "hub", tmp_path / "extra"])
    assert roots == [
        tmp_path / "hub",
        tmp_path / "extra",
        tmp_path / "hf" / "hub",
        tmp_path / "home" / ".cache" / "huggingface" / "hub",
    ]


def test_repo_in_hub_cache_needs_a_snapshot_with_config(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    _snapshot(hub, "org/present")
    _snapshot(hub, "org/empty", config=False)
    (hub / "models--org--dangling").symlink_to(tmp_path / "nowhere")
    assert repo_in_hub_cache("org/present", [hub])
    assert not repo_in_hub_cache("org/empty", [hub])
    assert not repo_in_hub_cache("org/dangling", [hub])
    assert not repo_in_hub_cache("org/absent", [hub, tmp_path / "missing-root"])


def test_vllm_local_path_verdicts(tmp_path: Path) -> None:
    present = tmp_path / "weights"
    present.mkdir()
    assert vllm_model_verdict(str(present)).present is True
    gone = vllm_model_verdict(str(tmp_path / "deleted"))
    assert gone.gone and "does not exist" in gone.reason
    assert vllm_model_verdict("").present is None
    assert vllm_model_verdict(None).present is None


def test_vllm_tilde_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m").mkdir()
    assert vllm_model_verdict("~/m").present is True
    assert vllm_model_verdict("~/missing").gone


def test_vllm_repo_id_checks_hub_caches(no_env_caches: Path) -> None:
    hub = no_env_caches / "home" / ".cache" / "huggingface" / "hub"
    _snapshot(hub, "Qwen/Qwen3.8-27B-FP8")
    assert vllm_model_verdict("Qwen/Qwen3.8-27B-FP8").present is True
    muse = vllm_model_verdict("meta-models/Muse-Glimmer-30B")
    assert muse.gone and "not in any Hugging Face hub cache" in muse.reason
    assert str(hub) in muse.reason


def test_vllm_repo_id_prefers_the_presets_hf_home(no_env_caches: Path) -> None:
    _snapshot(no_env_caches / "mine" / "hub", "org/model")
    assert vllm_model_verdict("org/model", hf_home=str(no_env_caches / "mine")).present is True
    assert vllm_model_verdict("org/model").gone


def test_vllm_unrecognised_shapes_are_unknown(no_env_caches: Path) -> None:
    assert vllm_model_verdict("https://example.com/x").present is None
    assert vllm_model_verdict("a/b/c").present is None


def test_ollama_installed_models_reads_tags_and_none_when_down() -> None:
    http = _tags({"http://a": ["x:latest", "y:8b"], "http://b": None})
    assert ollama_installed_models("http://a/", http_get=http) == ["x:latest", "y:8b"]
    assert ollama_installed_models("http://b", http_get=http) is None
    assert http.calls[0] == "http://a/api/tags"


def test_ollama_verdict_asks_the_first_server_that_answers() -> None:
    http = _tags({"http://lane": None, "http://host": ["gemma4:26b", "q:latest"]})
    found = ollama_model_verdict("gemma4:26b", ["http://lane", "http://host"], http_get=http)
    assert found.present is True and "http://host" in found.reason
    gone = ollama_model_verdict("qwen3.6:35b", ["http://lane", "http://host"], http_get=http)
    assert gone.gone and "`llmctl add qwen3.6:35b` pulls it" in gone.reason
    assert ollama_model_verdict("q", ["http://host"], http_get=http).present is True  # :latest
    assert ollama_model_verdict("q", ["http://lane"], http_get=http).present is None
    assert ollama_model_verdict(None, ["http://host"], http_get=http).present is None


def test_verdict_gone_only_when_false() -> None:
    assert Verdict(False, "x").gone and not Verdict(None, "x").gone and not Verdict(True, "x").gone


# --------------------------------------------------------------------------- #
# The run list
# --------------------------------------------------------------------------- #


class _Spec:
    def __init__(self, model: str) -> None:
        self.model = model
        self.served_name = "served"


def _catalog() -> list[SwapTarget]:
    gemma = Lane("gemma26b", "gemma4:26b", "http://127.0.0.1:11435", (), runtime="ollama")
    coder = Lane("coder", "qwen3-coder:30b", "http://127.0.0.1:11435", (), runtime="ollama")
    flash = Lane("flash", "flash", "http://127.0.0.1:8007", (), runtime="strata")
    return [
        SwapTarget(key="muse", kind="vllm", model="muse", engine="vLLM", upstream=""),
        SwapTarget(key="qwen", kind="vllm", model="qwen", engine="vLLM", upstream=""),
        SwapTarget(
            key="gemma26b",
            kind="lane",
            model="gemma4:26b",
            engine="Ollama",
            upstream=gemma.upstream,
            lane=gemma,
        ),
        SwapTarget(
            key="coder",
            kind="lane",
            model="qwen3-coder:30b",
            engine="Ollama",
            upstream=coder.upstream,
            lane=coder,
        ),
        SwapTarget(
            key="flash",
            kind="lane",
            model="flash",
            engine="Strata",
            upstream=flash.upstream,
            lane=flash,
        ),
        SwapTarget(key="gguf", kind="session", model="g", engine="llama.cpp", upstream=""),
    ]


def test_gone_targets_judges_vllm_presets_and_ollama_lanes_only(no_env_caches: Path) -> None:
    present = no_env_caches / "present"
    present.mkdir()
    presets = {"muse": _Spec("meta-models/Muse-Glimmer-30B"), "qwen": _Spec(str(present))}
    http = _tags({"http://127.0.0.1:11435": None, "http://127.0.0.1:11434": ["qwen3-coder:30b"]})
    gone = gone_targets(
        _catalog(), presets=presets, ollama_default="http://127.0.0.1:11434", http_get=http
    )
    assert set(gone) == {"muse", "gemma26b"}
    assert "hub cache" in gone["muse"] and "`llmctl add gemma4:26b`" in gone["gemma26b"]
    # The lane server was down, so the host's Ollama decided -- asked once, not per lane.
    assert http.calls.count("http://127.0.0.1:11434/api/tags") == 1
    assert http.calls.count("http://127.0.0.1:11435/api/tags") == 1


def test_gone_targets_without_any_ollama_answer_keeps_lanes_listed(no_env_caches: Path) -> None:
    http = _tags({"http://127.0.0.1:11435": None, "http://127.0.0.1:11434": None})
    assert (
        gone_targets(_catalog(), presets={}, ollama_default="http://127.0.0.1:11434", http_get=http)
        == {}
    )


def test_state_of_and_preview_say_weights_gone() -> None:
    catalog = _catalog()
    state = SwapState(
        catalog=catalog,
        serving=set(),
        active_units=set(),
        vllm_unit="vllm-engine",
        vllm_active=False,
        gone={"muse": "muse is not in any hub cache"},
    )
    muse = catalog[0]
    assert state.state_of(muse) == "weights gone"
    assert state.state_of(catalog[1]) == ""
    preview = state.preview(muse)
    assert preview.blocked == "muse is not in any hub cache"
    assert preview.lines()[0] == "weights gone: muse is not in any hub cache"
    assert "not starting" in preview.lines()[1]
    # Serving beats gone: a model that answers is not gone, whatever the disk says.
    state.serving.add("muse")
    assert state.state_of(muse) == "serving"
    assert state.preview(muse).blocked == ""


def test_gather_state_uses_the_weights_seam_and_survives_its_crash() -> None:
    from types import SimpleNamespace

    settings = SimpleNamespace(
        managed_units=SimpleNamespace(
            vllm_engine=SimpleNamespace(unit_name="vllm-engine", default_port=8003),
            vllm_engine2=None,
        ),
        runtimes={},
    )
    common: dict[str, Any] = dict(
        presets={"muse": _Spec("meta-models/Muse-Glimmer-30B")},
        lanes=[],
        probe=lambda url: [],
        is_active=lambda unit, user: False,
        gpu_sessions=[],
        session_rows=lambda catalog: [],
        live_fit=False,
    )
    state = gather_state(settings, weights_check=lambda cat, specs: {"muse": "gone"}, **common)
    assert state.gone == {"muse": "gone"} and state.state_of(state.catalog[0]) == "weights gone"

    def crash(cat: Any, specs: Any) -> dict[str, str]:
        raise RuntimeError("boom")

    assert gather_state(settings, weights_check=crash, **common).gone == {}


def test_gather_state_default_check_consults_hub_cache_and_ollama(
    no_env_caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        weights,
        "ollama_installed_models",
        lambda base, **_k: ["qwen3-coder:30b"] if base.endswith(":11434") else None,
    )
    settings = SimpleNamespace(
        managed_units=SimpleNamespace(
            vllm_engine=SimpleNamespace(unit_name="vllm-engine", default_port=8003),
            vllm_engine2=None,
        ),
        runtimes={},
    )
    lanes = [
        Lane("gemma26b", "gemma4:26b", "http://127.0.0.1:11435", (), runtime="ollama"),
        Lane("coder", "qwen3-coder:30b", "http://127.0.0.1:11435", (), runtime="ollama"),
    ]
    state = gather_state(
        settings,
        presets={"muse": _Spec("meta-models/Muse-Glimmer-30B")},
        lanes=lanes,
        probe=lambda url: [],
        is_active=lambda unit, user: False,
        gpu_sessions=[],
        session_rows=lambda catalog: [],
        live_fit=False,
    )
    assert set(state.gone) == {"muse", "gemma26b"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _state_with_gone() -> SwapState:
    catalog = _catalog()
    return SwapState(
        catalog=catalog,
        serving=set(),
        active_units=set(),
        vllm_unit="vllm-engine",
        vllm_active=False,
        gone={"muse": "meta-models/Muse-Glimmer-30B is not in any Hugging Face hub cache (/x)"},
    )


def test_cli_ls_marks_weights_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    monkeypatch.setattr(cli, "gather_state", lambda settings: _state_with_gone())
    result = CliRunner().invoke(cli.app, ["ls"])
    assert result.exit_code == 0, result.output
    assert "weights gone" in result.output


def test_cli_run_refuses_a_gone_target_before_anything_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "gather_state", lambda settings: _state_with_gone())
    stopped: list[str] = []
    monkeypatch.setattr(cli_swap, "stop_lanes_for_vllm", lambda *a, **k: stopped.append("vllm"))
    monkeypatch.setattr(cli_swap, "run_lane_swap", lambda *a, **k: stopped.append("lane"))
    monkeypatch.setattr(cli, "start_vllm_tp", lambda *a, **k: stopped.append("start"))
    result = CliRunner().invoke(cli.app, ["run", "muse", "--yes"])
    assert result.exit_code == 1, result.output
    assert "weights gone: meta-models/Muse-Glimmer-30B" in result.output
    assert "Refusing to run muse" in result.output
    assert stopped == []


def test_cli_run_dry_run_also_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "gather_state", lambda settings: _state_with_gone())
    result = CliRunner().invoke(cli.app, ["run", "muse", "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "Refusing to run muse" in result.output
