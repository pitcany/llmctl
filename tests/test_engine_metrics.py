"""Live engine load: ``llmctl metrics``, and the ``load`` in ``health`` and ``doctor``.

The engines publish what GPU telemetry cannot show -- requests in flight,
KV-cache fill, prefix-cache and speculative hit rates, preemptions. These
tests feed each parser the text its engine produces and check the one
column that matters, ``pressure``, plus the CLI, health and doctor wiring,
all against fake transports: no engine is contacted.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.config import Settings
from llmctl.services import doctor
from llmctl.services.engine_load import EngineLoadRow, collect_engine_loads, endpoints
from llmctl.services.health import HealthService
from llmctl.telemetry import engine_metrics as em
from llmctl.telemetry.engine_metrics import (
    EngineLoad,
    fetch_engine_load,
    parse_llama_cpp,
    parse_ollama_ps,
    parse_prometheus,
    parse_sglang,
    parse_strata,
    parse_vllm,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

VLLM_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen3.8-27b"} 3.0
vllm:num_requests_waiting{engine="0",model_name="qwen3.8-27b"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3.8-27b"} 0.936
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3.8-27b"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3.8-27b"} 440.0
vllm:num_preemptions_total{engine="0",model_name="qwen3.8-27b"} 170.0
vllm:spec_decode_num_drafts_total{engine="0",model_name="qwen3.8-27b"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="qwen3.8-27b"} 300.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="qwen3.8-27b"} 236.0
vllm:prompt_tokens_total{engine="0",model_name="qwen3.8-27b"} 13567646.0
vllm:generation_tokens_total{engine="0",model_name="qwen3.8-27b"} 1894735.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="qwen3.8-27b"} 1100.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="qwen3.8-27b"} 87.0
vllm:request_prompt_tokens_bucket{engine="0",le="100.0",model_name="qwen3.8-27b"} 492.0
"""

LLAMA_TEXT = """\
llamacpp:prompt_tokens_total 5000
llamacpp:tokens_predicted_total 1200
llamacpp:kv_cache_usage_ratio 0.42
llamacpp:requests_processing 1
llamacpp:requests_deferred 2
"""

SGLANG_TEXT = """\
sglang:num_running_reqs{model_name="m"} 2.0
sglang:num_queue_reqs{model_name="m"} 0.0
sglang:token_usage{model_name="m"} 0.88
sglang:cache_hit_rate{model_name="m"} 0.61
sglang:spec_accept_length{model_name="m"} 2.4
sglang:prompt_tokens_total{model_name="m"} 10.0
sglang:generation_tokens_total{model_name="m"} 20.0
"""


class _Resp:
    def __init__(self, body: str) -> None:
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body


def _http(routes: dict[str, str]):
    """A fake ``http_get``: ``routes`` maps a URL suffix to a body; others raise."""

    def get(url: str, timeout: float) -> Any:
        for suffix, body in routes.items():
            if url.endswith(suffix):
                return _Resp(body)
        raise OSError(f"unreachable: {url}")

    return get


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def test_parse_prometheus_handles_labels_comments_and_junk() -> None:
    text = 'a{x="1",y="q,r"} 2\nb 3\n# c 9\nd{z="1"}\nnot-a-number x\n'
    m = parse_prometheus(text)
    assert m["a"] == [({"x": "1", "y": "q,r"}, 2.0)]
    assert m["b"] == [({}, 3.0)]
    assert "c" not in m and "d" not in m


def test_parse_vllm_reads_queue_kv_cache_rates_and_preemptions() -> None:
    load = parse_vllm(VLLM_TEXT, upstream="http://127.0.0.1:8003", version="0.30.0")
    assert load.model == "qwen3.8-27b"
    assert load.version == "0.30.0"
    assert (load.running, load.waiting) == (3, 0)
    assert load.kv_usage_pct == pytest.approx(93.6)
    assert load.prefix_hit_pct == pytest.approx(44.0)
    assert load.preemptions == 170
    assert load.spec_accept_pct == pytest.approx(236 / 300 * 100)
    assert load.spec_tokens_per_step == pytest.approx(3.36)
    assert load.requests == 1187  # summed over finished_reason labels
    assert load.pressure == "high"
    assert "KV 94%" in load.summary() and "preempted 170x" in load.summary()


def test_parse_vllm_without_speculative_decoding_leaves_those_fields_none() -> None:
    text = "\n".join(line for line in VLLM_TEXT.splitlines() if "spec_decode" not in line)
    load = parse_vllm(text)
    assert load.spec_accept_pct is None and load.spec_tokens_per_step is None
    assert "spec accept" not in load.summary()


def test_parse_vllm_pre_v1_gpu_cache_metric_still_reads() -> None:
    load = parse_vllm('vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n')
    assert load.kv_usage_pct == 50.0


def test_parse_llama_cpp() -> None:
    load = parse_llama_cpp(LLAMA_TEXT)
    assert load.runtime == "llama_cpp"
    assert (load.running, load.waiting) == (1, 2)
    assert load.kv_usage_pct == pytest.approx(42.0)
    assert load.generation_tokens == 1200
    assert load.pressure == "saturated"  # deferred requests are a queue


def test_parse_sglang() -> None:
    load = parse_sglang(SGLANG_TEXT)
    assert load.runtime == "sglang" and load.model == "m"
    assert (load.running, load.waiting) == (2, 0)
    assert load.kv_usage_pct == pytest.approx(88.0)
    assert load.prefix_hit_pct == pytest.approx(61.0)
    assert load.spec_tokens_per_step == pytest.approx(2.4)
    assert load.pressure == "high"


def test_parse_strata_live_state_and_speed() -> None:
    payload = {
        "engine": {"model": "qwen3.8-flash-next-strata", "max_context": 262144},
        "live": {"state": "generating", "queued": 1, "tok_s": 93.4},
    }
    load = parse_strata(payload)
    assert load.runtime == "strata"
    assert load.model == "qwen3.8-flash-next-strata"
    assert (load.running, load.waiting) == (1, 1)
    assert load.tokens_per_s == pytest.approx(93.4)
    assert load.pressure == "saturated"
    assert "93 tok/s" in load.summary()

    idle = parse_strata({"engine": {"model": "x"}, "live": {"state": "idle"}})
    assert (idle.running, idle.waiting) == (0, 0) and idle.pressure == "ok"

    batched = parse_strata({"live": {"state": "idle", "running": 2, "waiting": 0}})
    assert batched.running == 2


def test_parse_ollama_ps_lists_resident_models_only() -> None:
    load = parse_ollama_ps({"models": [{"name": "a:latest"}, {"model": "b"}]})
    assert load.model == "a:latest, b"
    assert load.pressure == "unknown"
    assert load.summary() == "loaded, no queue or cache metrics"
    assert parse_ollama_ps({"models": []}).summary() == "no load data"


@pytest.mark.parametrize(
    ("kv", "waiting", "expected"),
    [(10.0, 0, "ok"), (85.0, 0, "high"), (95.0, 0, "saturated"), (10.0, 1, "saturated")],
)
def test_pressure_rule(kv: float, waiting: int, expected: str) -> None:
    assert EngineLoad("u", "vllm", kv_usage_pct=kv, waiting=waiting).pressure == expected


def test_as_dict_carries_pressure() -> None:
    data = EngineLoad("u", "vllm", kv_usage_pct=96.0).as_dict()
    assert data["pressure"] == "saturated" and data["upstream"] == "u"


# --------------------------------------------------------------------------- #
# Fetch dispatch
# --------------------------------------------------------------------------- #


def test_fetch_vllm_reads_metrics_and_version() -> None:
    http = _http({"/metrics": VLLM_TEXT, "/version": '{"version": "0.30.0"}'})
    load = fetch_engine_load("http://127.0.0.1:8003/v1", "vllm", http_get=http)
    assert load is not None and load.version == "0.30.0" and load.running == 3


def test_fetch_detects_engine_by_metric_prefix_when_runtime_unknown() -> None:
    assert (
        fetch_engine_load("http://h:1", None, http_get=_http({"/metrics": LLAMA_TEXT})).runtime
        == "llama_cpp"
    )
    assert (
        fetch_engine_load("http://h:1", None, http_get=_http({"/metrics": SGLANG_TEXT})).runtime
        == "sglang"
    )
    strata = json.dumps({"live": {"state": "idle"}, "engine": {"model": "m"}})
    assert (
        fetch_engine_load("http://h:1", None, http_get=_http({"/metrics": strata})).runtime
        == "strata"
    )


def test_fetch_ollama_uses_api_ps() -> None:
    http = _http({"/api/ps": '{"models": [{"name": "m"}]}'})
    load = fetch_engine_load("http://127.0.0.1:11434", "ollama", http_get=http)
    assert load is not None and load.model == "m"


def test_fetch_unreachable_or_malformed_is_none() -> None:
    assert fetch_engine_load("http://h:1", "vllm", http_get=_http({})) is None
    assert (
        fetch_engine_load("http://h:1", "strata", http_get=_http({"/metrics": "{not json"})) is None
    )
    assert fetch_engine_load("http://h:1", "ollama", http_get=_http({"/api/ps": "{"})) is None


def test_fetch_vllm_without_version_endpoint_still_answers() -> None:
    load = fetch_engine_load("http://h:1", "vllm", http_get=_http({"/metrics": VLLM_TEXT}))
    assert load is not None and load.version is None


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


def _settings(**managed: Any) -> Settings:
    return Settings.model_validate({"managed_units": managed} if managed else {})


def test_endpoints_cover_units_runtimes_and_lanes_without_duplicates() -> None:
    from llmctl.services.lanes import Lane

    settings = _settings(
        vllm_engine={"unit_name": "vllm-engine", "default_port": 8003},
        units={
            "gptoss": {"unit_name": "gpt-oss-120b", "default_port": 8005, "runtime": "llama_cpp"}
        },
    )
    lanes = [
        Lane("gptoss", "gpt-oss-120b", "http://localhost:8005", (), runtime="llama_cpp"),
        Lane("flash", "flash", "http://127.0.0.1:8007", (), runtime="strata"),
    ]
    rows = endpoints(settings, lanes)
    by_name = {r.name: r for r in rows}
    assert [r.name for r in rows] == ["vllm-engine", "gptoss", "ollama", "flash"]
    assert by_name["vllm-engine"].runtime == "vllm" and by_name["vllm-engine"].source == "unit"
    assert by_name["ollama"].upstream.endswith(":11434") and by_name["ollama"].source == "runtime"
    assert by_name["flash"].source == "lane"


def test_collect_probes_every_endpoint_and_keeps_order() -> None:
    settings = _settings(vllm_engine={"unit_name": "vllm-engine", "default_port": 8003})
    http = _http({":8003/metrics": VLLM_TEXT, ":8003/version": '{"version":"0.30.0"}'})
    rows = collect_engine_loads(settings, lanes=[], http_get=http)
    assert [r.name for r in rows] == ["vllm-engine", "ollama"]
    assert rows[0].answered and rows[0].load.version == "0.30.0"
    assert not rows[1].answered and rows[1].as_dict()["load"] is None


# --------------------------------------------------------------------------- #
# CLI, health, doctor
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    monkeypatch.setenv("COLUMNS", "200")  # a wide table: cells are not folded mid-word
    return tmp_path


def _rows() -> list[EngineLoadRow]:
    hot = parse_vllm(VLLM_TEXT, upstream="http://127.0.0.1:8003", version="0.30.0")
    return [
        EngineLoadRow("vllm-engine", "vllm", "http://127.0.0.1:8003", "unit", hot),
        EngineLoadRow("gptoss", "llama_cpp", "http://127.0.0.1:8005", "unit", None),
    ]


def test_cli_metrics_table_and_json(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import llmctl.services.engine_load as engine_load

    monkeypatch.setattr(engine_load, "collect_engine_loads", lambda *_a, **_k: _rows())
    runner = CliRunner()
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0, result.output
    assert "qwen3.8-27b" in result.output and "high" in result.output
    assert "gptoss" not in result.output  # silent endpoints hidden by default

    result = runner.invoke(app, ["metrics", "--all"])
    assert result.exit_code == 0, result.output
    assert "gptoss" in result.output and "not answering" in result.output

    result = runner.invoke(app, ["metrics", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["load"]["pressure"] == "high" and data[1]["load"] is None


def test_health_attaches_load_to_serving_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmctl.services.lanes import Lane

    lane = Lane("flash", "flash-model", "http://127.0.0.1:8007", (), runtime="strata")
    hot = EngineLoad("http://127.0.0.1:8007", "strata", model="flash-model", running=1, waiting=2)
    probes: list[tuple[str, str]] = []

    def load_probe(endpoint: str, runtime: str) -> EngineLoad | None:
        probes.append((endpoint, runtime))
        return hot if endpoint.endswith(":8007") else None

    async def lane_probe(_lane: Lane) -> bool:
        return True

    monkeypatch.setattr("llmctl.services.health.get_gpu_info", lambda: [])
    svc = HealthService(
        Settings.model_validate({}),
        lanes=[lane],
        lane_probe=lane_probe,
        load_probe=load_probe,
        engines=[],
        model_users={},
    )
    data = svc.get_health()
    strata = data["runtimes"]["strata"]
    session = next(s for s in strata["sessions"] if s.get("served_name") == "flash-model")
    assert session["load"]["pressure"] == "saturated"
    assert strata["pressure"] == "saturated"
    assert ("http://127.0.0.1:8007", "strata") in probes


def test_cli_health_prints_load_line(monkeypatch: pytest.MonkeyPatch) -> None:
    health = {
        "state": "ok",
        "safe_mode": True,
        "database_url": "sqlite://",
        "gpu_count": 0,
        "nvml_available": False,
        "session_state_available": True,
        "message": "ok",
        "runtimes": {
            "vllm": {
                "state": "ok",
                "message": "serving",
                "sessions": [
                    {
                        "id": None,
                        "status": "running",
                        "served_name": "qwen3.8-27b",
                        "endpoint_url": "http://127.0.0.1:8003",
                        "kind": None,
                        "managed": False,
                        "load": EngineLoad(
                            "http://127.0.0.1:8003", "vllm", running=3, waiting=0, kv_usage_pct=93.6
                        ).as_dict(),
                    }
                ],
                "serving_count": 1,
                "unmanaged_count": 1,
                "pressure": "high",
            }
        },
    }
    monkeypatch.setattr(HealthService, "get_health", lambda self: health)
    result = CliRunner().invoke(app, ["health"])
    assert result.exit_code == 0, result.output
    assert "KV 94%" in result.output.replace("\n", "")


def test_doctor_engine_load_check_warns_on_pressure_and_preemptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llmctl.services.engine_load as engine_load

    def rows(*_a: Any, **_k: Any) -> list[EngineLoadRow]:
        return [
            EngineLoadRow("a", "vllm", "u", "unit", EngineLoad("u", "vllm", kv_usage_pct=96.0)),
            EngineLoadRow("b", "vllm", "u", "unit", EngineLoad("u", "vllm", kv_usage_pct=90.0)),
            EngineLoadRow(
                "c", "vllm", "u", "unit", EngineLoad("u", "vllm", kv_usage_pct=10.0, preemptions=4)
            ),
            EngineLoadRow("d", "vllm", "u", "unit", EngineLoad("u", "vllm", kv_usage_pct=10.0)),
            EngineLoadRow("e", "llama_cpp", "u", "unit", None),
        ]

    monkeypatch.setattr(engine_load, "collect_engine_loads", rows)
    checks = {c.name: c for c in doctor._check_engine_load(Settings.model_validate({}))}
    assert (
        checks["engine-load:a"].verdict == "warn" and "saturated" in checks["engine-load:a"].detail
    )
    assert "kv-offloading-size" in (checks["engine-load:a"].remediation or "")
    assert checks["engine-load:b"].verdict == "warn" and checks["engine-load:b"].detail.startswith(
        "high"
    )
    assert checks["engine-load:c"].verdict == "warn" and "4 times" in checks["engine-load:c"].detail
    assert checks["engine-load:d"].verdict == "pass"
    assert "engine-load:e" not in checks


def test_doctor_report_includes_engine_load(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llmctl.services.engine_load as engine_load

    monkeypatch.setattr(engine_load, "collect_engine_loads", lambda *_a, **_k: _rows())
    report = doctor.run_doctor()
    names = {c["name"] for c in report["warnings"]}
    assert "engine-load:vllm-engine" in names


def test_default_http_get_is_the_only_network_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> Any:
        seen.append(url)
        return io.BytesIO(b"")

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)
    assert em._default_http_get("http://127.0.0.1:1/metrics", 1.0).read() == b""
    assert seen == ["http://127.0.0.1:1/metrics"]
