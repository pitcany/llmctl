"""The HTTP client side of the one-switch-engine door (llm-router /admin/switch)."""

from __future__ import annotations

import httpx
import pytest
import respx

from llmctl.services.lane_switcher import RouterRefused, router_key, switch_via_router

URL = "http://127.0.0.1:4100"


@respx.mock
def test_switch_returns_the_router_answer():
    respx.post(f"{URL}/admin/switch").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "strata",
                "served_model": "flash",
                "upstream": "http://127.0.0.1:8007",
                "switched": True,
                "already_active": False,
                "previous": "qwen3.8-27b",
            },
        )
    )
    result = switch_via_router("flash", url=URL, key="k")
    assert result is not None and result.backend == "strata" and result.previous == "qwen3.8-27b"
    sent = respx.calls.last.request
    assert sent.headers["authorization"] == "Bearer k"
    import json

    assert json.loads(sent.content)["force"] is True


@respx.mock
def test_unknown_backend_is_none_so_the_caller_falls_back():
    respx.post(f"{URL}/admin/switch").mock(
        return_value=httpx.Response(404, json={"error": {"message": "unknown"}})
    )
    assert switch_via_router("typo", url=URL, key="k") is None


@respx.mock
def test_refusal_carries_message_and_retry_after():
    respx.post(f"{URL}/admin/switch").mock(
        return_value=httpx.Response(
            503,
            headers={"Retry-After": "90"},
            json={"error": {"message": "held", "type": "model_switch_failed"}},
        )
    )
    with pytest.raises(RouterRefused, match="held") as exc:
        switch_via_router("flash", url=URL, key="k")
    assert exc.value.retry_after == 90.0


def test_router_key_prefers_env_then_file(tmp_path, monkeypatch):
    f = tmp_path / "env"
    f.write_text("# llm-router\nROUTER_API_KEY=from-file\n")
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    assert router_key(f) == "from-file"
    monkeypatch.setenv("ROUTER_API_KEY", "from-env")
    assert router_key(f) == "from-env"
    assert router_key(tmp_path / "missing") == "from-env"
