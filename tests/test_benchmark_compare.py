"""Comparison-eligibility gate tests (FIXPLAN 2026-11-03 §1D)."""

from __future__ import annotations

from types import SimpleNamespace

from llmctl.services.benchmark_compare import comparison_verdict


def _result(
    rid: str = "r",
    *,
    success: bool = True,
    mode: str = "live",
    kind: str | None = "chat",
    params: dict | None = None,
    manifest: dict | None = None,
) -> SimpleNamespace:
    merged: dict = {"mode": mode}
    merged.update(params or {})
    return SimpleNamespace(
        id=rid,
        success=success,
        kind=SimpleNamespace(value=kind) if kind else None,
        parameters=merged,
        manifest=manifest or {},
    )


def _workload_manifest(hash_value: str = "h1", gpus: list | None = None) -> dict:
    manifest: dict = {"workload": {"hash": hash_value}}
    if gpus is not None:
        manifest["serving_host"] = {"gpu_identity": gpus}
    return manifest


def test_compatible_model_comparison_is_eligible() -> None:
    """The model is the intended experimental variable: two live runs of the
    same workload on different models stay comparable."""
    manifest = _workload_manifest("h1", gpus=["RTX 5090", "RTX 5090"])
    a = _result("a", manifest=manifest, params={"max_tokens": 256, "total_requests": 4})
    b = _result("b", manifest=manifest, params={"max_tokens": 256, "total_requests": 4})
    verdict = comparison_verdict(b, a)
    assert verdict.eligible, verdict.reasons


def test_failed_runs_are_never_comparable() -> None:
    a = _result("a")
    b = _result("b", success=False)
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("failed" in reason for reason in verdict.reasons)


def test_live_vs_synthetic_is_never_comparable() -> None:
    a = _result("a")
    b = _result("b", mode="mock")
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("execution mode" in reason for reason in verdict.reasons)


def test_different_kinds_are_never_comparable() -> None:
    a = _result("a", kind="chat")
    b = _result("b", kind="health")
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("kinds" in reason for reason in verdict.reasons)


def test_different_workload_hash_is_not_comparable() -> None:
    a = _result("a", manifest=_workload_manifest("h1"))
    b = _result("b", manifest=_workload_manifest("h2"))
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("workload" in reason for reason in verdict.reasons)


def test_different_workload_size_falls_back_to_parameters() -> None:
    """Legacy rows without manifests still gate on recorded request counts."""
    a = _result("a", params={"total_requests": 4})
    b = _result("b", params={"total_requests": 8})
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("8" in reason for reason in verdict.reasons)


def test_different_generation_settings_are_not_comparable() -> None:
    a = _result("a", params={"max_tokens": 256})
    b = _result("b", params={"max_tokens": 128})
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("max_tokens" in reason for reason in verdict.reasons)


def test_mixed_token_count_methods_are_not_comparable() -> None:
    a = _result("a", params={"token_count_source": "server_usage"})
    b = _result("b", params={"token_count_source": "estimate"})
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("token-count" in reason for reason in verdict.reasons)


def test_different_serving_hardware_is_not_comparable() -> None:
    a = _result("a", manifest=_workload_manifest(gpus=["RTX 5090"]))
    b = _result("b", manifest=_workload_manifest(gpus=["RTX 4090"]))
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("hardware" in reason for reason in verdict.reasons)


def test_symmetric_unknowns_do_not_block_legacy_comparison() -> None:
    """Two old rows that recorded nothing beyond mode/success stay comparable:
    absence of evidence is not evidence of a difference."""
    a = _result("a", params={"mode": "live"})
    a.parameters = {"mode": "live"}
    b = _result("b")
    b.parameters = {"mode": "live"}
    assert comparison_verdict(b, a).eligible


def test_asymmetric_manifest_presence_is_explained() -> None:
    a = _result("a", manifest=_workload_manifest("h1"))
    b = _result("b")
    verdict = comparison_verdict(b, a)
    assert not verdict.eligible
    assert any("only one run" in reason for reason in verdict.reasons)
