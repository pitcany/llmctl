"""Tests for the ranking integration in search (without hitting the network)."""

from unittest.mock import MagicMock, patch

from llmctl.hf.parsing import parse_query, score_model


def _fake_model(model_id, downloads=1000, likes=10, tags=None, pipeline_tag="text-generation"):
    m = MagicMock()
    m.id = model_id
    m.downloads = downloads
    m.likes = likes
    m.tags = tags or []
    m.pipeline_tag = pipeline_tag
    return m


def test_search_ranks_better_match_first():
    from llmctl.hf import search as search_mod

    fake_models = [
        _fake_model("randomuser/qwen-base-1.5b", downloads=50_000),
        _fake_model("Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", downloads=20_000, tags=["awq"]),
        _fake_model("microsoft/phi-3-mini", downloads=80_000),
    ]

    # autospec: a keyword the installed huggingface_hub does not accept fails here
    # (a bare MagicMock hid `direction=`, which hub 1.x removed).
    with patch("huggingface_hub.HfApi", autospec=True) as HfApi:
        HfApi.return_value.list_models.return_value = fake_models
        results = search_mod.search_models("qwen coder 32b awq", limit=3)

    assert results[0].model_id == "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"


def test_score_reasons_explain_match():
    parsed = parse_query("llama 70b")
    score, reasons = score_model("meta-llama/Llama-3.3-70B-Instruct", [], parsed, downloads=1000)
    assert score > 0
    assert any("family match" in r for r in reasons)
    assert any("size match" in r for r in reasons)


def test_score_no_signals_returns_low_score():
    parsed = parse_query("llama 70b awq")
    score, reasons = score_model("randomuser/some-random-thing", [], parsed)
    assert score == 0 or score < 1
    assert reasons == [] or all("keyword" in r for r in reasons)
