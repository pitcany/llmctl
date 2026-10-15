from llmctl.hf.parsing import (
    canonicalize_keyword,
    detect_family,
    detect_param_count,
    detect_quantization,
    parse_query,
    score_model,
)


def test_detect_family():
    assert detect_family("llama 3.3 70b") == "llama"
    assert detect_family("Qwen2.5-Coder-32B") == "qwen"
    assert detect_family("DeepSeek-R1-70B") == "deepseek"
    assert detect_family("mistral nemo 12b") == "mistral"
    assert detect_family("totally-unknown-model") is None


def test_detect_param_count():
    assert detect_param_count("llama 3.3 70b") == 70.0
    assert detect_param_count("Qwen2.5-Coder-32B-AWQ") == 32.0
    assert detect_param_count("phi-3.5-mini") is None
    assert detect_param_count("Mistral-7B-Instruct") == 7.0
    assert detect_param_count("model with no size hint") is None


def test_detect_param_count_decimal():
    assert detect_param_count("gemma-2.5b") == 2.5


def test_detect_quantization():
    assert detect_quantization("Qwen2.5-Coder-32B-Instruct-AWQ") == "awq"
    assert detect_quantization("Meta-Llama-3.1-70B-GPTQ-Int4") == "gptq"
    assert detect_quantization("llama-3.3-70b-fp8") == "fp8"
    assert detect_quantization("model-q4_k_m.gguf") == "gguf"
    assert detect_quantization("bare-llama-7b") is None


def test_parse_query_natural():
    p = parse_query("best coding model qwen 32b awq")
    assert p.family == "qwen"
    assert p.param_count_b == 32.0
    assert p.quantization == "awq"
    assert "coding" in p.keywords
    assert "best" not in p.keywords  # stopword


def test_parse_query_drops_comparison_words():
    # Words like "under" / "below" / "max" are not useful as HF search substrings —
    # they used to leak into the search string and zero out HF results.
    p = parse_query("best coding model under 35B")
    assert "under" not in p.keywords
    assert "coding" in p.keywords
    assert p.param_count_b == 35.0


def test_parse_query_detects_size_ceiling():
    for q in ("under 35B", "below 35b", "less than 35B", "max 35B", "up to 35B", "<=35B"):
        p = parse_query(f"coding model {q}")
        assert p.param_count_b == 35.0, q
        assert p.size_is_max is True, q


def test_parse_query_exact_size_is_not_max():
    p = parse_query("llama 3.3 70b")
    assert p.param_count_b == 70.0
    assert p.size_is_max is False


def test_keyword_alias_coding_to_coder():
    assert canonicalize_keyword("coding") == "coder"
    assert canonicalize_keyword("code") == "coder"
    assert canonicalize_keyword("instruct") == "instruct"  # unmapped passes through


def test_score_model_credits_alias_match():
    parsed = parse_query("best coding model under 35B")
    # Qwen2.5-Coder-32B contains "coder" (the alias of "coding"), not "coding".
    coder, reasons_coder = score_model(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        ["text-generation"],
        parsed,
    )
    plain, _ = score_model(
        "Qwen/Qwen2.5-32B-Instruct",
        ["text-generation"],
        parsed,
    )
    assert coder > plain
    assert any("coding" in r for r in reasons_coder)


def test_score_model_size_ceiling_prefers_under():
    parsed = parse_query("coding model under 35B")
    under, _ = score_model("Qwen/Qwen2.5-Coder-32B-Instruct", ["text-generation"], parsed)
    over, _ = score_model("meta-llama/Llama-3.3-70B-Instruct", ["text-generation"], parsed)
    assert under > over


def test_parse_query_model_id():
    p = parse_query("Qwen/Qwen2.5-Coder-32B-Instruct-AWQ")
    assert p.is_model_id() is True
    assert p.family == "qwen"
    assert p.quantization == "awq"


def test_score_model_prefers_match():
    parsed = parse_query("qwen coder 32b awq")
    high, _ = score_model(
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        ["awq", "text-generation"],
        parsed,
        downloads=10000,
        likes=100,
    )
    low, _ = score_model(
        "Qwen/Qwen2.5-1.5B",
        ["text-generation"],
        parsed,
        downloads=10000,
        likes=100,
    )
    assert high > low


def test_score_model_penalizes_gguf_when_unrequested():
    parsed = parse_query("llama 3.3 70b")
    plain, _ = score_model("meta-llama/Llama-3.3-70B-Instruct", [], parsed)
    gguf, _ = score_model("bartowski/Llama-3.3-70B-Instruct-GGUF", [], parsed)
    assert plain > gguf


def test_score_model_size_match_dominates():
    parsed = parse_query("llama 70b")
    big, _ = score_model("meta-llama/Llama-3.3-70B-Instruct", [], parsed)
    small, _ = score_model("meta-llama/Llama-3.2-1B-Instruct", [], parsed)
    assert big > small
