from llmctl.hf.hardware import GPU, HardwareInfo
from llmctl.hf.inspect import ModelMetadata
from llmctl.hf.plan import build_plan, render_command


def _hw(*vrams_gib: float) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPU(index=i, name="RTX 5090", vram_mib=int(v * 1024)) for i, v in enumerate(vrams_gib)
        ],
    )


def _meta(model_id: str, **cfg) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        config=cfg,
        tokenizer_config={},
        tags=[],
        pipeline_tag="text-generation",
        gated=False,
        private=False,
    )


def test_plan_single_gpu_small_model():
    meta = _meta(
        "Qwen/Qwen2.5-7B-Instruct",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=32768,
    )
    plan = build_plan(meta, _hw(32.0))
    assert plan.tensor_parallel_size == 1
    assert plan.host == "127.0.0.1"  # safe default
    assert plan.max_model_len <= 32768
    assert plan.vllm_supported is True


def test_plan_70b_dual_gpu_picks_tp2():
    meta = _meta(
        "meta-llama/Llama-3.3-70B-Instruct",
        architectures=["LlamaForCausalLM"],
        max_position_embeddings=131072,
    )
    plan = build_plan(meta, _hw(80.0, 80.0))
    assert plan.tensor_parallel_size == 2
    # 128k context should be capped.
    assert plan.max_model_len <= 32768


def test_plan_derates_max_len_under_weight_pressure():
    # 32B AWQ on a single 32 GiB card — the case that crashed before.
    # Weight pressure ~0.47 (≈15 GiB of weights / 32 GiB), should derate from 32k.
    meta = _meta(
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        architectures=["Qwen2ForCausalLM"],
        quantization_config={"quant_method": "awq"},
        max_position_embeddings=131072,
    )
    plan = build_plan(meta, _hw(32.0))
    assert plan.max_model_len < 32768, "should derate under weight pressure"
    assert plan.max_model_len >= 8192  # but not absurdly small


def test_plan_keeps_32k_for_small_models_with_room():
    meta = _meta(
        "Qwen/Qwen2.5-1.5B-Instruct",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=131072,
    )
    plan = build_plan(meta, _hw(32.0))
    assert plan.max_model_len == 32768


def test_plan_awq_sets_quantization_flag():
    meta = _meta(
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        architectures=["Qwen2ForCausalLM"],
        quantization_config={"quant_method": "awq"},
    )
    plan = build_plan(meta, _hw(32.0))
    assert plan.vllm_quantization_flag == "awq"
    assert plan.quantization == "awq"


def test_plan_flags_gguf_as_unsupported():
    meta = _meta("bartowski/Llama-3.3-70B-Instruct-GGUF", architectures=["LlamaForCausalLM"])
    plan = build_plan(meta, _hw(80.0))
    assert plan.vllm_supported is False
    assert any("GGUF" in w for w in plan.warnings)


def test_plan_warns_on_no_gpu():
    meta = _meta("Qwen/Qwen2.5-7B-Instruct", architectures=["Qwen2ForCausalLM"])
    plan = build_plan(meta, HardwareInfo(gpus=[]))
    assert any("No NVIDIA GPUs" in w for w in plan.warnings)


def test_render_command_basics():
    meta = _meta(
        "Qwen/Qwen2.5-7B-Instruct",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=32768,
    )
    plan = build_plan(meta, _hw(32.0), host="127.0.0.1", port=8001)
    cmd = render_command(plan)
    assert cmd[:2] == ["vllm", "serve"]
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8001" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "--max-model-len" in cmd


def test_render_command_includes_awq_flag():
    meta = _meta(
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        architectures=["Qwen2ForCausalLM"],
        quantization_config={"quant_method": "awq"},
    )
    plan = build_plan(meta, _hw(32.0))
    cmd = render_command(plan)
    assert "--quantization" in cmd
    assert "awq" in cmd


def test_repo_name_alone_never_asks_for_remote_code():
    # Anyone can name a repo "x/deepseek-..."; the name is not evidence of custom code.
    meta = _meta("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", architectures=["LlamaForCausalLM"])
    plan = build_plan(meta, _hw(80.0, 80.0))
    assert plan.trust_remote_code is False
    assert "--trust-remote-code" not in render_command(plan)


def test_custom_code_is_a_warning_not_a_flag_in_the_printed_command():
    # The printed command is meant to be copied and run; it must not run the
    # repo's own Python unless the user adds the flag on purpose.
    meta = _meta(
        "someone/custom-arch-7b",
        architectures=["CustomForCausalLM"],
        auto_map={"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"},
    )
    plan = build_plan(meta, _hw(32.0))
    assert plan.trust_remote_code is True
    assert "--trust-remote-code" not in render_command(plan)
    assert any("--trust-remote-code" in w and "runs" in w for w in plan.warnings)


def test_oom_warning_on_undersized_gpu():
    meta = _meta("meta-llama/Llama-3.3-70B-Instruct", architectures=["LlamaForCausalLM"])
    plan = build_plan(meta, _hw(24.0))  # 24 GiB single GPU
    assert any("OOM risk" in w for w in plan.warnings)
