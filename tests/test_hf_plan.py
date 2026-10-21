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


def test_tp_is_never_a_size_vllm_rejects_on_three_gpus():
    # 64 heads are not divisible by 3; nothing fits, so fall back to the
    # largest valid size, not to the GPU count.
    meta = _meta(
        "Qwen/Qwen2.5-72B-Instruct", architectures=["Qwen2ForCausalLM"], num_attention_heads=64
    )
    plan = build_plan(meta, _hw(40.0, 40.0, 40.0))
    assert plan.tensor_parallel_size == 2


def test_tp_skips_sizes_that_do_not_divide_the_heads():
    # 40 heads: tp=8 is invalid, tp=4 fits weights that need more than 2 cards.
    meta = _meta("org/model-70B", architectures=["LlamaForCausalLM"], num_attention_heads=40)
    plan = build_plan(meta, _hw(*[24.0] * 8))
    assert 40 % plan.tensor_parallel_size == 0


def test_mixtral_counts_all_experts():
    meta = _meta("mistralai/Mixtral-8x7B-Instruct-v0.1", architectures=["MixtralForCausalLM"])
    plan = build_plan(meta, _hw(32.0, 32.0))
    assert plan.param_count_b == 56.0
    assert plan.tensor_parallel_size == 2


def test_tp_respects_kv_heads_as_vllm_partitions_them():
    # 40 heads divide by 4, but 10 KV heads do not (and 4 < 10): vLLM rejects tp=4.
    meta = _meta(
        "org/phi-4-14B",
        architectures=["Phi3ForCausalLM"],
        num_attention_heads=40,
        num_key_value_heads=10,
    )
    plan = build_plan(meta, _hw(12.0, 12.0, 12.0, 12.0))
    assert plan.tensor_parallel_size == 2


def test_tp_may_exceed_kv_heads_when_it_is_a_multiple():
    # 2 KV heads at tp=4 are replicated; vLLM accepts tp % kv == 0.
    meta = _meta(
        "org/model-70B",
        architectures=["LlamaForCausalLM"],
        num_attention_heads=64,
        num_key_value_heads=2,
    )
    plan = build_plan(meta, _hw(24.0, 24.0, 24.0, 24.0))
    assert plan.tensor_parallel_size == 4


def test_expert_suffix_names_active_params_so_size_is_unknown():
    # Llama-4-Scout-17B-16E: 17B active, 109B total; do not size it as 17B.
    meta = _meta(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        architectures=["Llama4ForConditionalGeneration"],
    )
    plan = build_plan(meta, _hw(48.0))
    assert plan.param_count_b is None
    assert any("active" in w for w in plan.warnings)


def test_a_suffix_moe_keeps_the_named_total():
    meta = _meta("Qwen/Qwen3-30B-A3B", architectures=["Qwen3MoeForCausalLM"])
    assert build_plan(meta, _hw(32.0, 32.0)).param_count_b == 30.0


def _ct_cfg(bits: int) -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": bits}}},
        },
    }


def test_compressed_tensors_4bit_is_sized_at_4_bits():
    plan = build_plan(_meta("someone/Qwen3.8-27B-AWQ-INT4", **_ct_cfg(4)), _hw(32.0, 32.0))
    estimate = next(n for n in plan.notes if "Estimated weights" in n)
    assert "~12.6 GiB" in estimate  # 27B x 0.5 bytes, not ~50 GiB at 16-bit
    assert plan.tensor_parallel_size == 1


def test_modelopt_nvfp4_is_sized_at_4_bits():
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }
    plan = build_plan(_meta("someone/Qwen3.8-27B-NVFP4", **cfg), _hw(32.0, 32.0))
    assert any("~12.6 GiB" in n for n in plan.notes)


def test_hub_file_sizes_win_over_the_estimate():
    meta = _meta("someone/Qwen3.8-27B-AWQ-INT4", **_ct_cfg(4))
    meta.weight_bytes = int(19.6 * 1024**3)
    plan = build_plan(meta, _hw(32.0, 32.0))
    assert "Weights 19.6 GiB (the Hub's safetensors file sizes)." in plan.notes
    assert not any("Estimated weights" in n for n in plan.notes)


def test_safetensors_bytes_sums_only_weights():
    from types import SimpleNamespace

    from llmctl.hf.inspect import _safetensors_bytes

    sibs = [
        SimpleNamespace(rfilename="model-00001.safetensors", size=10),
        SimpleNamespace(rfilename="model-00002.safetensors", size=5),
        SimpleNamespace(rfilename="config.json", size=99),
        SimpleNamespace(rfilename="x.safetensors", size=None),
    ]
    assert _safetensors_bytes(sibs) == 15
    assert _safetensors_bytes([]) is None
