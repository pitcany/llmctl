"""`llmctl hf`: CLI wiring, with Hugging Face and the GPUs stubbed out."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmctl import cli_hf
from llmctl.cli import app
from llmctl.hf.hardware import GPU, HardwareInfo
from llmctl.hf.inspect import ModelMetadata
from llmctl.hf.search import Candidate

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HW = HardwareInfo(
    gpus=[
        GPU(index=0, name="RTX 5090", vram_mib=32 * 1024),
        GPU(index=1, name="RTX 5090", vram_mib=32 * 1024),
    ]
)


def _meta(model_id: str = MODEL_ID) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        config={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
        },
        tokenizer_config={"chat_template": "{{ x }}"},
        tags=["text-generation"],
        pipeline_tag="text-generation",
        gated=False,
        private=False,
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_hf, "detect_hardware", lambda: HW)
    monkeypatch.setattr(cli_hf, "inspect_model", lambda model_id: _meta(model_id))
    monkeypatch.setattr(
        cli_hf,
        "search_models",
        lambda query, limit=10: [
            Candidate(
                model_id=MODEL_ID,
                downloads=1234,
                likes=5,
                tags=[],
                pipeline_tag="text-generation",
                score=9.5,
                reasons=["family qwen"],
            ),
            Candidate(
                model_id="org/bracket[INST]-7b",
                downloads=1,
                likes=0,
                tags=[],
                pipeline_tag=None,
                score=1.0,
                reasons=["[/INST] token"],
            ),
        ][:limit],
    )


def test_plan_prints_the_command_and_the_add_step(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hf", "plan", MODEL_ID])
    assert result.exit_code == 0, result.output
    assert f"vllm serve {MODEL_ID}" in result.output
    assert "--tensor-parallel-size 1" in result.output
    assert f"llmctl add {MODEL_ID}" in result.output


def test_plan_flags_override_the_planner(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hf", "plan", MODEL_ID, "--tp", "2", "--max-model-len", "4096"])
    assert result.exit_code == 0, result.output
    assert "--tensor-parallel-size 2" in result.output
    assert "--max-model-len 4096" in result.output


def test_plan_stops_on_an_unreadable_model(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = ModelMetadata(
        model_id="x/y",
        config={},
        tokenizer_config={},
        tags=[],
        pipeline_tag=None,
        gated=False,
        private=False,
        error="Model not found on Hugging Face: x/y",
    )
    monkeypatch.setattr(cli_hf, "inspect_model", lambda model_id: failed)
    result = runner.invoke(app, ["hf", "plan", "x/y"])
    assert result.exit_code == 1
    assert "Model not found" in result.output


def test_search_lists_candidates_and_survives_bracketed_text(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hf", "search", "qwen 7b"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert MODEL_ID in result.output
    assert "[/INST]" in result.output  # printed literally, not parsed as markup


def test_inspect_shows_config_keys(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hf", "inspect", MODEL_ID])
    assert result.exit_code == 0, result.output
    assert "model_type: qwen2" in result.output
    assert "tokenizer.chat_template: present" in result.output


def test_fit_marks_each_candidate(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hf", "fit", "qwen 7b", "-n", "1"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert MODEL_ID in result.output
    assert "yes" in result.output


def test_hf_group_is_listed_in_root_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert " hf " in result.output


def test_fit_says_unknown_or_error_when_it_cannot_judge(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def meta_for(model_id: str) -> ModelMetadata:
        if "bracket" in model_id:
            return ModelMetadata(
                model_id=model_id,
                config={},
                tokenizer_config={},
                tags=[],
                pipeline_tag=None,
                gated=True,
                private=False,
                error="Gated repo. Accept the license.",
            )
        return _meta(model_id)

    monkeypatch.setattr(cli_hf, "inspect_model", meta_for)
    monkeypatch.setattr(cli_hf, "detect_hardware", lambda: HardwareInfo(gpus=[]))
    result = runner.invoke(app, ["hf", "fit", "qwen 7b"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "unknown" in result.output  # no GPUs: no basis for "yes"
    assert "error" in result.output and "Gated repo" in result.output
    assert " yes " not in result.output
