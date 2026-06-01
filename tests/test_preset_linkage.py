"""Tests for preset ↔ Model registry linkage resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

from llmctl.db import RuntimeName
from llmctl.presets import Model as PresetSchemaModel
from llmctl.schemas import Model as RegistryModel
from llmctl.services.preset_loader import (
    load_preset_views,
    preset_count_by_model,
    resolve_preset_link,
)


def _preset(**overrides) -> PresetSchemaModel:
    base = {
        "alias": "x",
        "served_name": "x",
        "model_id": "org/x",
        "quantization": "awq",
        "vllm_quantization_flag": "awq_marlin",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
    }
    base.update(overrides)
    return PresetSchemaModel(**base)


def _model(id: str, name: str, *, source: str | None = None) -> RegistryModel:
    return RegistryModel(
        id=id,
        name=name,
        runtime=RuntimeName.VLLM,
        source=source,
    )


def _write_preset(directory: Path, alias: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{alias}.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


def test_explicit_model_ref_resolves() -> None:
    """model_ref pointing at a known registry id yields explicit linkage."""
    preset = _preset(model_ref="model-123")
    models = [_model(id="model-123", name="llama-3.3-70b")]
    match, state = resolve_preset_link(preset, models)
    assert state == "explicit"
    assert match is not None and match.id == "model-123"


def test_explicit_model_ref_missing_is_flagged() -> None:
    """A model_ref that doesn't match anything is 'missing', not falling back."""
    preset = _preset(model_ref="model-does-not-exist")
    models = [_model(id="other", name="other", source="org/x")]
    match, state = resolve_preset_link(preset, models)
    assert match is None
    assert state == "missing"


def test_auto_match_by_model_id_to_source() -> None:
    """No model_ref: HF-style model_id matches Model.source."""
    preset = _preset(model_id="casperhansen/llama-3.3-70b-instruct-awq")
    models = [
        _model(
            id="m-1",
            name="llama",
            source="casperhansen/llama-3.3-70b-instruct-awq",
        )
    ]
    match, state = resolve_preset_link(preset, models)
    assert state == "auto"
    assert match is not None and match.id == "m-1"


def test_auto_match_by_served_name_to_source() -> None:
    """vLLM-discovered Model.source carries the served_name string."""
    preset = _preset(served_name="llama-3.3-70b", model_id="org/something-else")
    models = [_model(id="m-2", name="llama-3.3-70b", source="llama-3.3-70b")]
    match, state = resolve_preset_link(preset, models)
    assert state == "auto"
    assert match is not None and match.id == "m-2"


def test_unlinked_when_no_match() -> None:
    """Nothing matches → unlinked, no exception."""
    preset = _preset(served_name="totally-unique", model_id="org/totally-unique")
    models = [_model(id="m", name="other", source="elsewhere")]
    match, state = resolve_preset_link(preset, models)
    assert match is None
    assert state == "unlinked"


def test_load_preset_views_populates_linkage(
    tmp_path: Path,
) -> None:
    """Passing models=... to load_preset_views fills linkage fields."""
    _write_preset(
        tmp_path,
        "x",
        """
        alias: x
        served_name: x
        model_id: org/x
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    models = [_model(id="m-id", name="x-model", source="org/x")]
    views = load_preset_views(config_dir=tmp_path, models=models)
    assert len(views) == 1
    view = views[0]
    assert view.linkage_state == "auto"
    assert view.linked_model_id == "m-id"
    assert view.linked_model_name == "x-model"


def test_load_preset_views_skips_linkage_when_models_omitted(
    tmp_path: Path,
) -> None:
    """CLI callers that don't pass models keep linkage_state=None."""
    _write_preset(
        tmp_path,
        "x",
        """
        alias: x
        served_name: x
        model_id: org/x
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    views = load_preset_views(config_dir=tmp_path)
    assert views[0].linkage_state is None
    assert views[0].linked_model_id is None


def test_preset_count_by_model_aggregates_explicit_and_auto(
    tmp_path: Path,
) -> None:
    """Both explicit refs and auto matches contribute to the per-model count."""
    _write_preset(
        tmp_path,
        "explicit-link",
        """
        alias: explicit-link
        served_name: explicit-link
        model_id: org/x
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        model_ref: m-1
        """,
    )
    _write_preset(
        tmp_path,
        "auto-link",
        """
        alias: auto-link
        served_name: auto-link
        model_id: org/x
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    _write_preset(
        tmp_path,
        "orphan",
        """
        alias: orphan
        served_name: orphan
        model_id: org/orphan
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    models = [_model(id="m-1", name="x-model", source="org/x")]
    counts = preset_count_by_model(config_dir=tmp_path, models=models)
    # Both explicit-link (via model_ref=m-1) and auto-link (via
    # model_id->source) resolve to m-1; orphan does not resolve.
    assert counts == {"m-1": 2}
