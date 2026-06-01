"""llmctl preset schema v1."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llmctl.presets.paths import validate_alias

SCHEMA_VERSION = 1

CANONICAL_QUANTIZATIONS = frozenset(
    {"awq", "gptq", "fp8", "bnb", "none", "gguf", "compressed-tensors"}
)


class PresetSchemaError(ValueError):
    """Raised when a preset YAML does not satisfy the llmctl schema."""


class Model(BaseModel):
    """One configured model preset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str
    served_name: str
    model_id: str
    quantization: str
    vllm_quantization_flag: str
    tensor_parallel_size: int
    max_model_len: int

    family: str | None = None
    param_count_b: float | None = None
    architectures: tuple[str, ...] = Field(default_factory=tuple)

    max_num_seqs: int | None = Field(
        default=None,
        description="Maximum concurrent sequences; None falls back to renderer defaults.",
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        description="vLLM GPU memory fraction; None falls back to renderer defaults.",
    )
    kv_cache_dtype: str | None = Field(
        default=None,
        description="vLLM KV cache dtype; None falls back to renderer defaults.",
    )
    dtype: str | None = Field(
        default=None,
        description="Model dtype override; None falls back to renderer defaults.",
    )
    trust_remote_code: bool | None = Field(
        default=None,
        description="Whether to trust remote model code; None falls back to renderer defaults.",
    )
    host: str | None = Field(
        default=None,
        description="Bind host; None falls back to renderer defaults.",
    )
    port: int | None = Field(
        default=None,
        description="Preset port template; None falls back to renderer defaults.",
    )

    tool_parser: str | None = None
    reasoning_parser: str | None = None

    tq: bool | None = Field(
        default=None,
        description="TurboQuant preference; None falls back to renderer defaults.",
    )
    shortcuts: tuple[str, ...] = Field(default_factory=tuple)

    schema_version: int = SCHEMA_VERSION

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise PresetSchemaError(str(exc)) from exc

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Model:
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as exc:
            raise PresetSchemaError(str(exc)) from exc

    @field_validator("alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        try:
            validate_alias(value)
        except ValueError as exc:
            raise PresetSchemaError(str(exc)) from exc
        return value

    @field_validator("shortcuts")
    @classmethod
    def _validate_shortcuts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for shortcut in value:
            try:
                validate_alias(shortcut)
            except ValueError as exc:
                raise PresetSchemaError(str(exc)) from exc
        return value

    @field_validator("quantization")
    @classmethod
    def _validate_quantization(cls, value: str) -> str:
        if value not in CANONICAL_QUANTIZATIONS:
            raise PresetSchemaError(
                f"quantization must be one of {sorted(CANONICAL_QUANTIZATIONS)}; "
                f"got {value!r}"
            )
        return value

    @field_validator("tensor_parallel_size")
    @classmethod
    def _validate_tensor_parallel_size(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise PresetSchemaError(
                f"tensor_parallel_size must be in 1..8; got {value}"
            )
        return value

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _validate_gpu_memory_utilization(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 < value <= 1.0:
            raise PresetSchemaError(
                f"gpu_memory_utilization must be in (0, 1]; got {value}"
            )
        return value

    @field_validator("max_model_len")
    @classmethod
    def _validate_max_model_len(cls, value: int) -> int:
        if value < 1:
            raise PresetSchemaError(f"max_model_len must be >= 1; got {value}")
        return value

    @field_validator("max_num_seqs")
    @classmethod
    def _validate_max_num_seqs(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value < 1:
            raise PresetSchemaError(f"max_num_seqs must be >= 1; got {value}")
        return value

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value < 1 or value > 65535:
            raise PresetSchemaError(f"port must be in 1..65535; got {value}")
        return value

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise PresetSchemaError(
                f"unsupported schema_version {value}; this build only reads v{SCHEMA_VERSION}"
            )
        return value
