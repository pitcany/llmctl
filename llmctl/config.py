"""Configuration loading for LLM Mission Control.

The loader intentionally avoids hardcoded machine paths. Defaults are derived
from platform-specific user config/data directories and can be overridden via
YAML or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_dir, user_data_dir, user_log_dir
from pydantic import BaseModel, Field

APP_NAME = "llmctl"


class AppSettings(BaseModel):
    """Application-level settings."""

    name: str = APP_NAME
    log_level: str = "INFO"
    safe_mode: bool = True


class DatabaseSettings(BaseModel):
    """Database settings."""

    url: str | None = None


class APISettings(BaseModel):
    """API server settings."""

    host: str = "127.0.0.1"
    port: int = 8088
    cors_origins: list[str] = Field(default_factory=list)


class TelemetrySettings(BaseModel):
    """Telemetry polling settings."""

    gpu_poll_interval_seconds: float = 2.0
    process_poll_interval_seconds: float = 2.0


class SchedulerSettings(BaseModel):
    """Scheduler safety and placement settings."""

    dry_run_default: bool = True
    require_confirmation_for_start: bool = True
    require_confirmation_for_stop: bool = True
    require_confirmation_for_delete: bool = True
    default_host: str = "127.0.0.1"
    port_ranges: dict[str, list[int]] = Field(default_factory=dict)
    gpu_policy: str = "most-free"
    safety_margin_gb: float = 1.0
    allow_public_bind: bool = False
    require_auth_token: bool = False
    logs_dir: str | None = None


class RuntimeConfig(BaseModel):
    """Per-runtime connection and launch configuration.

    ``endpoint`` is used by HTTP server runtimes (Ollama, LM Studio). ``binary``,
    ``host`` and ``port_range`` are used by process-launch runtimes (vLLM,
    llama.cpp, python scripts).
    """

    endpoint: str | None = None
    binary: str | None = None
    host: str = "127.0.0.1"
    port_range: list[int] = Field(default_factory=lambda: [8000, 8099])
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ManagedUnitConfig(BaseModel):
    """Config for an externally-installed systemd unit that llmctl manages.

    Every field has a default so a fresh install works out of the box,
    but each can be overridden in ``settings.yaml`` so llmctl can run on
    a host with a different layout, different unit names, or a different
    launcher script. This keeps llmctl independent of any specific
    repo's path conventions (``~/AI/services/...`` is just one option).
    """

    enabled: bool = False
    unit_name: str = "vllm-tp"
    env_file_path: Path | None = None
    # The marker string the legacy-unit guard looks for in `systemctl cat`
    # output. If the unit's ExecStart doesn't contain this substring, the
    # guard refuses to write to env_file_path. Set to None to disable the
    # guard for unusual launcher setups.
    launcher_marker: str | None = "vllm-launcher.sh"
    # Default port the readiness poll hits. Overridden per-spec at launch.
    default_port: int = 8003

    def resolve_env_file(self) -> Path:
        """Return the env file path, with sensible fallback.

        Order:
        1. Explicit ``env_file_path`` from config
        2. ``$LLMCTL_VLLM_ENV_FILE`` environment override
        3. ``$AI_HOME/services/<unit_name>.env`` if AI_HOME is set
        4. ``~/AI/services/<unit_name>.env`` (matches yannik-desktop, the
           original cutover target — won't surprise existing installs but
           a clean install on another host will likely override via config)
        """
        if self.env_file_path is not None:
            return Path(self.env_file_path).expanduser()
        env_override = os.environ.get("LLMCTL_VLLM_ENV_FILE")
        if env_override:
            return Path(env_override).expanduser()
        ai_home = os.environ.get("AI_HOME")
        if ai_home:
            return Path(ai_home).expanduser() / "services" / f"{self.unit_name}.env"
        return Path.home() / "AI" / "services" / f"{self.unit_name}.env"


class VLLMDefaultsConfig(BaseModel):
    """Cross-preset launch defaults for the vLLM runtime.

    The :class:`~llm_models_config.schema.Model` from
    ``~/.config/llm-models/<alias>.yaml`` carries per-preset values
    (served name, model id, quantization, max sequences, etc.); this
    block carries the values that don't vary per preset (GPU layout,
    port, batching, NCCL flags).

    Default values match the production posture on yannik-desktop so
    a fresh install behaves the same as the existing setup. Override
    via ``settings.yaml`` to relocate the production endpoint, pin
    different GPUs, or tune batching.
    """

    gpus: str = "0,1"
    tensor_parallel: int = 2
    port: int = 8003
    host: str = "0.0.0.0"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    prefix_cache: bool = True
    chunked_prefill: bool = True
    nccl_p2p_disable: bool = False
    max_num_seqs: int | None = None
    max_batched_tokens: int | None = None


class VLLMConfig(BaseModel):
    """Top-level vLLM-specific settings."""

    defaults: VLLMDefaultsConfig = Field(default_factory=VLLMDefaultsConfig)


class SlotConfig(BaseModel):
    """One per-GPU TP=1 slot the slot subcommand recognises.

    A slot is the *stable* serving identity (e.g. ``coder`` on GPU 0
    port 8001). The user-chosen preset only contributes
    model/quant/ctx; ``gpu``, ``port``, and the slot ``name`` are
    pinned here so downstream client configs never change when the
    underlying model swaps.
    """

    enabled: bool = True
    gpu: str = Field(description="CUDA_VISIBLE_DEVICES value, usually one index.")
    port: int = Field(ge=1, le=65535)
    unit_name: str = Field(description="Bare systemd unit name (no .service).")
    env_file_path: Path | None = None

    def resolve_env_file(self, slot_name: str) -> Path:
        """Same resolution chain as :meth:`ManagedUnitConfig.resolve_env_file`,
        scoped to this slot's unit name."""
        if self.env_file_path is not None:
            return Path(self.env_file_path).expanduser()
        env_override = os.environ.get(f"LLMCTL_SLOT_{slot_name.upper()}_ENV_FILE")
        if env_override:
            return Path(env_override).expanduser()
        ai_home = os.environ.get("AI_HOME")
        if ai_home:
            return Path(ai_home).expanduser() / "services" / f"{self.unit_name}.env"
        return Path.home() / "AI" / "services" / f"{self.unit_name}.env"


class SlotsConfig(BaseModel):
    """Container for all per-GPU slots. Extendable beyond coder/reasoner.

    The default coder/reasoner posture matches yannik-desktop so the
    cutover doesn't require config changes on the existing host. Add a
    third slot (e.g. ``vision``) by setting ``slots.vision`` in
    ``settings.yaml``.
    """

    coder: SlotConfig = Field(
        default_factory=lambda: SlotConfig(
            enabled=True, gpu="0", port=8001, unit_name="vllm-coder"
        )
    )
    reasoner: SlotConfig = Field(
        default_factory=lambda: SlotConfig(
            enabled=True, gpu="1", port=8002, unit_name="vllm-reasoner"
        )
    )

    def get(self, slot_name: str) -> SlotConfig | None:
        """Return the named slot or ``None`` if undefined.

        Used by the slot CLI so unknown slot names produce a clean
        error with the available list rather than an AttributeError.
        """
        return getattr(self, slot_name, None) if hasattr(self, slot_name) else None


class FleetUnitsConfig(BaseModel):
    """Bare unit names for the fleet preflight orchestrator.

    The preflight ("stop competing units before starting the TP unit")
    needs to know which units claim the same GPUs as the target. These
    defaults match the ``NOPASSWD`` sudoers entries on yannik-desktop;
    re-target via ``settings.yaml`` for other hosts.
    """

    tp: str = "vllm-tp"
    coder: str = "vllm-coder"
    reasoner: str = "vllm-reasoner"
    ollama: str = "ollama"
    fleet_target: str = "agents.target"


class ManagedUnitsConfig(BaseModel):
    """Container for all managed systemd units llmctl knows about.

    Keyed by logical role rather than unit name so the same role can be
    re-targeted to a different unit on another host without touching code.
    """

    vllm_tp: ManagedUnitConfig = Field(
        default_factory=lambda: ManagedUnitConfig(
            enabled=False, unit_name="vllm-tp", default_port=8003
        )
    )
    # Slot units (per-GPU TP=1) are wired in Phase 4. Defaulted here so
    # config files written today don't need a migration when slots ship.
    vllm_coder: ManagedUnitConfig = Field(
        default_factory=lambda: ManagedUnitConfig(
            enabled=False, unit_name="vllm-coder", default_port=8001
        )
    )
    vllm_reasoner: ManagedUnitConfig = Field(
        default_factory=lambda: ManagedUnitConfig(
            enabled=False, unit_name="vllm-reasoner", default_port=8002
        )
    )
    slots: SlotsConfig = Field(default_factory=SlotsConfig)
    fleet: FleetUnitsConfig = Field(default_factory=FleetUnitsConfig)


def default_runtime_configs() -> dict[str, RuntimeConfig]:
    """Return built-in default configuration for every supported runtime."""
    return {
        "ollama": RuntimeConfig(endpoint="http://127.0.0.1:11434"),
        "lmstudio": RuntimeConfig(endpoint="http://127.0.0.1:1234"),
        "vllm": RuntimeConfig(binary="vllm", host="127.0.0.1", port_range=[8000, 8099]),
        "llama_cpp": RuntimeConfig(
            binary="llama-server", host="127.0.0.1", port_range=[8100, 8199]
        ),
        "python_script": RuntimeConfig(host="127.0.0.1", port_range=[8200, 8299]),
    }


class PathSettings(BaseModel):
    """Filesystem paths used by the application."""

    config_dir: Path | None = None
    data_dir: Path | None = None
    logs_dir: Path | None = None


class Settings(BaseModel):
    """Typed settings object for all subsystems."""

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    runtimes: dict[str, RuntimeConfig] = Field(default_factory=dict)
    managed_units: ManagedUnitsConfig = Field(default_factory=ManagedUnitsConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)

    def runtime_config(self, runtime: str) -> RuntimeConfig:
        """Return effective runtime config, merging defaults with YAML overrides."""
        defaults = default_runtime_configs()
        base = defaults.get(runtime, RuntimeConfig())
        override = self.runtimes.get(runtime)
        if override is None:
            return base
        merged = base.model_dump()
        merged.update({k: v for k, v in override.model_dump().items() if v not in (None, [], {})})
        return RuntimeConfig.model_validate(merged)

    @property
    def config_dir(self) -> Path:
        """Return the effective config directory."""
        return Path(
            os.getenv("LLMCTL_CONFIG_DIR")
            or self.paths.config_dir
            or user_config_dir(APP_NAME)
        )

    @property
    def data_dir(self) -> Path:
        """Return the effective data directory."""
        return Path(self.paths.data_dir or user_data_dir(APP_NAME))

    @property
    def logs_dir(self) -> Path:
        """Return the effective log directory."""
        return Path(self.paths.logs_dir or user_log_dir(APP_NAME))

    @property
    def database_url(self) -> str:
        """Return effective database URL, preferring environment overrides."""
        explicit = os.getenv("LLMCTL_DB_URL") or self.database.url
        if explicit:
            return explicit
        return f"sqlite:///{self.data_dir / 'llmctl.sqlite3'}"

    @property
    def log_level(self) -> str:
        """Return effective log level."""
        return os.getenv("LLMCTL_LOG_LEVEL") or self.app.log_level


class ModelRoot(BaseModel):
    """A model discovery root from model_dirs.yaml."""

    name: str
    enabled: bool = True
    env_var: str | None = None
    relative_path: str = "."
    runtimes: list[str] = Field(default_factory=list)

    def resolve_path(self) -> Path | None:
        """Resolve root path from environment plus relative path.

        Returns None when the configured environment variable is unavailable.
        """
        if self.env_var:
            base = os.getenv(self.env_var)
            if not base:
                return None
            return Path(base).expanduser() / self.relative_path
        return Path(self.relative_path).expanduser()


class ModelDirsConfig(BaseModel):
    """Model directory scan configuration."""

    model_roots: list[ModelRoot] = Field(default_factory=list)
    scan: dict[str, Any] = Field(default_factory=dict)


class ProfilesConfig(BaseModel):
    """Runtime profiles configuration."""

    profiles: list[dict[str, Any]] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, returning an empty dict if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def load_settings(path: Path | None = None) -> Settings:
    """Load application settings from YAML and environment variables."""
    config_dir = Path(os.getenv("LLMCTL_CONFIG_DIR") or path or user_config_dir(APP_NAME))
    settings_file = config_dir / "settings.yaml" if config_dir.is_dir() else config_dir
    data = _read_yaml(settings_file)
    return Settings.model_validate(data)


def load_model_dirs(path: Path | None = None) -> ModelDirsConfig:
    """Load model discovery config."""
    config_dir = Path(os.getenv("LLMCTL_CONFIG_DIR") or path or user_config_dir(APP_NAME))
    config_file = config_dir / "model_dirs.yaml" if config_dir.is_dir() else config_dir
    return ModelDirsConfig.model_validate(_read_yaml(config_file))


def load_profiles(path: Path | None = None) -> ProfilesConfig:
    """Load runtime profile config."""
    config_dir = Path(os.getenv("LLMCTL_CONFIG_DIR") or path or user_config_dir(APP_NAME))
    config_file = config_dir / "profiles.yaml" if config_dir.is_dir() else config_dir
    return ProfilesConfig.model_validate(_read_yaml(config_file))
