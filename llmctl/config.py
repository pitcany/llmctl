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

APP_NAME = "llm-mission-control"


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
