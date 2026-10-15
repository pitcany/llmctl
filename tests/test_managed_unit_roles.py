"""Generic managed-unit roles: ManagedUnitsConfig.units + roles().

``managed_units`` used to hardcode exactly two fields (``vllm_engine`` and
``fleet``), so a healthy non-vLLM server (e.g. a llama.cpp unit) was
structurally invisible to ``llmctl status``. The generic ``units`` mapping
adds arbitrary roles keyed by name; these tests pin the compatibility
contract: existing ``vllm_engine``/``fleet`` consumers see identical behavior,
and a config without ``units`` is a no-op migration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig, Settings


def test_default_roles_is_vllm_engine_only() -> None:
    cfg = ManagedUnitsConfig()
    roles = cfg.roles()
    assert list(roles) == ["vllm-engine"]
    assert roles["vllm-engine"] is cfg.vllm_engine


def test_units_appear_in_roles_alongside_vllm_engine() -> None:
    cfg = ManagedUnitsConfig(
        units={
            "deepseek": ManagedUnitConfig(
                unit_name="ds4-server", default_port=8000, runtime="llama_cpp"
            )
        }
    )
    roles = cfg.roles()
    assert set(roles) == {"vllm-engine", "deepseek"}
    assert roles["deepseek"].unit_name == "ds4-server"
    assert roles["deepseek"].default_port == 8000
    assert roles["deepseek"].runtime == "llama_cpp"


def test_units_reserved_role_names_rejected() -> None:
    for reserved in ("vllm-engine", "vllm_engine", "vllm-tp", "vllm_tp", "fleet"):
        with pytest.raises(ValidationError, match="reserved role"):
            ManagedUnitsConfig(units={reserved: ManagedUnitConfig()})


def test_settings_without_units_is_noop_migration() -> None:
    """A settings payload predating the feature parses to an empty mapping."""
    settings = Settings.model_validate(
        {"managed_units": {"vllm_engine": {"enabled": True, "unit_name": "vllm-tp"}}}
    )
    assert settings.managed_units.units == {}
    assert settings.managed_units.vllm_engine.enabled is True
    assert list(settings.managed_units.roles()) == ["vllm-engine"]


def test_settings_yaml_shape_parses_units() -> None:
    """The documented settings.yaml shape round-trips through Settings."""
    settings = Settings.model_validate(
        {
            "managed_units": {
                "units": {
                    "deepseek": {
                        "unit_name": "ds4-server",
                        "default_port": 8000,
                        "runtime": "llama_cpp",
                    }
                }
            }
        }
    )
    roles = settings.managed_units.roles()
    assert roles["deepseek"].default_port == 8000
    # vllm_engine keeps its compiled-in defaults untouched.
    assert roles["vllm-engine"].default_port == 8003
    assert settings.managed_units.fleet.tp == "vllm-engine"


def test_runtime_field_defaults_to_vllm() -> None:
    """Existing configs never mention runtime; the default must not change
    adopt-managed's historical behavior (it always adopted as vLLM)."""
    assert ManagedUnitConfig().runtime == "vllm"


# -- CLI surfaces -----------------------------------------------------------


def _settings_with_deepseek_role() -> Settings:
    return Settings.model_validate(
        {
            "managed_units": {
                "units": {
                    "deepseek": {
                        "unit_name": "ds4-server",
                        "default_port": 8000,
                        "runtime": "llama_cpp",
                    }
                }
            }
        }
    )


def test_status_lists_generic_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from typer.testing import CliRunner

    import llmctl.cli as cli_mod
    from llmctl.cli import app

    monkeypatch.setattr(cli_mod, "load_settings", _settings_with_deepseek_role)
    monkeypatch.setattr(
        "llmctl.services.backends.probe_openai_v1_models", lambda url, timeout: None
    )
    result = CliRunner().invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.output)
    by_role = {row["role"]: row for row in rows}
    assert set(by_role) == {"vllm-engine", "deepseek"}
    assert by_role["deepseek"]["unit_name"] == "ds4-server"
    assert by_role["deepseek"]["port"] == 8000
    assert by_role["deepseek"]["serving"] is False


def test_adopt_managed_accepts_generic_role(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from typer.testing import CliRunner

    import llmctl.cli as cli_mod
    from llmctl.cli import app
    from llmctl.db import RuntimeName

    monkeypatch.setattr(cli_mod, "load_settings", _settings_with_deepseek_role)

    @contextmanager
    def _fake_session():
        yield None

    monkeypatch.setattr(cli_mod, "_session", _fake_session)

    calls: list[tuple] = []

    class _FakeService:
        def __init__(self, db) -> None:
            pass

        def adopt(self, runtime, endpoint, *, systemd_unit=None, timeout_s=1.5):
            calls.append((runtime, endpoint, systemd_unit))
            return SimpleNamespace(
                id="fake-id",
                runtime=runtime,
                served_name="deepseek-v4-flash",
                endpoint_url=endpoint,
                systemd_unit=systemd_unit,
            )

    monkeypatch.setattr(cli_mod, "SessionService", _FakeService)

    result = CliRunner().invoke(app, ["adopt-managed", "deepseek"])
    assert result.exit_code == 0
    assert calls == [
        (RuntimeName.LLAMA_CPP, "http://127.0.0.1:8000", "ds4-server.service")
    ]


def test_adopt_managed_unknown_role_lists_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    import llmctl.cli as cli_mod
    from llmctl.cli import app

    monkeypatch.setattr(cli_mod, "load_settings", _settings_with_deepseek_role)
    result = CliRunner().invoke(app, ["adopt-managed", "nope"])
    assert result.exit_code != 0
    assert "deepseek" in result.output
    assert "vllm-engine" in result.output


def test_status_marks_absent_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A role whose env file does not exist must not render a path.

    `resolve_env_file()` always returns *a* path, even for a runtime llmctl
    never writes an env file for. Printing it reads as "edit this file" for a
    file that isn't there — which is exactly the noise registering a llama.cpp
    role introduced.
    """
    import json

    from typer.testing import CliRunner

    import llmctl.cli as cli_mod
    from llmctl.cli import app

    real = tmp_path / "vllm-tp.env"
    real.write_text("VLLM_PORT=8003\n")

    settings = Settings.model_validate(
        {
            "managed_units": {
                "vllm_engine": {"unit_name": "vllm-tp", "env_file_path": str(real)},
                "units": {
                    "deepseek": {
                        "unit_name": "ds4",
                        "default_port": 8000,
                        "runtime": "llama_cpp",
                        "env_file_path": str(tmp_path / "does-not-exist.env"),
                    }
                },
            }
        }
    )
    monkeypatch.setattr(cli_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(
        "llmctl.services.backends.probe_openai_v1_models", lambda url, timeout: None
    )

    rows = json.loads(CliRunner().invoke(app, ["status", "--json"]).output)
    by_role = {r["role"]: r for r in rows}
    assert by_role["vllm-engine"]["env_file_exists"] is True
    assert by_role["deepseek"]["env_file_exists"] is False
    # JSON keeps the resolved path either way; only the table hides it.
    assert by_role["deepseek"]["env_file"].endswith("does-not-exist.env")

    table = CliRunner().invoke(app, ["status"]).output
    assert "does-not-exist.env" not in table
    assert "—" in table


# -- the vllm-tp -> vllm-engine rename (2026-10) ------------------------------


def test_the_old_vllm_tp_key_is_still_read_with_a_warning() -> None:
    """Ignoring it would drop a host's pinned env_file_path and unit_name."""
    with pytest.warns(FutureWarning, match="managed_units.vllm_tp is deprecated"):
        settings = Settings.model_validate(
            {"managed_units": {"vllm_tp": {"unit_name": "vllm-tp", "default_port": 9003}}}
        )
    assert settings.managed_units.vllm_engine.unit_name == "vllm-tp"
    assert settings.managed_units.vllm_engine.default_port == 9003


def test_setting_both_keys_is_refused() -> None:
    with pytest.raises(ValidationError, match="both vllm_engine and its old name vllm_tp"):
        ManagedUnitsConfig.model_validate({"vllm_tp": {}, "vllm_engine": {}})


def test_vllm_tp_attribute_still_returns_the_engine_config() -> None:
    cfg = ManagedUnitsConfig()
    assert cfg.vllm_tp is cfg.vllm_engine


def test_the_old_role_name_resolves() -> None:
    from llmctl.config import resolve_role

    assert resolve_role("vllm-tp") == "vllm-engine"
    assert resolve_role("deepseek") == "deepseek"


def test_adopt_managed_accepts_the_old_role_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from typer.testing import CliRunner

    import llmctl.cli as cli_mod
    from llmctl.cli import app

    monkeypatch.setattr(cli_mod, "load_settings", _settings_with_deepseek_role)

    @contextmanager
    def _fake_session():
        yield None

    monkeypatch.setattr(cli_mod, "_session", _fake_session)
    units: list[str | None] = []

    class _FakeService:
        def __init__(self, db) -> None:
            pass

        def adopt(self, runtime, endpoint, *, systemd_unit=None, timeout_s=1.5):
            units.append(systemd_unit)
            return SimpleNamespace(id="fake-id", runtime=runtime, served_name="m",
                                   endpoint_url=endpoint, systemd_unit=systemd_unit)

    monkeypatch.setattr(cli_mod, "SessionService", _FakeService)
    result = CliRunner().invoke(app, ["adopt-managed", "vllm-tp"])
    assert result.exit_code == 0, result.output
    assert units == ["vllm-engine.service"]
