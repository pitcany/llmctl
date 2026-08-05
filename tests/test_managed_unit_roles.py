"""Generic managed-unit roles: ManagedUnitsConfig.units + roles().

``managed_units`` used to hardcode exactly two fields (``vllm_tp`` and
``fleet``), so a healthy non-vLLM server (e.g. a llama.cpp unit) was
structurally invisible to ``llmctl status``. The generic ``units`` mapping
adds arbitrary roles keyed by name; these tests pin the compatibility
contract: existing ``vllm_tp``/``fleet`` consumers see identical behavior,
and a config without ``units`` is a no-op migration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig, Settings


def test_default_roles_is_vllm_tp_only() -> None:
    cfg = ManagedUnitsConfig()
    roles = cfg.roles()
    assert list(roles) == ["vllm-tp"]
    assert roles["vllm-tp"] is cfg.vllm_tp


def test_units_appear_in_roles_alongside_vllm_tp() -> None:
    cfg = ManagedUnitsConfig(
        units={
            "deepseek": ManagedUnitConfig(
                unit_name="ds4-server", default_port=8000, runtime="llama_cpp"
            )
        }
    )
    roles = cfg.roles()
    assert set(roles) == {"vllm-tp", "deepseek"}
    assert roles["deepseek"].unit_name == "ds4-server"
    assert roles["deepseek"].default_port == 8000
    assert roles["deepseek"].runtime == "llama_cpp"


def test_units_reserved_role_names_rejected() -> None:
    for reserved in ("vllm-tp", "vllm_tp", "fleet"):
        with pytest.raises(ValidationError, match="reserved role"):
            ManagedUnitsConfig(units={reserved: ManagedUnitConfig()})


def test_settings_without_units_is_noop_migration() -> None:
    """A settings payload predating the feature parses to an empty mapping."""
    settings = Settings.model_validate(
        {"managed_units": {"vllm_tp": {"enabled": True, "unit_name": "vllm-tp"}}}
    )
    assert settings.managed_units.units == {}
    assert settings.managed_units.vllm_tp.enabled is True
    assert list(settings.managed_units.roles()) == ["vllm-tp"]


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
    # vllm_tp keeps its compiled-in defaults untouched.
    assert roles["vllm-tp"].default_port == 8003
    assert settings.managed_units.fleet.tp == "vllm-tp"


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
    assert set(by_role) == {"vllm-tp", "deepseek"}
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
    assert "vllm-tp" in result.output
