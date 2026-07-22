"""Tests for the host-validation checks behind ``llmctl validate``.

Each check exists because the 2026-07 environment audit found that class
of drift by hand: presets pointing at deleted checkpoints, registry rows
whose target was gone, an orphaned dangling store symlink, and vLLM
listening on :8004 while everything recorded :8003.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from llmctl.config import ManagedUnitConfig, ModelDirsConfig, ModelRoot
from llmctl.db import RuntimeName
from llmctl.presets.schema import Model as PresetModel
from llmctl.schemas import Model
from llmctl.services.validate import (
    as_local_path,
    check_managed_unit_ports,
    check_model_root_symlinks,
    check_preset_model_ids,
    check_registry_paths,
)


def _preset(alias: str, model_id: str) -> PresetModel:
    return PresetModel(
        alias=alias,
        served_name=alias,
        model_id=model_id,
        quantization="fp8",
        vllm_quantization_flag="fp8",
        tensor_parallel_size=2,
        max_model_len=4096,
    )


class _FakeSystemctl:
    """Stand-in for SystemctlRunner with scripted is-active answers."""

    def __init__(self, active: set[str], *, available: bool = True) -> None:
        self._active = active
        self._available = available

    def available(self) -> bool:
        return self._available

    def is_active(self, unit: str) -> bool:
        return unit in self._active


def _responder(serving_ports: dict[int, list[str]]) -> Any:
    def get(url: str, timeout: float) -> Any:
        port = int(url.split(":")[2].split("/")[0])
        if port not in serving_ports:
            raise OSError(f"connection refused on {port}")
        payload = {"data": [{"id": i} for i in serving_ports[port]]}
        return io.BytesIO(json.dumps(payload).encode())

    return get


# --- as_local_path ---------------------------------------------------------


def test_hf_repo_id_is_not_a_local_path() -> None:
    """`org/model` is a legitimate model_id, not a filesystem claim."""
    assert as_local_path("deepreinforce-ai/Ornith-1.0-35B-FP8") is None


def test_absolute_and_home_relative_values_are_paths() -> None:
    assert as_local_path("/models/v6") == Path("/models/v6")
    assert as_local_path("~/models/v6") == Path.home() / "models/v6"


# --- preset model_id -------------------------------------------------------


def test_preset_with_missing_model_id_is_flagged(tmp_path: Path) -> None:
    presets = {"gone": _preset("gone", str(tmp_path / "not-there"))}
    findings = check_preset_model_ids(presets)
    assert len(findings) == 1
    assert findings[0].check == "preset-model-missing"
    assert findings[0].target == "gone"


def test_preset_with_present_model_id_is_clean(tmp_path: Path) -> None:
    (tmp_path / "ckpt").mkdir()
    assert check_preset_model_ids({"ok": _preset("ok", str(tmp_path / "ckpt"))}) == []


def test_preset_pointing_through_a_dead_symlink_is_flagged(tmp_path: Path) -> None:
    """The store-symlink case: the link exists, its target does not."""
    link = tmp_path / "store-entry"
    link.symlink_to(tmp_path / "deleted-checkpoint")
    findings = check_preset_model_ids({"dangling": _preset("dangling", str(link))})
    assert [f.check for f in findings] == ["preset-model-missing"]


def test_preset_with_hub_model_id_is_not_flagged() -> None:
    """A hub id must not be reported as a missing local path."""
    assert check_preset_model_ids({"hub": _preset("hub", "org/model")}) == []


# --- registry paths --------------------------------------------------------


def test_registry_row_with_missing_path_is_flagged(tmp_path: Path) -> None:
    model = Model(
        name="ornith-35b",
        runtime=RuntimeName.VLLM,
        path=str(tmp_path / "vanished"),
    )
    findings = check_registry_paths([model])
    assert len(findings) == 1
    assert findings[0].check == "registry-path-missing"
    assert findings[0].target == "ornith-35b"


def test_registry_row_without_path_is_skipped() -> None:
    """A null path is 'unknown', not 'missing' — the pre-root-field state."""
    assert check_registry_paths([Model(name="m", runtime=RuntimeName.VLLM)]) == []


def test_registry_row_with_hub_source_is_skipped() -> None:
    model = Model(name="m", runtime=RuntimeName.VLLM, path="org/model")
    assert check_registry_paths([model]) == []


# --- model-root symlinks ---------------------------------------------------


def _root_config(tmp_path: Path, **scan: Any) -> ModelDirsConfig:
    return ModelDirsConfig(
        model_roots=[ModelRoot(name="store", default_path=str(tmp_path))],
        scan=scan,
    )


def test_dangling_symlink_in_model_root_is_flagged(tmp_path: Path) -> None:
    (tmp_path / "dead").symlink_to(tmp_path / "never-existed")
    findings = check_model_root_symlinks(_root_config(tmp_path))
    assert len(findings) == 1
    assert findings[0].check == "broken-symlink"
    assert findings[0].target == "store"


def test_live_symlink_in_model_root_is_clean(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert check_model_root_symlinks(_root_config(tmp_path)) == []


def test_disabled_root_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "dead").symlink_to(tmp_path / "never-existed")
    config = ModelDirsConfig(
        model_roots=[ModelRoot(name="store", default_path=str(tmp_path), enabled=False)]
    )
    assert check_model_root_symlinks(config) == []


def test_missing_root_is_skipped(tmp_path: Path) -> None:
    config = ModelDirsConfig(
        model_roots=[ModelRoot(name="store", default_path=str(tmp_path / "absent"))]
    )
    assert check_model_root_symlinks(config) == []


def test_duplicate_roots_are_swept_once(tmp_path: Path) -> None:
    """Two roots resolving to the same directory must not double-report."""
    (tmp_path / "dead").symlink_to(tmp_path / "never-existed")
    config = ModelDirsConfig(
        model_roots=[
            ModelRoot(name="a", default_path=str(tmp_path)),
            ModelRoot(name="b", default_path=str(tmp_path)),
        ]
    )
    assert len(check_model_root_symlinks(config)) == 1


def test_symlink_below_max_depth_is_not_swept(tmp_path: Path) -> None:
    """Depth is bounded so a deep cache tree cannot stall validation."""
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "dead").symlink_to(tmp_path / "never-existed")
    assert check_model_root_symlinks(_root_config(tmp_path, max_depth=2)) == []
    assert len(check_model_root_symlinks(_root_config(tmp_path, max_depth=4))) == 1


# --- managed-unit ports ----------------------------------------------------


def _unit(port: int = 8003) -> ManagedUnitConfig:
    return ManagedUnitConfig(unit_name="vllm-tp", default_port=port)


def test_active_unit_serving_its_port_is_clean() -> None:
    findings = check_managed_unit_ports(
        [_unit(8003)],
        systemctl=_FakeSystemctl({"vllm-tp"}),
        http_get=_responder({8003: ["ornith-35b"]}),
    )
    assert findings == []


def test_active_unit_listening_elsewhere_is_flagged() -> None:
    """The audit case: vLLM took :8004 while everything recorded :8003."""
    findings = check_managed_unit_ports(
        [_unit(8003)],
        systemctl=_FakeSystemctl({"vllm-tp"}),
        http_get=_responder({8004: ["ornith-35b"]}),
    )
    assert len(findings) == 1
    assert findings[0].check == "port-drift"
    assert "8003" in findings[0].detail


def test_inactive_unit_is_not_drift() -> None:
    """A stopped unit serving nothing is expected, not a finding."""
    findings = check_managed_unit_ports(
        [_unit(8003)],
        systemctl=_FakeSystemctl(set()),
        http_get=_responder({}),
    )
    assert findings == []


def test_unit_answering_with_empty_catalog_is_flagged() -> None:
    """Bound but serving nothing is still 'not where it is registered'."""
    findings = check_managed_unit_ports(
        [_unit(8003)],
        systemctl=_FakeSystemctl({"vllm-tp"}),
        http_get=_responder({8003: []}),
    )
    assert [f.check for f in findings] == ["port-drift"]


def test_host_without_systemctl_reports_nothing() -> None:
    """Containers and dev laptops must not fail validation on this check."""
    findings = check_managed_unit_ports(
        [_unit(8003)],
        systemctl=_FakeSystemctl({"vllm-tp"}, available=False),
        http_get=_responder({}),
    )
    assert findings == []
