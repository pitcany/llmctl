"""llama.cpp registry models appear in the swap list, once.

`llmctl swap NAME` could already serve a llama.cpp registry model, but the
list (`swap --list`, the bare `swap` picker, the TUI Swap screen) showed only
presets and lanes. These tests pin which models are listed, that a model a
lane already serves is not listed twice (or started twice), and how the rows
read.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ModelCreate
from llmctl.services.lanes import (
    Lane,
    SwapState,
    SwapTarget,
    cards_label,
    lane_for_model,
    serving_keys,
    session_rows_for,
)
from llmctl.services.registry import RegistryService
from llmctl.services.serve import route
from llmctl.services.session_swap import GpuSession, Placement

DSV4_FILE = "/m/dsv4/DeepSeek-MXFP4-00001-of-00002.gguf"


def _lane_row(name="dsv4", served="deepseek-v4-flash", unit="deepseek.service") -> SwapTarget:
    lane = Lane(
        name=name,
        served_model=served,
        upstream="http://127.0.0.1:8000",
        start_cmd=("start",),
        unit=unit,
        runtime="llama_cpp",
    )
    return SwapTarget(
        key=name, kind="lane", model=served, engine="llama.cpp", upstream=lane.upstream, lane=lane
    )


def _units(unit: str) -> str:
    if unit != "deepseek.service":
        return ""
    return f"ExecStart=llama-server -m {DSV4_FILE} --port 8000"


def _model(name, path=None, runtime="llama_cpp", host=None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, path=path, runtime=SimpleNamespace(value=runtime), host=host, id=name
    )


def _target(model, cards=2, problem=None) -> SwapTarget:
    return SwapTarget(
        key=model.name,
        kind="session",
        model=model.name,
        engine="llama.cpp",
        upstream="",
        model_id=model.id,
        placement=Placement(cards=cards, problem=problem),
    )


# -- which models are listed ------------------------------------------------------


def test_lane_link_by_the_file_its_unit_loads() -> None:
    catalog = [_lane_row()]
    found = lane_for_model("deepseek-v4-flash-0731-mxfp4", DSV4_FILE, catalog, unit_text=_units)
    assert found is not None and found.key == "dsv4"
    assert lane_for_model("gpt-oss", "/m/gpt-oss.gguf", catalog, unit_text=_units) is None


def test_lane_link_by_name() -> None:
    assert lane_for_model("deepseek-v4-flash", None, [_lane_row()], unit_text=_units) is not None


def test_session_rows_list_only_local_fitting_models_no_lane_serves() -> None:
    models = [
        _model("zeta-gguf", "/m/z.gguf"),
        _model("gpt-oss", "/m/gpt-oss.gguf"),
        _model("dsv4-registry-name", DSV4_FILE),  # the dsv4 lane serves it
        _model("too-big", "/m/big.gguf"),
        _model("on-the-mac", "/m/mac.gguf", host="mac"),
        _model("qwen3:8b", None, runtime="ollama"),
    ]
    rows = session_rows_for(
        models,
        [_lane_row()],
        target_for=lambda m: _target(m, problem="does not fit" if m.name == "too-big" else None),
        is_local=lambda m: m.host is None,
        unit_text=_units,
    )
    assert [r.key for r in rows] == ["gpt-oss", "zeta-gguf"]


# -- how the rows read --------------------------------------------------------------


def test_cards_label_for_session_rows() -> None:
    assert cards_label(_target(_model("a"), cards=2)) == "2 (split)"
    assert cards_label(_target(_model("a"), cards=1)) == "any 1"
    assert cards_label(_lane_row()) == "all"  # lane without a gpus: footprint


def test_a_running_session_row_reads_serving() -> None:
    row = _target(_model("gpt-oss"))
    state = SwapState(
        catalog=[row],
        serving=set(),
        active_units=set(),
        vllm_unit="vllm-tp.service",
        vllm_active=False,
        gpu_sessions=[GpuSession("s1", "gpt-oss", "llama_cpp", 8005, (0, 1))],
    )
    assert state.state_of(row) == "serving"
    state.gpu_sessions = []
    assert state.state_of(row) == ""


def test_session_rows_are_not_probed() -> None:
    probed: list[str] = []

    def probe(url: str) -> list[str]:
        probed.append(url)
        return []

    serving_keys([_target(_model("gpt-oss")), _lane_row()], probe)
    assert probed == ["http://127.0.0.1:8000"]


# -- swap NAME with the registry name -----------------------------------------------


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    units = tmp_path / "xdg" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "deepseek.service").write_text(f"ExecStart=%h/bin/llama-server -m {DSV4_FILE}\n")
    url = f"sqlite:///{tmp_path / 'swap-list.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def test_swap_by_registry_name_uses_the_lane_that_serves_it(db) -> None:
    model = RegistryService(db).add_model(
        ModelCreate(
            name="deepseek-v4-flash-0731-mxfp4",
            runtime=RuntimeName.LLAMA_CPP,
            source=DSV4_FILE,
            path=DSV4_FILE,
        )
    )
    served = route(db, model, [_lane_row()])
    assert served.kind == "lane"
    assert served.target is not None and served.target.key == "dsv4"


def test_route_carries_the_port_the_default_profile_pins(db) -> None:
    from llmctl.schemas import ModelUpdate, ProfileCreate
    from llmctl.services.profiles import ProfileService

    registry = RegistryService(db)
    model = registry.add_model(
        ModelCreate(name="gpt-oss", runtime=RuntimeName.LLAMA_CPP, source="/m/g.gguf")
    )
    assert route(db, model, []).target.port is None
    profile = ProfileService(db).create_profile(
        ProfileCreate(name="gpt-oss-swap", runtime=RuntimeName.LLAMA_CPP, parameters={"port": 8005})
    )
    model = registry.update_model(model.id, ModelUpdate(default_profile_id=profile.id))
    assert route(db, model, []).target.port == 8005
