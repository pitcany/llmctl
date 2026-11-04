"""save_preset writes only set keys, keeps comments and never deletes files."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.presets import save_preset
from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.schema import Model, PresetSchemaError
from llmctl.presets.store import load_one

REQ = (
    "alias: x\nserved_name: x\nmodel_id: org/m\nquantization: awq\n"
    "vllm_quantization_flag: awq\ntensor_parallel_size: 2\nmax_model_len: 4096\n"
)
REQ_KW = {
    "alias": "x", "served_name": "x", "model_id": "org/m", "quantization": "awq",
    "vllm_quantization_flag": "awq", "tensor_parallel_size": 2, "max_model_len": 4096,
}


@pytest.fixture(autouse=True)
def _xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _legacy(text: str) -> Path:
    p = user_config_dir() / "x.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(text)
    return p


def test_unset_keys_stay_absent_and_default_keys_stay_present() -> None:
    p = _legacy(REQ + "port: 8000\n")
    m = load_one("x")
    save_preset(m)
    text = p.read_text()
    assert "gpu_memory_utilization" not in text and "max_num_seqs" not in text
    assert "port: 8000" in text


def test_new_model_writes_exactly_given_keys() -> None:
    m = Model(**REQ_KW)
    path = save_preset(m)
    assert path == default_preset_dir() / "x.yaml"
    assert len(path.read_text().strip().splitlines()) == 7
    assert load_one("x") == m


def test_comments_survive_and_changes_apply() -> None:
    text = (
        "# header\n# more\n" + REQ.replace("awq\nvllm", "awq  # q\nvllm", 1)
        + "# between\nmax_num_seqs: 4  # seqs\nspec_config:\n  model: a\n  n: 2\n"
        "shortcuts:\n- s1\n- s2\nport: 9\n"
    )
    p = _legacy(text)
    m = load_one("x")
    new = Model(**{**m.model_dump(exclude_unset=True), "max_num_seqs": 8,
                   "spec_config": {"model": "b"}})
    new = Model(**{k: v for k, v in new.model_dump(exclude_unset=True).items() if k != "port"})
    save_preset(new)
    out = p.read_text()
    assert "# header\n# more\n" in out and "# between\n" in out and "# q\n" in out
    assert "max_num_seqs: 8  # seqs\n" in out
    assert "port" not in out
    assert "shortcuts:\n- s1\n- s2\n" in out
    assert load_one("x") == new
    assert load_one("x").spec_config == {"model": "b"}


def test_symlink_canonical_to_legacy_and_reverse() -> None:
    legacy = _legacy(REQ)
    canon = default_preset_dir() / "x.yaml"
    canon.parent.mkdir(parents=True)
    canon.symlink_to(legacy)
    save_preset(Model(**{**REQ_KW, "max_model_len": 1024}))
    assert canon.is_symlink() and legacy.exists() and "1024" in legacy.read_text()
    assert load_one("x").max_model_len == 1024

    canon.unlink()
    legacy.unlink()
    canon.write_text(REQ)
    legacy.symlink_to(canon)
    save_preset(Model(**{**REQ_KW, "max_model_len": 2048}))
    assert legacy.is_symlink() and not canon.is_symlink()
    assert "2048" in canon.read_text()


def test_config_dir_symlink_stays(tmp_path: Path) -> None:
    d = tmp_path / "cfg"
    d.mkdir()
    real = tmp_path / "real.yaml"
    real.write_text(REQ)
    (d / "x.yaml").symlink_to(real)
    save_preset(Model(**{**REQ_KW, "max_model_len": 512}), config_dir=d)
    assert (d / "x.yaml").is_symlink() and "512" in real.read_text()


def test_invalid_model_leaves_file_unchanged() -> None:
    p = _legacy(REQ)
    before = p.read_bytes()
    bad = Model.model_construct(_fields_set={"alias", "tensor_parallel_size"},
                                alias="x", tensor_parallel_size=99)
    with pytest.raises(PresetSchemaError):
        save_preset(bad)
    assert p.read_bytes() == before
