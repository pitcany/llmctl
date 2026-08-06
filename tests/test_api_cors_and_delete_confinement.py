"""Two ways the API handed out more than it meant to.

**CORS.** ``cors_origins: ["*"]`` combined with the hardcoded
``allow_credentials=True`` makes Starlette echo whatever ``Origin`` it is given
and mirror any requested header — so any page the operator visits can drive the
control plane from their browser, including ``POST /sessions/start``.

**delete_files.** ``DELETE /models/{id}?delete_files=true`` recursively unlinked
``record.path``, and ``POST/PUT /models`` accept any string for that field. So a
row could name ``/home/yannik/AI`` and the delete would walk it. Errors were
swallowed and the route returned 204 either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.config import APISettings
from llmctl.services.registry import RegistryService

# --- M4: wildcard CORS --------------------------------------------------------


def test_wildcard_origin_is_rejected_at_config_load() -> None:
    """`*` plus credentials is the combination that opens the API to any page."""
    with pytest.raises(ValueError) as exc:
        APISettings(cors_origins=["*"])
    assert "*" in str(exc.value)


def test_explicit_origins_are_fine() -> None:
    s = APISettings(cors_origins=["http://localhost:3000"])
    assert s.cors_origins == ["http://localhost:3000"]


def test_no_origins_is_the_default() -> None:
    assert APISettings().cors_origins == []


# --- M3: delete_files confinement --------------------------------------------


def _roots(tmp_path: Path) -> list[Path]:
    root = tmp_path / "models"
    root.mkdir()
    return [root]


def test_delete_inside_a_model_root_is_allowed(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    target = roots[0] / "some-model"
    target.mkdir()
    (target / "weights.bin").write_text("x")

    removed = RegistryService._delete_artifact(str(target), roots=roots)

    assert removed is True
    assert not target.exists()


def test_delete_outside_every_root_is_refused(tmp_path: Path) -> None:
    """The attack: a registry row naming a path the operator never meant."""
    roots = _roots(tmp_path)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "data.txt").write_text("keep me")

    removed = RegistryService._delete_artifact(str(outside), roots=roots)

    assert removed is False
    assert (outside / "data.txt").read_text() == "keep me", "deleted outside the roots"


def test_traversal_out_of_a_root_is_refused(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    outside = tmp_path / "escape"
    outside.mkdir()
    (outside / "data.txt").write_text("keep me")
    sneaky = str(roots[0] / ".." / "escape")

    removed = RegistryService._delete_artifact(sneaky, roots=roots)

    assert removed is False
    assert (outside / "data.txt").exists()


def test_a_symlink_out_of_a_root_is_not_followed(tmp_path: Path) -> None:
    """Unlinking the link is fine; walking through it is not."""
    roots = _roots(tmp_path)
    outside = tmp_path / "target"
    outside.mkdir()
    (outside / "data.txt").write_text("keep me")
    link = roots[0] / "link"
    link.symlink_to(outside, target_is_directory=True)

    RegistryService._delete_artifact(str(link), roots=roots)

    assert (outside / "data.txt").exists(), "followed a symlink out of the root"


def test_a_configured_root_itself_is_never_deletable(tmp_path: Path) -> None:
    """A row naming the root would otherwise take every model with it."""
    roots = _roots(tmp_path)
    (roots[0] / "keep.bin").write_text("x")

    removed = RegistryService._delete_artifact(str(roots[0]), roots=roots)

    assert removed is False
    assert (roots[0] / "keep.bin").exists(), "deleted the whole model root"


def test_no_roots_configured_refuses_rather_than_deleting_anything(tmp_path: Path) -> None:
    """Fail closed: an empty root list must not mean 'anything goes'."""
    target = tmp_path / "model"
    target.mkdir()

    removed = RegistryService._delete_artifact(str(target), roots=[])

    assert removed is False
    assert target.exists()
