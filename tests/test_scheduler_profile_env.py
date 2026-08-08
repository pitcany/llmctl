"""A profile's ``environment_variables`` must reach the launched process.

The field has existed on `Profile`, `ProfileCreate` and `ProfileUpdate` since the
schema was written, and `llmctl profile edit` will happily persist it — but the
scheduler built the launch environment from ``runtime_config.env`` alone and
never looked at the profile. Anything set there was silently inert.

That matters beyond tidiness: `runtimes.<name>.env` is per-*runtime*, so it is
shared by every model on that runtime. Per-model environment — a key file to
materialize, a cache directory, a token — has nowhere else to live.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ProfileCreate, SessionStartRequest
from llmctl.services.profiles import ProfileService
from llmctl.services.scheduler import SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'env.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _profile(db: Session, env: dict[str, str]) -> str:
    profile = ProfileService(db).create_profile(
        ProfileCreate(
            name="with-env",
            runtime=RuntimeName.LLAMA_CPP,
            environment_variables=env,
        )
    )
    assert profile.id is not None
    return profile.id


def _plan(db: Session, profile_id: str):
    return SchedulerService(db).create_launch_plan(
        SessionStartRequest(
            model_id="m1",
            profile_id=profile_id,
            runtime=RuntimeName.LLAMA_CPP,
            dry_run=True,
        )
    )


def test_profile_environment_reaches_the_plan(tmp_path: Path) -> None:
    db = _db(tmp_path)
    try:
        pid = _profile(db, {"LLAMA_GUARD_APIKEY_DST": "/run/user/1000/key"})

        plan = _plan(db, pid)

        assert plan.env.get("LLAMA_GUARD_APIKEY_DST") == "/run/user/1000/key"
    finally:
        db.close()


def test_profile_environment_beats_the_runtime_default(tmp_path: Path) -> None:
    """Per-model config is more specific than per-runtime config, so it wins."""
    db = _db(tmp_path)
    try:
        service = SchedulerService(db)
        runtime_env = dict(service.settings.runtimes["llama_cpp"].env)
        runtime_env["LLAMA_GUARD_MIN_FREE_MIB"] = "25000"
        service.settings.runtimes["llama_cpp"].env = runtime_env

        pid = _profile(db, {"LLAMA_GUARD_MIN_FREE_MIB": "40000"})
        plan = service.create_launch_plan(
            SessionStartRequest(
                model_id="m1",
                profile_id=pid,
                runtime=RuntimeName.LLAMA_CPP,
                dry_run=True,
            )
        )

        assert plan.env.get("LLAMA_GUARD_MIN_FREE_MIB") == "40000"
    finally:
        db.close()


def test_cuda_visible_devices_still_wins_over_both(tmp_path: Path) -> None:
    """GPU selection is the scheduler's own decision and must not be overridden.

    A profile that pinned CUDA_VISIBLE_DEVICES would silently contradict
    `--gpus`, and the plan would report a GPU set the process never saw.
    """
    db = _db(tmp_path)
    try:
        pid = _profile(db, {"CUDA_VISIBLE_DEVICES": "7"})

        plan = _plan(db, pid)

        assert plan.env.get("CUDA_VISIBLE_DEVICES") != "7"
    finally:
        db.close()


def test_no_profile_environment_is_unchanged(tmp_path: Path) -> None:
    db = _db(tmp_path)
    try:
        pid = _profile(db, {})

        plan = _plan(db, pid)

        assert "LLAMA_GUARD_APIKEY_DST" not in plan.env
    finally:
        db.close()
