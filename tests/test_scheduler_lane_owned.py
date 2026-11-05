"""`llmctl start`/`plan` must not launch a second server for a lane's model.

`llmctl run gpt-oss-120b-mxfp4` already swaps to the `gptoss` lane (its tuned
systemd unit). `llmctl start`/`plan` and POST /sessions went straight to the
scheduler and planned a second llama-server on the same weights with llmctl
defaults: port 8000 (the DeepSeek lane's), one card, none of the unit's flags
(P3 trace, 2026-11-05). The scheduler now refuses that launch and names the
lane, and `start`/`plan` apply the model's default profile like `run` does.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import ModelRecord, ProfileRecord, RuntimeName, get_engine, init_db
from llmctl.schemas import SessionStartRequest
from llmctl.services.lanes import Lane, build_catalog
from llmctl.services.scheduler import SchedulerService

WEIGHTS = "/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf"


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'lane.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _model(db: Session, runtime: RuntimeName = RuntimeName.LLAMA_CPP) -> ModelRecord:
    model = ModelRecord(name="gpt-oss-120b-mxfp4", path=WEIGHTS, runtime=runtime)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _catalog(name: str = "gpt-oss-120b") -> list:
    lane = Lane(
        name="gptoss",
        served_model=name,
        upstream="http://127.0.0.1:8005",
        start_cmd=("gptoss-start", "--take-gpus"),
        unit="gpt-oss-120b.service",
        runtime="llama_cpp",
    )
    return build_catalog({}, [lane], vllm_port=0)


def _plan(db: Session, model: ModelRecord, catalog) -> list[str]:
    source = catalog if callable(catalog) else (lambda: catalog)
    scheduler = SchedulerService(db, lane_catalog=source)
    plan = scheduler.create_launch_plan(
        SessionStartRequest(model_id=model.id, runtime=model.runtime, dry_run=True)
    )
    return plan.refusal_reasons


def _lane_refusals(reasons: list[str]) -> list[str]:
    return [r for r in reasons if "lane" in r]


def test_model_served_by_lane_unit_file_is_refused(tmp_path, monkeypatch):
    """Matched by the weights the lane's unit loads, as registry names rarely match."""
    monkeypatch.setattr(
        "llmctl.services.lanes.user_unit_text", lambda unit: f"ExecStart=x --model {WEIGHTS}"
    )
    db = _db(tmp_path)
    model = _model(db)
    reasons = _lane_refusals(_plan(db, model, _catalog()))
    assert len(reasons) == 1
    assert "`gptoss` lane (gpt-oss-120b.service)" in reasons[0]
    assert "llmctl run gptoss" in reasons[0]


def test_model_served_by_lane_name_is_refused(tmp_path):
    db = _db(tmp_path)
    model = _model(db)
    reasons = _lane_refusals(_plan(db, model, _catalog(name="gpt-oss-120b-mxfp4")))
    assert len(reasons) == 1


def test_model_without_lane_is_not_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("llmctl.services.lanes.user_unit_text", lambda unit: "")
    db = _db(tmp_path)
    model = _model(db)
    assert _lane_refusals(_plan(db, model, _catalog())) == []


def test_non_llama_cpp_runtime_is_not_checked(tmp_path):
    db = _db(tmp_path)
    model = _model(db, runtime=RuntimeName.VLLM)
    assert _lane_refusals(_plan(db, model, _catalog(name="gpt-oss-120b-mxfp4"))) == []


def test_broken_lane_source_does_not_block_planning(tmp_path):
    def broken() -> list:
        raise RuntimeError("router.yaml unreadable")

    db = _db(tmp_path)
    model = _model(db)
    assert _lane_refusals(_plan(db, model, broken)) == []


def test_start_request_falls_back_to_the_default_profile(tmp_path):
    from llmctl.cli import _build_start_request

    db = _db(tmp_path)
    profile = ProfileRecord(
        name="gpt-oss-120b-swap", runtime=RuntimeName.LLAMA_CPP, parameters={"port": 8005}
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    model = _model(db)
    model.default_profile_id = profile.id
    db.add(model)
    db.commit()

    request = _build_start_request(db, model.name, None, "auto", False, False, True)
    assert request.profile_id == profile.id

    explicit = ProfileRecord(name="other", runtime=RuntimeName.LLAMA_CPP, parameters={})
    db.add(explicit)
    db.commit()
    request = _build_start_request(db, model.name, "other", "auto", False, False, True)
    assert request.profile_id == explicit.id
