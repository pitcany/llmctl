"""`runtime=python_script` must not accept an interpreter flag as its script.

`_build_python_command` produced ``[sys.executable, str(parameters['script']),
*args]`` with no check on what ``script`` was. ``script="-c"`` therefore reached
the interpreter as a flag, turning the first element of ``args`` into code:

    {"script": "-c", "args": ["import os; os.system('...')"]}

`force=true` is a plain field on the start request and makes `validate` skip
every refusal, so a path-existence refusal is not enough on its own -- the check
has to live where `force` cannot reach it.
"""

from __future__ import annotations

import sys

import pytest

from llmctl.db import RuntimeName
from llmctl.schemas import SessionStartRequest
from llmctl.services.scheduler import SchedulerService, python_script_refusal


@pytest.fixture
def scheduler() -> SchedulerService:
    return SchedulerService(None)


def _build(scheduler: SchedulerService, script: object, args: list | None = None):
    params: dict[str, object] = {"script": script}
    if args is not None:
        params["args"] = args
    return scheduler._build_python_command(None, params)


def test_interpreter_flag_as_script_builds_no_command(scheduler) -> None:
    """The exact RCE shape: `-c` plus a payload in args.

    The payload below is an inert string literal that only ever reaches
    ``_build_python_command`` as data; the assertion is that no command is
    built from it, so nothing can execute it. It is spelled out rather than
    stubbed because the point is the shape a real caller would send.

    An empty command is the hard stop: the adapter fails the session with
    "Launch plan has no command to execute", and `force` cannot conjure one.
    """
    assert _build(scheduler, "-c", ["import os; os.system('id')"]) == []
    assert "-c" in (python_script_refusal("-c") or "")


@pytest.mark.parametrize("flag", ["-c", "-m", "-", "--version"])
def test_no_interpreter_flag_is_accepted_as_a_script(scheduler, flag) -> None:
    assert _build(scheduler, flag) == []
    assert python_script_refusal(flag) is not None


def test_a_missing_script_still_builds(scheduler, tmp_path) -> None:
    """Existence is deliberately not checked here.

    Without a leading "-" the interpreter reads the argument as a path, so a
    missing file is a failed launch rather than code execution — and requiring
    existence at build time would break `llmctl plan`, which previews a command
    before the script is necessarily in place. Non-existence is already a plan
    refusal.
    """
    missing = str(tmp_path / "does-not-exist.py")

    assert _build(scheduler, missing) == [sys.executable, missing]


def test_a_real_script_still_builds(scheduler, tmp_path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")

    command = _build(scheduler, str(script), ["--flag", "1"])

    assert command == [sys.executable, str(script), "--flag", "1"]


def test_no_script_at_all_is_still_an_empty_command(scheduler) -> None:
    """Unchanged: an absent script yields no command rather than raising."""
    assert scheduler._build_python_command(None, {}) == []


def test_planning_a_flag_script_refuses_instead_of_raising() -> None:
    """`llmctl plan` / `POST /sessions/plan` must return a plan, not blow up.

    Every other refusal in the planner arrives as a `refusal_reasons` entry on a
    `LaunchPlan`. Raising out of `_build_command` instead would turn the preview
    path into a traceback (or a 500), which is both worse to diagnose and
    inconsistent with how the planner reports everything else.
    """
    scheduler = SchedulerService(None)
    request = SessionStartRequest(
        model_id="m1",
        runtime=RuntimeName.PYTHON_SCRIPT,
        dry_run=True,
        parameters={"script": "-c", "args": ["import os"]},
    )

    plan = scheduler.create_launch_plan(request)

    assert plan.command == [], "a flag-shaped script must not become a command"
    assert any("cannot start with" in r for r in plan.refusal_reasons), (
        f"the operator is not told why: {plan.refusal_reasons}"
    )


def test_planning_a_normal_script_is_unaffected(tmp_path) -> None:
    scheduler = SchedulerService(None)
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    request = SessionStartRequest(
        model_id="m1",
        runtime=RuntimeName.PYTHON_SCRIPT,
        dry_run=True,
        parameters={"script": str(script)},
    )

    plan = scheduler.create_launch_plan(request)

    assert plan.command[:2] == [sys.executable, str(script)]
    assert not any("cannot start with" in r for r in plan.refusal_reasons)
