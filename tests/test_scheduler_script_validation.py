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

from llmctl.services.scheduler import SchedulerError, SchedulerService


@pytest.fixture
def scheduler() -> SchedulerService:
    return SchedulerService(None)


def _build(scheduler: SchedulerService, script: object, args: list | None = None):
    params: dict[str, object] = {"script": script}
    if args is not None:
        params["args"] = args
    return scheduler._build_python_command(None, params)


def test_interpreter_flag_as_script_is_refused(scheduler) -> None:
    """The exact RCE shape: `-c` plus a payload in args.

    The payload below is an inert string literal that only ever reaches
    ``_build_python_command`` as data; the assertion is that it is *refused*, so
    nothing executes it. It is spelled out rather than stubbed because the point
    of the test is the shape a real caller would send.
    """
    with pytest.raises(SchedulerError) as exc:
        _build(scheduler, "-c", ["import os; os.system('id')"])
    assert "-c" in str(exc.value)


@pytest.mark.parametrize("flag", ["-c", "-m", "-", "--version"])
def test_no_interpreter_flag_is_accepted_as_a_script(scheduler, flag) -> None:
    with pytest.raises(SchedulerError):
        _build(scheduler, flag)


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
