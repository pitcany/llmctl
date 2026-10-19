import sys
import threading
from datetime import UTC, datetime

import pytest

from llmctl.services import swap_jobs
from llmctl.services.swap_jobs import JobBusy, SwapJobs, build_run_argv, run_argv


def test_run_argv_collects_lines():
    got: list[str] = []
    rc = run_argv([sys.executable, "-c", "print('a'); print('b')"], got.append)
    assert got == ["a", "b"] and rc == 0


def test_run_argv_exit_code():
    assert run_argv([sys.executable, "-c", "raise SystemExit(3)"], lambda _l: None) == 3


def test_run_argv_env():
    got: list[str] = []
    code = "import os; print(os.environ['NO_COLOR'], os.environ['COLUMNS'])"
    run_argv([sys.executable, "-c", code], got.append)
    no_color, cols = got[0].split()
    assert no_color == "1" and int(cols) >= 200


def test_success(monkeypatch):
    def fake(argv, on_line):
        on_line("x")
        on_line("y")
        return 0

    monkeypatch.setattr(swap_jobs, "run_argv", fake)
    jobs = SwapJobs()
    job = jobs.start("n", "m", ["z"])
    jobs.wait(job.job_id, 5)
    assert job.status == "succeeded" and job.exit_code == 0
    assert list(job.output) == ["x", "y"] and job.finished_at is not None
    assert jobs.get(job.job_id) is job and jobs.running() is None


def test_failure(monkeypatch):
    monkeypatch.setattr(swap_jobs, "run_argv", lambda a, cb: 2)
    jobs = SwapJobs()
    job = jobs.start("n", "m", [])
    jobs.wait(job.job_id, 5)
    assert job.status == "failed" and job.exit_code == 2


def test_raise(monkeypatch):
    def boom(a, cb):
        raise OSError("nope")

    monkeypatch.setattr(swap_jobs, "run_argv", boom)
    jobs = SwapJobs()
    job = jobs.start("n", "m", [])
    jobs.wait(job.job_id, 5)
    assert job.status == "failed" and job.exit_code is None
    assert len(job.output) == 1 and "nope" in job.output[0]


def test_busy_then_free(monkeypatch):
    ev = threading.Event()
    monkeypatch.setattr(swap_jobs, "run_argv", lambda a, cb: (ev.wait(5), 0)[1])
    jobs = SwapJobs()
    job = jobs.start("n", "m", [])
    assert jobs.running() is job
    d = job.to_dict()
    assert d["finished_at"] is None
    with pytest.raises(JobBusy) as ei:
        jobs.start("n2", "m2", [])
    assert ei.value.job.job_id == job.job_id
    ev.set()
    jobs.wait(job.job_id, 5)
    assert jobs.start("n2", "m2", []).job_id != job.job_id


def test_to_dict_shape(monkeypatch):
    monkeypatch.setattr(swap_jobs, "run_argv", lambda a, cb: 0)
    jobs = SwapJobs()
    job = jobs.start("n", "m", [])
    jobs.wait(job.job_id, 5)
    d = job.to_dict()
    assert set(d) == {
        "job_id",
        "name",
        "model",
        "status",
        "exit_code",
        "output",
        "started_at",
        "finished_at",
    }
    for k in ("started_at", "finished_at"):
        assert datetime.fromisoformat(d[k]).utcoffset() == UTC.utcoffset(None)
    assert set(job.summary()) == {"job_id", "name", "status", "started_at"}


def test_build_run_argv():
    assert build_run_argv("k") == [sys.executable, "-m", "llmctl.main", "run", "k", "--yes"]


def test_run_argv_survives_invalid_utf8():
    import sys

    from llmctl.services import swap_jobs

    lines: list[str] = []
    code = swap_jobs.run_argv(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'a\\xff\\xfeb\\n')"],
        lines.append,
    )
    assert code == 0
    assert len(lines) == 1 and lines[0].startswith("a") and lines[0].endswith("b")


def test_output_keeps_the_tail(monkeypatch):
    """The error and rollback lines come last; the cap must not drop them."""
    monkeypatch.setattr(swap_jobs, "MAX_OUTPUT_LINES", 3)

    def fake(argv, on_line):
        for n in range(10):
            on_line(f"line {n}")
        return 1

    monkeypatch.setattr(swap_jobs, "run_argv", fake)
    jobs = SwapJobs()
    job = jobs.start("n", "m", ["z"])
    jobs.wait(job.job_id, 5)
    assert job.to_dict()["output"] == ["line 7", "line 8", "line 9"]


def test_a_thread_that_will_not_start_does_not_block_later_swaps(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    jobs = SwapJobs()
    with pytest.raises(RuntimeError):
        jobs.start("n", "m", ["z"])
    assert jobs.running() is None
    (job,) = jobs._jobs.values()
    assert job.status == "failed" and "can't start" in job.output[-1]


def test_a_stop_holds_off_new_swaps_and_refuses_while_one_runs(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(swap_jobs, "run_argv", lambda argv, on_line: gate.wait(5) and 0)
    jobs = SwapJobs()
    with jobs.stopping():
        with pytest.raises(swap_jobs.StopInProgress):
            jobs.start("n", "m", ["z"])
    job = jobs.start("n", "m", ["z"])  # free again once the stop is done
    try:
        with pytest.raises(JobBusy), jobs.stopping():
            pass
    finally:
        gate.set()
    jobs.wait(job.job_id, 5)
    assert job.status == "succeeded"
