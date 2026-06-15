"""Unit tests for systemd-unit GPU-pinning introspection (``services.unit_gpus``)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from llmctl.services.unit_gpus import parse_cuda_visible_devices, unit_gpu_ids


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0,1", [0, 1]),
        ("1", [1]),
        ("", []),
        ("0, 1 , 2", [0, 1, 2]),  # tolerate whitespace
        ("GPU-abc,0", [0]),  # drop UUID-form ids, keep the integer
        ("0,GPU-x,3", [0, 3]),
        ("-1", []),  # the "disable" sentinel is not a bare index
        ("3,3", [3, 3]),  # order/dupes preserved verbatim (no policy here)
    ],
)
def test_parse_cuda_visible_devices(raw: str, expected: list[int]) -> None:
    assert parse_cuda_visible_devices(raw) == expected


def _fake_run(stdout: str):
    """Build a ``subprocess.run`` stand-in that returns ``stdout``."""

    def run(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout, returncode=0)

    return run


def test_unit_gpu_ids_reads_cuda_from_main_pid_environ() -> None:
    environ = "PATH=/usr/bin\0CUDA_VISIBLE_DEVICES=0,1\0HOME=/home/yannik"
    ids = unit_gpu_ids(
        "vllm-tp.service",
        run=_fake_run("1234\n"),
        read_environ=lambda pid: environ if pid == 1234 else None,
    )
    assert ids == [0, 1]


def test_unit_gpu_ids_none_unit_name() -> None:
    assert unit_gpu_ids(None) == []


def test_unit_gpu_ids_unit_not_running_main_pid_zero() -> None:
    # systemd reports MainPID=0 for an inactive unit.
    called = {"environ": False}

    def read_environ(_pid: int) -> str | None:
        called["environ"] = True
        return ""

    assert (
        unit_gpu_ids("vllm-tp.service", run=_fake_run("0\n"), read_environ=read_environ)
        == []
    )
    assert called["environ"] is False  # never tried to read a pid-0 environ


def test_unit_gpu_ids_systemctl_absent_returns_empty() -> None:
    def run(*_args, **_kwargs):
        raise FileNotFoundError("systemctl not installed")

    assert unit_gpu_ids("vllm-tp.service", run=run, read_environ=lambda _p: "") == []


def test_unit_gpu_ids_systemctl_timeout_returns_empty() -> None:
    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=2.0)

    assert unit_gpu_ids("vllm-tp.service", run=run, read_environ=lambda _p: "") == []


def test_unit_gpu_ids_environ_unreadable_returns_empty() -> None:
    assert (
        unit_gpu_ids(
            "vllm-tp.service", run=_fake_run("1234"), read_environ=lambda _p: None
        )
        == []
    )


def test_unit_gpu_ids_var_absent_in_environ_returns_empty() -> None:
    environ = "PATH=/usr/bin\0HOME=/home/yannik"
    assert (
        unit_gpu_ids(
            "vllm-tp.service", run=_fake_run("1234"), read_environ=lambda _p: environ
        )
        == []
    )


def test_unit_gpu_ids_non_numeric_main_pid_returns_empty() -> None:
    # Defensive: a malformed `systemctl show` value must not crash.
    assert (
        unit_gpu_ids(
            "vllm-tp.service", run=_fake_run("not-a-pid"), read_environ=lambda _p: ""
        )
        == []
    )
