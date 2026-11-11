"""A user-scope unit must be asked of the user manager, not the system one."""

from __future__ import annotations

from types import SimpleNamespace

from llmctl.services.lanes import gather_state
from llmctl.services.unit_gpus import unit_gpu_ids


def test_gather_state_asks_for_the_vllm_unit_in_its_configured_scope() -> None:
    asked: list[tuple[str, bool]] = []

    def is_active(unit: str, user: bool) -> bool:
        asked.append((unit, user))
        return user  # running, but only the user manager knows it

    settings = SimpleNamespace(
        managed_units=SimpleNamespace(
            vllm_engine=SimpleNamespace(
                unit_name="vllm-engine", default_port=8003, user_scope=True
            )
        )
    )
    state = gather_state(
        settings,
        presets={},
        lanes=[],
        probe=lambda url: [],
        is_active=is_active,
        gpu_sessions=[],
        session_rows=lambda catalog: [],
    )
    assert ("vllm-engine", True) in asked
    assert ("vllm-engine", False) not in asked
    assert state.vllm_active is True


def _systemd(*, system_pid: str, user_pid: str, calls: list[list[str]]):
    def run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            stdout=user_pid if "--user" in argv else system_pid, returncode=0
        )

    return run


def test_unit_gpu_ids_falls_back_to_the_user_manager() -> None:
    calls: list[list[str]] = []
    ids = unit_gpu_ids(
        "llama.service",
        run=_systemd(system_pid="0\n", user_pid="4242\n", calls=calls),
        read_environ=lambda pid: "CUDA_VISIBLE_DEVICES=1\0" if pid == 4242 else None,
    )
    assert ids == [1]


def test_unit_gpu_ids_system_unit_is_not_asked_of_the_user_manager() -> None:
    calls: list[list[str]] = []
    ids = unit_gpu_ids(
        "vllm-tp.service",
        run=_systemd(system_pid="77\n", user_pid="4242\n", calls=calls),
        read_environ=lambda pid: "CUDA_VISIBLE_DEVICES=0,1\0" if pid == 77 else None,
    )
    assert ids == [0, 1]
    assert not any("--user" in argv for argv in calls)


def test_unit_gpu_ids_reads_the_user_units_environment_file(tmp_path) -> None:
    env = tmp_path / "llama.env"
    env.write_text('CUDA_VISIBLE_DEVICES="1"\n', encoding="utf-8")
    calls: list[list[str]] = []

    def run(argv, **_kwargs):
        calls.append(list(argv))
        user = "--user" in argv
        if "--property=MainPID" in argv:
            return SimpleNamespace(stdout="4242\n" if user else "0\n", returncode=0)
        return SimpleNamespace(
            stdout=f"{env} (ignore_errors=no)\n" if user else "", returncode=0
        )

    assert unit_gpu_ids("llama.service", run=run, read_environ=lambda pid: None) == [1]
