"""NVIDIA GPU telemetry helpers."""

from __future__ import annotations

from typing import Any

from llmctl.schemas import GPUInfo


def nvml_available() -> bool:
    """Return True when NVML can be initialized on this host.

    Safe on non-NVIDIA hosts: any import or initialization error yields False.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            return True
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def get_gpu_info() -> list[GPUInfo]:
    """Return NVIDIA GPU telemetry using NVML when available.

    The function is safe on non-NVIDIA hosts and returns an empty list when NVML
    cannot initialize. Future phases can add richer process correlation.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8", errors="replace")
            count = pynvml.nvmlDeviceGetCount()
            gpus: list[GPUInfo] = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name: str | bytes = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                uuid: str | bytes | None = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", errors="replace")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temperature = None
                try:
                    power_draw_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power_draw_watts = None
                processes: list[dict[str, Any]] = []
                try:
                    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        processes.append(
                            {
                                "pid": proc.pid,
                                "used_memory_mb": proc.usedGpuMemory // 1048576,
                            }
                        )
                except Exception:
                    processes = []
                gpus.append(
                    GPUInfo(
                        index=index,
                        uuid=uuid,
                        name=str(name),
                        driver_version=str(driver_version),
                        memory_total_mb=memory.total // 1048576,
                        memory_used_mb=memory.used // 1048576,
                        memory_free_mb=memory.free // 1048576,
                        utilization_gpu_percent=util.gpu,
                        utilization_memory_percent=util.memory,
                        temperature_c=temperature,
                        power_draw_watts=power_draw_watts,
                        processes=processes,
                    )
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []
