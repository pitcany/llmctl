"""GPU inventory for sizing plans, read through llmctl's NVML telemetry."""

from __future__ import annotations

from dataclasses import dataclass

from llmctl.telemetry.gpu import get_gpu_info


@dataclass
class GPU:
    index: int
    name: str
    vram_mib: int

    @property
    def vram_gib(self) -> float:
        return self.vram_mib / 1024.0


@dataclass
class HardwareInfo:
    gpus: list[GPU]

    @property
    def total_vram_gib(self) -> float:
        return sum(g.vram_gib for g in self.gpus)

    @property
    def count(self) -> int:
        return len(self.gpus)


def detect_hardware() -> HardwareInfo:
    """GPUs NVML reports; an empty list on a host without NVIDIA GPUs."""
    return HardwareInfo(
        gpus=[
            GPU(index=g.index, name=g.name, vram_mib=g.memory_total_mb)
            for g in get_gpu_info()
            if g.memory_total_mb
        ]
    )
