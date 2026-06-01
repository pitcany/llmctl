"""Tests for the rolling NVML sampler used during benchmark runs.

The real :mod:`pynvml` module is not available on most CI hosts and not safe
to call when no NVIDIA device is present. These tests install a fake module
into :mod:`sys.modules` so the sampler's thread can be exercised without
touching real hardware.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

from llmctl.telemetry import gpu_sampler


class _FakeMemory:
    def __init__(self, used_mb: int, total_mb: int = 49152) -> None:
        self.used = used_mb * 1024 * 1024
        self.total = total_mb * 1024 * 1024


class _FakeUtil:
    def __init__(self, gpu_pct: int, memory_pct: int = 0) -> None:
        self.gpu = gpu_pct
        self.memory = memory_pct


def _make_fake_pynvml(samples: list[tuple[int, int]]):
    """Return a fake pynvml module whose readings come from ``samples``.

    Each entry in ``samples`` is ``(memory_used_mb, gpu_util_pct)`` for the
    single visible GPU; the iterator cycles when exhausted so the sampler
    can keep polling for the duration of the test.
    """
    iterator = iter(samples)
    sentinel = samples[-1] if samples else (0, 0)

    def next_sample() -> tuple[int, int]:
        nonlocal sentinel
        try:
            sentinel = next(iterator)
        except StopIteration:
            pass
        return sentinel

    module = types.ModuleType("pynvml")

    def nvmlInit() -> None:  # noqa: N802 - mirror NVML API
        return None

    def nvmlShutdown() -> None:  # noqa: N802
        return None

    def nvmlDeviceGetCount() -> int:  # noqa: N802
        return 1

    def nvmlDeviceGetHandleByIndex(_idx: int) -> object:  # noqa: N802
        return object()

    def nvmlDeviceGetMemoryInfo(_handle: object) -> _FakeMemory:  # noqa: N802
        used, _ = next_sample()
        return _FakeMemory(used_mb=used)

    def nvmlDeviceGetUtilizationRates(_handle: object) -> _FakeUtil:  # noqa: N802
        _, util = sentinel
        return _FakeUtil(gpu_pct=util)

    module.nvmlInit = nvmlInit
    module.nvmlShutdown = nvmlShutdown
    module.nvmlDeviceGetCount = nvmlDeviceGetCount
    module.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    module.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
    module.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
    return module


@pytest.fixture
def fake_pynvml(monkeypatch):
    """Install a fake pynvml that emits a known sequence of readings."""
    fake = _make_fake_pynvml(
        samples=[
            (10000, 10),  # 10 GB, 10%
            (15000, 80),  # peak VRAM + high util
            (12000, 30),
        ]
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    yield fake


def test_sampler_aggregates_peak_and_average(fake_pynvml) -> None:
    """``summary()`` reports peak VRAM, average util, max util across samples."""
    with gpu_sampler.GPUSampler(interval_s=0.05) as sampler:
        time.sleep(0.25)  # ~5 polls
    summary = sampler.summary()
    assert summary.sample_count >= 3
    # peak VRAM is the highest used_mb across the sample window.
    assert summary.peak_vram_mb is not None and summary.peak_vram_mb >= 10000
    assert summary.max_gpu_util_pct is not None and summary.max_gpu_util_pct >= 10
    assert summary.avg_gpu_util_pct is not None
    assert 0 <= summary.avg_gpu_util_pct <= 100


def test_sampler_safe_without_pynvml(monkeypatch) -> None:
    """Sampler is a no-op when pynvml cannot be imported."""
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with gpu_sampler.GPUSampler(interval_s=0.05) as sampler:
        time.sleep(0.1)
    summary = sampler.summary()
    assert summary.sample_count == 0
    assert summary.peak_vram_mb is None
    assert summary.avg_gpu_util_pct is None
    assert summary.max_gpu_util_pct is None


def test_sampler_rejects_non_positive_interval() -> None:
    """``interval_s`` must be > 0 to guard against tight infinite loops."""
    with pytest.raises(ValueError):
        gpu_sampler.GPUSampler(interval_s=0)
    with pytest.raises(ValueError):
        gpu_sampler.GPUSampler(interval_s=-1.0)
