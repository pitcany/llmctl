"""Rolling NVML sampler for benchmark runs.

The :class:`GPUSampler` context manager spawns a background thread that polls
NVML at a configurable interval (default 500 ms) for memory and utilisation
counters. On exit it aggregates the samples into peak VRAM, average and peak
GPU utilisation, and sample count -- the numbers shown in benchmark records.

The sampler is safe on hosts without NVML / pynvml available: it silently
records zero samples and :meth:`summary` returns ``None`` aggregates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

#: Default polling interval in seconds; the spec calls for 250-500ms.
DEFAULT_INTERVAL_S = 0.5


@dataclass
class GPUSample:
    """A single NVML sample across all visible GPUs."""

    timestamp: float
    memory_used_mb: int
    utilization_gpu_pct: int
    per_gpu: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GPUSamplerSummary:
    """Aggregated counters from a sampler run."""

    sample_count: int
    peak_vram_mb: int | None
    avg_gpu_util_pct: float | None
    max_gpu_util_pct: float | None


class GPUSampler:
    """Background NVML poller used as a context manager.

    Usage::

        with GPUSampler() as sampler:
            do_benchmark()
        summary = sampler.summary()

    Stopping the sampler is idempotent and always joins the thread.
    """

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = interval_s
        self._samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    def __enter__(self) -> GPUSampler:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        """Spawn the polling thread if NVML is available; otherwise no-op."""
        if self._thread is not None:
            return
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name="llmctl-gpu-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to exit and join it."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 3 + 1.0)
        self._stopped_at = time.perf_counter()
        self._thread = None

    @property
    def samples(self) -> list[GPUSample]:
        """Return the captured samples (a defensive copy)."""
        return list(self._samples)

    def summary(self) -> GPUSamplerSummary:
        """Aggregate the captured samples into the benchmark metrics."""
        if not self._samples:
            return GPUSamplerSummary(
                sample_count=0,
                peak_vram_mb=None,
                avg_gpu_util_pct=None,
                max_gpu_util_pct=None,
            )
        peak_vram = max(sample.memory_used_mb for sample in self._samples)
        utils = [sample.utilization_gpu_pct for sample in self._samples]
        return GPUSamplerSummary(
            sample_count=len(self._samples),
            peak_vram_mb=peak_vram,
            avg_gpu_util_pct=round(sum(utils) / len(utils), 2),
            max_gpu_util_pct=float(max(utils)),
        )

    # -- polling loop -------------------------------------------------------

    def _run(self) -> None:
        """Poll NVML until stop is signalled; swallow all NVML errors silently.

        We intentionally do not log here -- the sampler is best-effort and
        runs alongside the user-visible benchmark; surfacing NVML quirks
        would only add noise.
        """
        try:
            import pynvml  # type: ignore[import-not-found]
        except Exception:
            return
        try:
            pynvml.nvmlInit()
        except Exception:
            return
        try:
            while not self._stop.is_set():
                try:
                    sample = self._capture(pynvml)
                except Exception:
                    sample = None
                if sample is not None:
                    self._samples.append(sample)
                self._stop.wait(self.interval_s)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _capture(pynvml: Any) -> GPUSample | None:
        """Capture a sample across all visible GPUs.

        Memory used is summed across GPUs; utilization is the *max* across
        GPUs (the busiest device dominates throughput on tensor-parallel
        deployments). Per-GPU detail is retained for debugging.
        """
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        total_used = 0
        max_util = 0
        per_gpu: list[dict[str, Any]] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            used_mb = int(memory.used // 1048576)
            gpu_util = int(util.gpu)
            total_used += used_mb
            max_util = max(max_util, gpu_util)
            per_gpu.append(
                {
                    "index": index,
                    "memory_used_mb": used_mb,
                    "memory_total_mb": int(memory.total // 1048576),
                    "utilization_gpu_pct": gpu_util,
                    "utilization_memory_pct": int(util.memory),
                }
            )
        return GPUSample(
            timestamp=time.time(),
            memory_used_mb=total_used,
            utilization_gpu_pct=max_util,
            per_gpu=per_gpu,
        )
