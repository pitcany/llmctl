"""vLLM runtime adapter.

vLLM is launched as a child process (``vllm serve ...``) exposing an
OpenAI-compatible server. Model discovery scans the configured Hugging Face
style model roots. Process control is delegated to the shared supervisor.
"""

from __future__ import annotations

from llmctl.adapters._common import ProcessRuntimeAdapter
from llmctl.config import RuntimeConfig, default_runtime_configs
from llmctl.db import RuntimeName
from llmctl.telemetry.process import ProcessSupervisor


class VLLMAdapter(ProcessRuntimeAdapter):
    """Adapter for the vLLM runtime."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.VLLM,
            "vLLM",
            config or default_runtime_configs()["vllm"],
            supervisor,
            filesystem_discovery=True,
        )
