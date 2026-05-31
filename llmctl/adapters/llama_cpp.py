"""llama.cpp runtime adapter.

llama.cpp is launched as a child process (``llama-server -m model.gguf ...``).
Model discovery scans the configured roots for ``*.gguf`` files. Process control
is delegated to the shared supervisor.
"""

from __future__ import annotations

from llmctl.adapters._common import ProcessRuntimeAdapter
from llmctl.config import RuntimeConfig, default_runtime_configs
from llmctl.db import RuntimeName
from llmctl.telemetry.process import ProcessSupervisor


class LlamaCppAdapter(ProcessRuntimeAdapter):
    """Adapter for the llama.cpp server runtime."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.LLAMA_CPP,
            "llama.cpp",
            config or default_runtime_configs()["llama_cpp"],
            supervisor,
            filesystem_discovery=True,
        )
