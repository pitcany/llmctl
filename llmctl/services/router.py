"""Runtime routing service.

Maps runtime identifiers to fully configured adapter instances, injecting the
shared :class:`ProcessSupervisor` and per-runtime configuration derived from
settings.
"""

from __future__ import annotations

from llmctl.adapters.base import RuntimeAdapter
from llmctl.adapters.llama_cpp import LlamaCppAdapter
from llmctl.adapters.lmstudio import LMStudioAdapter
from llmctl.adapters.ollama import OllamaAdapter
from llmctl.adapters.python_script import PythonScriptAdapter
from llmctl.adapters.vllm import VLLMAdapter
from llmctl.config import Settings, load_settings
from llmctl.db import RuntimeName
from llmctl.telemetry.process import ProcessSupervisor


class RuntimeRouter:
    """Maps runtime names to configured adapter instances."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.supervisor = supervisor or ProcessSupervisor(
            log_dir=self.settings.logs_dir / "sessions"
        )
        runtime_config = self.settings.runtime_config
        self.adapters: dict[RuntimeName, RuntimeAdapter] = {
            RuntimeName.VLLM: VLLMAdapter(runtime_config("vllm"), self.supervisor),
            RuntimeName.LLAMA_CPP: LlamaCppAdapter(runtime_config("llama_cpp"), self.supervisor),
            RuntimeName.LMSTUDIO: LMStudioAdapter(runtime_config("lmstudio").endpoint),
            RuntimeName.OLLAMA: OllamaAdapter(runtime_config("ollama").endpoint),
            RuntimeName.PYTHON_SCRIPT: PythonScriptAdapter(
                runtime_config("python_script"), self.supervisor
            ),
        }

    def get_adapter(self, runtime: RuntimeName) -> RuntimeAdapter:
        """Return adapter for runtime, raising KeyError when unsupported."""
        return self.adapters[runtime]

    def list_runtimes(self) -> list[RuntimeName]:
        """Return supported runtime identifiers."""
        return list(self.adapters.keys())
