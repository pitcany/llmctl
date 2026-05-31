"""Arbitrary Python launch-script runtime adapter.

Launches a user-provided Python entry point as a supervised child process. There
is no automatic model discovery for this runtime; models are registered
explicitly with the path to the launch script.
"""

from __future__ import annotations

from llmctl.adapters._common import ProcessRuntimeAdapter
from llmctl.config import RuntimeConfig, default_runtime_configs
from llmctl.db import RuntimeName
from llmctl.telemetry.process import ProcessSupervisor


class PythonScriptAdapter(ProcessRuntimeAdapter):
    """Adapter for arbitrary Python launch scripts."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.PYTHON_SCRIPT,
            "Python script",
            config or default_runtime_configs()["python_script"],
            supervisor,
            filesystem_discovery=False,
        )
