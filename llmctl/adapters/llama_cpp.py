"""llama.cpp runtime adapter.

llama.cpp is launched as a child process (``llama-server -m model.gguf ...``).
Model discovery scans the configured roots for ``*.gguf`` files. Process control
is delegated to the shared supervisor.
"""

from __future__ import annotations

import asyncio
import shutil

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

    def capabilities(self) -> dict[str, bool]:
        caps = super().capabilities()
        caps["version"] = True
        return caps

    async def version(self) -> str | None:
        """Return ``llama-server --version`` output (first line), if runnable."""
        binary = self.config.binary or "llama-server"
        resolved = shutil.which(binary)
        if not resolved:
            return None
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                resolved,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except (TimeoutError, OSError):
            if proc is not None and proc.returncode is None:
                # wait_for only cancels the await; kill the child so a binary
                # that ignores --version doesn't linger as an untracked process.
                proc.kill()
                await proc.wait()
            return None
        for line in out.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                return line.strip()
        return None
