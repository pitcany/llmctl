"""One producer for the managed-unit table.

``llmctl status`` and the TUI Units screen answer the same question — what units
are configured, where their env files are, and what each is serving right now —
and they used to answer it with separate code. They drifted: the screen carried
a private copy of the ``/v1/models`` probe that collapsed the tri-state, and it
kept rendering env-file paths that do not exist after the CLI stopped.

Anything that appears on more than one surface belongs here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmctl.config import Settings
from llmctl.services.backends import probe_openai_v1_models

#: Per-unit probe budget. The TUI refreshes every 3s and probes each unit, so
#: this bounds how far behind a down unit can push a refresh.
PROBE_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class UnitRow:
    """One managed unit, as both the CLI and the TUI describe it."""

    role: str
    unit_name: str
    env_file: Path
    #: ``resolve_env_file()`` always returns a path, even for a role llmctl
    #: never writes an env file for. Rendering one that is not there reads as
    #: "edit this file" for a file nothing reads, so callers must check.
    env_file_exists: bool
    port: int
    #: Tri-state, and the distinction is the point: a list (possibly empty)
    #: means the endpoint answered; ``None`` means it could not be reached.
    #: An idle unit and an unreachable one are different facts, and conflating
    #: them is what the operator pays for.
    served: list[str] | None

    @property
    def serving(self) -> bool:
        """True when the endpoint answered at all, loaded or not."""
        return self.served is not None


def unit_rows(
    settings: Settings,
    *,
    http_get: Callable[[str, float], Any] | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> list[UnitRow]:
    """Describe every configured managed unit, probing each one's port.

    Args:
        settings: Resolved settings; ``managed_units.roles()`` drives the rows.
        http_get: Transport override for tests. Production uses the default in
            :mod:`llmctl.services.backends`.
        timeout_s: Per-probe timeout.
    """
    rows: list[UnitRow] = []
    for role, unit in settings.managed_units.roles().items():
        served = probe_openai_v1_models(
            f"http://127.0.0.1:{unit.default_port}", timeout_s, http_get
        )
        env_file = unit.resolve_env_file()
        rows.append(
            UnitRow(
                role=role,
                unit_name=unit.unit_name,
                env_file=env_file,
                env_file_exists=env_file.exists(),
                port=unit.default_port,
                served=served,
            )
        )
    return rows
