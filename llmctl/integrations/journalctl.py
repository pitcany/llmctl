"""Thin wrapper around ``journalctl`` for reading adopted-unit logs.

Adopted sessions point at externally managed systemd units (e.g.
``vllm-tp.service``); llmctl never owns a log file for them, so the unit's
journal is the only log surface. This module is the single chokepoint for
``journalctl`` calls — one place to mock in tests, one place to audit.

Read-only by design: only ``journalctl -u <unit>`` tail queries, never
``--vacuum*``/``--rotate``/``--flush``. No ``sudo``: the invoking user's own
journal permissions apply (system-journal access needs membership in
``adm``/``systemd-journal``; a permission problem surfaces as a journalctl
hint on stderr, which callers pass through to the user).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

#: Bounded so a wedged journald can't hang ``llmctl logs``/the TUI forever.
#: Journal tail reads are fast; 30s is already generous under heavy IO.
_TIMEOUT_S = 30.0

#: journalctl prints this (exit 0) when a filter matches nothing; callers
#: should treat it as "no output", not as log content.
NO_ENTRIES_MARKER = "-- No entries --"


@dataclass(frozen=True)
class JournalResult:
    """Result of a single ``journalctl`` invocation."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """``True`` when the command exited with status 0."""
        return self.returncode == 0

    @property
    def has_entries(self) -> bool:
        """``True`` when the query succeeded and produced real log lines."""
        text = self.stdout.strip()
        return self.ok and bool(text) and text != NO_ENTRIES_MARKER


class JournalctlRunner:
    """Run read-only ``journalctl`` tail queries for a single unit.

    Inject ``runner`` for tests — any callable taking ``list[str]`` and
    returning a :class:`subprocess.CompletedProcess` works.
    """

    def __init__(
        self,
        runner: object | None = None,
        *,
        journalctl_bin: str = "journalctl",
    ) -> None:
        self._runner = runner
        self.journalctl_bin = journalctl_bin

    def available(self) -> bool:
        """``True`` when ``journalctl`` is on PATH (false in most containers)."""
        return shutil.which(self.journalctl_bin) is not None

    def tail_unit(self, unit: str, *, lines: int = 50, user: bool = False) -> JournalResult:
        """Return the last ``lines`` journal entries for ``unit``.

        ``user`` switches to the caller's user-scope journal (``--user``) for
        ``systemctl --user`` units; the default reads the system journal.
        Never raises on non-zero exit — caller inspects ``.ok``.
        """
        argv: list[str] = [self.journalctl_bin]
        if user:
            argv.append("--user")
        argv.extend(["-u", unit, "-n", str(max(1, lines)), "--no-pager", "-o", "short-iso"])
        if self._runner is not None:
            completed = self._runner(argv)  # type: ignore[misc]
        else:
            try:
                completed = subprocess.run(  # noqa: S603 - argv from constants + unit name
                    argv,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                # Bare phrase, no command echo: callers prefix their own
                # "journalctl -u <unit> failed:" context when reporting.
                return JournalResult(
                    returncode=124,
                    stdout="",
                    stderr=f"timed out after {int(_TIMEOUT_S)}s",
                )
        return JournalResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
