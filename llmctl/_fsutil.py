"""Filesystem helpers shared across llmctl.

:func:`atomic_write_text` exists because several files llmctl writes are
load-bearing (the ``vllm-tp`` env file a systemd unit reads on restart,
preset YAMLs, the gateway alias overlay). A plain ``Path.write_text``
truncates first, so a crash or a full disk mid-write leaves an empty or
half-written file behind — which the next ``systemctl restart`` then
dutifully launches. Writing a sibling temp file and ``os.replace``-ing it
into place means readers only ever see the old bytes or the new bytes.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``).

    * Symlinks are followed, so a canonical-dir symlink pointing at a
      legacy preset keeps being a symlink and the *target* gets the new
      bytes (``os.replace`` on the link path would swap the link for a
      regular file).
    * An existing file's permission bits are preserved; a new file gets
      the usual ``0o666 & ~umask``.
    * The temp file lives in the destination directory so the rename
      never crosses a filesystem boundary.
    """
    target = Path(path)
    if target.is_symlink():
        target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
