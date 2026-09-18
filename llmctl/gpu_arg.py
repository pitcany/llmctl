"""The one parser for every ``--gpus`` CLI option.

``cli.start/plan`` and ``preview`` used to parse ``--gpus`` separately: one
raised a bare ``ValueError`` traceback on ``0,x`` or a typo'd mode, the other
silently dropped non-digit parts and passed unknown modes through to the
scheduler. Both now go through :func:`parse_gpus`, which rejects bad input
with a ``typer.BadParameter`` (a clean usage error, exit 2).
"""

from __future__ import annotations

import typer

GPU_MODES = ("auto", "balanced", "most-free", "least-used")


def parse_gpus(gpus: str | None, *, cpu: bool = False) -> tuple[list[int], str, bool]:
    """Parse ``--gpus`` into ``(gpu_ids, mode, allow_cpu)``.

    Accepts a named mode (``auto``, ``balanced``, ``most-free``,
    ``least-used``), ``cpu``, or an explicit id list like ``0,1`` / ``0, 1``
    (mode ``explicit``).
    """
    value = (gpus or "auto").strip().lower()
    if cpu or value == "cpu":
        return [], "cpu", True
    if value in GPU_MODES:
        return [], value, False
    parts = [part.strip() for part in value.split(",")]
    if not parts or not all(part.isdigit() for part in parts):
        raise typer.BadParameter(
            f"--gpus {gpus!r}: expected one of {', '.join((*GPU_MODES, 'cpu'))} "
            "or a comma-separated GPU id list like '0,1'."
        )
    ids = [int(part) for part in parts]
    if len(set(ids)) != len(ids):
        raise typer.BadParameter(f"--gpus {gpus!r}: duplicate GPU id.")
    return ids, "explicit", False
