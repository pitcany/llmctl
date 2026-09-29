"""`llmctl recommend`: measured routing advice from the model lab (~/AI/lab).

Read-only. The lab (`modellab run ...`) writes lab/recommendations.json after
every run; this command shows it per role together with the `llmctl swap`
command that serves the recommended target. Nothing here changes routing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


def lab_dir() -> Path:
    return Path(os.environ.get("MODELLAB_DIR", Path.home() / "AI" / "lab"))


def _swap_names() -> dict[str, str]:
    try:
        raw = yaml.safe_load((lab_dir() / "targets.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {n: s["swap"] for n, s in (raw.get("targets") or {}).items() if s.get("swap")}


def recommend_cmd(
    role: Annotated[
        str | None, typer.Argument(help="Role from lab/roles.yaml (default: all).")
    ] = None,
) -> None:
    """Measured per-role model advice from the model lab (read-only; run `modellab`)."""
    path = lab_dir() / "recommendations.json"
    if not path.exists():
        console.print(
            f"No lab results at {escape(str(path))}. Run `modellab run <target>` first "
            "(see ~/AI/lab/README.md)."
        )
        raise typer.Exit(1)
    data = json.loads(path.read_text())
    roles = [r for r in data.get("roles", []) if role in (None, r.get("role"))]
    if not roles:
        console.print(f"No role named {escape(str(role))} in {escape(str(path))}.")
        raise typer.Exit(1)
    swaps = _swap_names()
    table = Table(title=f"Model lab advice (generated {data.get('generated', '?')})")
    for col in ("role", "think", "routed today", "measured best", "serve it with", "advice"):
        table.add_column(col, overflow="fold")
    for r in roles:
        best = r.get("best") or "—"
        serve = f"llmctl swap {swaps[best]}" if best in swaps else "—"
        table.add_row(
            escape(r["role"]), escape(str(r.get("think"))), escape(str(r.get("current"))),
            escape(best), escape(serve), escape(r.get("advice", "")),
        )
    console.print(table)
    for d in data.get("redundant", []):
        console.print(
            f"[yellow]redundant?[/] {escape(d['redundant'])} (think {escape(d['think'])}) "
            f"is never measurably better than {escape(d['given'])}"
        )


def register(app: typer.Typer) -> None:
    app.command("recommend")(recommend_cmd)
