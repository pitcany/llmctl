"""``llmctl swap`` front end: the picker, the preview, and the lane path.

The vLLM path still runs through :func:`start_vllm_tp` in ``cli.py``; this
module adds what is shared by both paths (list, preview, confirmation) and
the lane swap itself. Kept out of ``cli.py`` the same way ``cli_registry``
is.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from llmctl.services.lane_swap import (
    LaneSwapDeps,
    LaneSwapResult,
    restore_previous,
    stop_user_unit,
    swap_to_lane,
)
from llmctl.services.lanes import (
    Lane,
    SwapState,
    SwapTarget,
    format_duration,
    last_load_s,
    resolve,
)


def render_table(state: SwapState, console: Console) -> None:
    """Numbered swap list: model, engine, key, state, last load time."""
    table = Table(
        title="Swap targets", title_justify="left", caption="llmctl swap NAME  (or: llmctl swap)"
    )
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Last load")
    for index, row in enumerate(state.catalog, start=1):
        status = state.state_of(row)
        style = {"serving": "green", "loading": "yellow"}.get(status, "")
        table.add_row(
            str(index),
            escape(row.model),
            row.engine,
            escape(row.key),
            f"[{style}]{status}[/{style}]" if style else "—",
            format_duration(last_load_s(row.key)) if last_load_s(row.key) else "—",
        )
    console.print(table)


def pick(state: SwapState, console: Console) -> SwapTarget:
    """Show the list and ask for a number. TTY only; Enter cancels."""
    render_table(state, console)
    if not sys.stdin.isatty():
        console.print(
            "[red]No model given[/red] and stdin is not a terminal. "
            "Run `llmctl swap NAME` with a name from the table."
        )
        raise typer.Exit(2)
    answer = typer.prompt(
        f"Swap to # (1-{len(state.catalog)}, Enter to cancel)",
        default="",
        show_default=False,
    ).strip()
    if not answer:
        console.print("[yellow]Cancelled; nothing changed.[/yellow]")
        raise typer.Exit(0)
    if answer.isdigit() and 1 <= int(answer) <= len(state.catalog):
        return state.catalog[int(answer) - 1]
    target = resolve(answer, state.catalog)
    if target is None:
        console.print(f"[red]No such entry:[/red] {escape(answer)}")
        raise typer.Exit(2)
    return target


def print_preview(state: SwapState, target: SwapTarget, console: Console) -> bool:
    """Print what the swap will do. Returns False when there is nothing to do."""
    preview = state.preview(target)
    console.print(
        f"[bold]Swap to {escape(target.model)}[/bold] ({target.engine}, `{escape(target.key)}`)"
    )
    for line in preview.lines():
        console.print(f"  {escape(line)}")
    return not (preview.already_serving and target.kind == "lane")


def stop_lanes_for_vllm(state: SwapState, console: Console) -> list[Lane]:
    """Stop every active lane unit before vLLM takes both GPUs.

    Returns the lanes that were stopped (to restart if vLLM fails). Raises
    ``typer.Exit(1)`` if one will not stop, having restarted the ones that
    already did — vLLM would OOM next to it.
    """
    deps = LaneSwapDeps()
    stopped: list[Lane] = []
    for lane in state.active_lanes():
        assert lane.unit is not None
        console.print(f"stopping {escape(lane.unit)} ...")
        if stop_user_unit(lane.unit, deps):
            stopped.append(lane)
            continue
        console.print(f"[red]Could not stop {escape(lane.unit)}; not starting vLLM.[/red]")
        restore_lanes(stopped, console)
        raise typer.Exit(1)
    return stopped


def restore_lanes(lanes: list[Lane], console: Console) -> None:
    """Restart lanes a failed vLLM swap had stopped."""
    if not lanes:
        return
    result = LaneSwapResult(ok=False)
    restore_previous(lanes, restart_vllm=False, vllm_unit="", deps=LaneSwapDeps(), result=result)
    if result.restored:
        console.print(
            f"  [green]rollback: started again:[/green] {escape(', '.join(result.restored))}"
        )
    if result.restore_failed:
        console.print(
            f"  [red]rollback: could not start:[/red] {escape(', '.join(result.restore_failed))}"
        )


def run_lane_swap(state: SwapState, target: SwapTarget, console: Console) -> None:
    """Swap onto a lane; exit 1 on failure after rolling back."""
    assert target.lane is not None
    result = swap_to_lane(
        target.lane,
        other_active=state.active_lanes(exclude=target.key),
        vllm_active=state.vllm_active,
        vllm_unit=state.vllm_unit,
    )
    if result.ok:
        console.print(
            f"[green]Ready[/green] — {escape(target.model)} on {escape(target.upstream)} "
            f"({format_duration(result.seconds)})"
        )
        return
    console.print(f"[red]Swap failed:[/red] {escape(result.error or 'unknown error')}")
    if result.restored:
        console.print(
            f"  [green]rolled back:[/green] {escape(', '.join(result.restored))} serving again"
        )
    if result.restore_failed:
        console.print(
            f"  [red]not restored:[/red] {escape(', '.join(result.restore_failed))} "
            "(commands printed above)"
        )
    raise typer.Exit(1)
