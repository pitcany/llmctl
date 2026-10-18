"""``llmctl run`` front end: the picker, the preview, and the lane path.

The vLLM path still runs through :func:`start_vllm_tp` in ``cli.py``; this
module adds what is shared by both paths (list, preview, confirmation) and
the lane swap itself. Kept out of ``cli.py`` the same way ``cli_registry``
is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from llmctl.config import load_settings
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
    cards_label,
    format_duration,
    last_load_s,
    resolve,
)
from llmctl.services.session_swap import (
    pause_for_swap,
    resume_after_failed_swap,
)
from llmctl.services.session_swap import (
    run_session_swap as run_session_swap_service,
)


def render_table(state: SwapState, console: Console) -> None:
    """Numbered swap list: model, engine, key, cards, state, last load time."""
    table = Table(
        title="Models you can run",
        title_justify="left",
        caption="llmctl run NAME  (or: llmctl run)",
    )
    name_width = max((len(r.key) for r in state.catalog), default=4)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Engine", no_wrap=True)
    table.add_column("Name", no_wrap=True, min_width=max(name_width, 4))
    table.add_column("Cards", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Last load", no_wrap=True)
    for index, row in enumerate(state.catalog, start=1):
        status = state.state_of(row)
        style = {"serving": "green", "loading": "yellow"}.get(status, "")
        table.add_row(
            str(index),
            escape(row.model),
            row.engine,
            escape(row.key),
            cards_label(row),
            f"[{style}]{status}[/{style}]" if style else "—",
            format_duration(last_load_s(row.key)) if last_load_s(row.key) else "—",
        )
    console.print(table)


def rows_json(state: SwapState) -> list[dict[str, Any]]:
    """The swap list as plain data, in catalog order (``llmctl ls --json``)."""
    return [
        {
            "number": index,
            "name": row.key,
            "model": row.model,
            "engine": row.engine,
            "kind": row.kind,
            "cards": cards_label(row),
            "state": state.state_of(row),
            "last_load_s": last_load_s(row.key),
        }
        for index, row in enumerate(state.catalog, start=1)
    ]


def pick(state: SwapState, console: Console) -> SwapTarget:
    """Show the list and ask for a number. TTY only; Enter cancels."""
    render_table(state, console)
    if not sys.stdin.isatty():
        console.print(
            "[red]No model given[/red] and stdin is not a terminal. "
            "Run `llmctl run NAME` with a name from the table."
        )
        raise typer.Exit(2)
    answer = typer.prompt(
        f"Run # (1-{len(state.catalog)}, Enter to cancel)",
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
        f"[bold]Run {escape(target.model)}[/bold] ({target.engine}, `{escape(target.key)}`)"
    )
    for line in preview.lines():
        console.print(f"  {escape(line)}")
    return not (preview.already_serving and target.kind in ("lane", "session"))


@dataclass
class Paused:
    """What a vLLM swap stopped up front, to put back if vLLM fails."""

    lanes: list[Lane] = field(default_factory=list)
    sessions: list[Any] = field(default_factory=list)


def stop_lanes_for_vllm(state: SwapState, console: Console) -> Paused:
    """Stop every active lane, and llama.cpp processes llmctl started, before vLLM.

    vLLM takes both GPUs, so anything else on them makes it fail to load.
    Returns what was stopped (restarted if vLLM fails). Raises
    ``typer.Exit(1)`` if something will not stop, having restarted what
    already did.
    """
    paused = Paused()
    if state.gpu_sessions:  # vLLM takes every card, so all of them are in the way
        vllm_row = SwapTarget(key="", kind="vllm", model="", engine="vLLM", upstream="")
        paused.sessions, error = pause_for_swap(
            state, vllm_row, settings=load_settings(), log=lambda m: console.print(escape(m))
        )
        if error:
            console.print(f"[red]{escape(error)}; not starting vLLM.[/red]")
            restore_lanes(paused, console)
            raise typer.Exit(1)
    deps = LaneSwapDeps()
    for lane in state.active_lanes():
        assert lane.unit is not None
        console.print(f"stopping {escape(lane.unit)} ...")
        if stop_user_unit(lane.unit, deps):
            paused.lanes.append(lane)
            continue
        console.print(f"[red]Could not stop {escape(lane.unit)}; not starting vLLM.[/red]")
        restore_lanes(paused, console)
        raise typer.Exit(1)
    return paused


def restore_lanes(paused: Paused | list[Lane], console: Console) -> None:
    """Restart what a failed vLLM swap had stopped: lanes, then llama.cpp processes."""
    if isinstance(paused, list):
        paused = Paused(lanes=paused)
    result = LaneSwapResult(ok=False)
    if paused.lanes:
        restore_previous(
            paused.lanes, restart_vllm=False, vllm_unit="", deps=LaneSwapDeps(), result=result
        )
    if paused.sessions:
        again = resume_after_failed_swap(
            paused.sessions, settings=load_settings(), log=lambda m: console.print(escape(m))
        )
        result.restored += again.restored
        result.restore_failed += again.restore_failed
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
    sessions: list[Any] = []
    if state.sessions_in_the_way(target):
        sessions, error = pause_for_swap(
            state, target, settings=load_settings(), log=lambda m: console.print(escape(m))
        )
        if error:
            console.print(f"[red]Swap failed:[/red] {escape(error)}")
            restore_lanes(Paused(sessions=sessions), console)
            raise typer.Exit(1)
    result = swap_to_lane(
        target.lane,
        other_active=state.lanes_in_the_way(target),
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
    if sessions:
        restore_lanes(Paused(sessions=sessions), console)
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


def run_session_swap(
    state: SwapState,
    target: SwapTarget,
    console: Console,
    parameters: dict | None = None,
) -> None:
    """Free the GPUs and start a llama.cpp model; exit 1 after rolling back."""
    deps = LaneSwapDeps(log=lambda m: console.print(escape(m)))
    result = run_session_swap_service(
        state, target, settings=load_settings(), deps=deps, parameters=parameters
    )
    if result.ok:
        session = result.session
        sid = (getattr(session, "id", "") or "")[:8]
        url = getattr(session, "endpoint_url", None) or "its port"
        console.print(
            f"[green]Ready[/green] — {escape(target.model)} answers at {escape(str(url))} "
            f"({format_duration(result.seconds)})."
        )
        console.print(
            f"  Stop it: llmctl stop {sid}   ·   go back to what was serving: "
            f"llmctl run {escape(_previous_preset(state) or 'PRESET')}"
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


def _previous_preset(state: SwapState) -> str | None:
    """The vLLM preset that was serving before the swap, if any."""
    return next((r.key for r in state.catalog if r.kind == "vllm" and r.key in state.serving), None)
