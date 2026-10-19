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
from llmctl.services.swap_view import (  # noqa: F401 - re-exported for callers
    _aliases_for,
    gateway_aliases,
    gateway_url,
    rows_json,
)


def render_table(state: SwapState, console: Console, *, command: str = "run") -> None:
    """Numbered swap list: model, engine, key, cards, state, last load time.

    ``command`` is the verb the caption suggests: the list is also the
    `llmctl settings` picker, where "llmctl run NAME" would be the wrong hint.
    """
    table = Table(
        title="Models you can run",
        title_justify="left",
        caption=f"llmctl {command} NAME  (or: llmctl {command})",
    )
    rows = [
        (
            str(index),
            row.model,
            row.engine,
            row.key,
            cards_label(row),
            state.state_of(row),
            format_duration(last_load_s(row.key)) if last_load_s(row.key) else "—",
        )
        for index, row in enumerate(state.catalog, start=1)
    ]
    headers = ("#", "Model", "Engine", "Name", "Cards", "State", "Last load")
    # Every column but Model keeps its full width: Name is what `llmctl run`
    # takes, so it must never be cut. Model gets what is left and is cut on
    # one line (left to Rich, it either folds into a narrow strip that makes
    # each row several lines tall, or squeezes `#` and Cards to nothing).
    # `ls --json` has the whole model.
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    chrome = 3 * len(headers) + 1  # a border and two padding cells per column
    model_width = max(4, console.width - chrome - sum(widths) + widths[1])
    for i, header in enumerate(headers):
        if header == "Model":
            table.add_column(header, no_wrap=True, overflow="ellipsis", max_width=model_width)
        else:
            table.add_column(
                header, no_wrap=True, min_width=widths[i], justify="right" if i == 0 else "left"
            )
    for index, row, (*_, status, load) in zip(
        range(1, len(rows) + 1), state.catalog, rows, strict=True
    ):
        style = {"serving": "green", "loading": "yellow"}.get(status, "")
        table.add_row(
            str(index),
            escape(row.model),
            row.engine,
            escape(row.key),
            cards_label(row),
            f"[{style}]{status}[/{style}]" if style else "—",
            load,
        )
    console.print(table)


def render_endpoints(
    state: SwapState, console: Console, aliases: list[Any] | None, router: Any
) -> None:
    """Under the table: where each serving row answers, and its gateway ids."""
    for row in state.catalog:
        serving = state.state_of(row) == "serving"
        if not serving or not row.upstream:
            continue
        console.print(
            f"{escape(row.key)} → {escape(row.upstream + '/v1')}  (model {escape(row.model)})"
        )
        names = _aliases_for(row, True, aliases)
        if names:
            url = gateway_url(router)
            console.print(f"  gateway: {escape(url)} as {escape(', '.join(names))}")


def pick(
    state: SwapState, console: Console, *, command: str = "run", prompt: str = "Run"
) -> SwapTarget:
    """Show the list and ask for a number. TTY only; Enter cancels.

    ``command`` names the verb in the caption and the non-TTY hint, so
    `llmctl settings` with no NAME does not tell the user to type
    `llmctl run NAME`.
    """
    render_table(state, console, command=command)
    if not sys.stdin.isatty():
        console.print(
            "[red]No model given[/red] and stdin is not a terminal. "
            f"Run `llmctl {command} NAME` with a name from the table."
        )
        raise typer.Exit(2)
    answer = typer.prompt(
        f"{prompt} # (1-{len(state.catalog)}, Enter to cancel)",
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
    if preview.already_serving:
        url = _serving_endpoint(state, target)
        if url:
            console.print(f"  answers at {escape(url)}")
    return not (preview.already_serving and target.kind in ("lane", "session"))


def _serving_endpoint(state: SwapState, target: SwapTarget) -> str | None:
    """Where an already-serving lane or session answers, when known."""
    if target.kind == "lane":
        return f"{target.upstream}/v1" if target.upstream else None
    if target.kind == "session":
        # The running process's own port wins over the one its profile pins.
        port = next((s.port for s in state.gpu_sessions if s.name == target.key), None)
        port = port or target.port
        if port:
            return f"http://127.0.0.1:{port}/v1"
    return None


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
