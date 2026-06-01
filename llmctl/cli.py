"""Typer CLI skeleton for LLM Mission Control."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from sqlmodel import Session

from llmctl.api.app import create_app
from llmctl.config import load_settings
from llmctl.db import get_engine, init_db
from llmctl.schemas import BenchmarkRunRequest, ModelCreate, SessionStartRequest
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.preset_loader import load_preset_views
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import SessionService
from llmctl.services.vllm_orchestrator import (
    OrchestratorOptions,
    UnknownPresetError,
    start_slot,
    start_vllm_tp,
)
from llmctl.telemetry.gpu import get_gpu_info

app = typer.Typer(
    name="llmctl",
    help="Local-first mission control scaffold for LLM runtimes.",
    no_args_is_help=True,
)
console = Console()

# Registry/profile management commands (CRUD, preview, export/import) live in
# their own module to keep cli.py focused on launch/orchestration. ``register``
# attaches sub-typers (``model``, ``profile``) plus top-level commands
# (``profiles``, ``preview``, ``export-registry``, ``import-registry``).
from llmctl import cli_registry  # noqa: E402

cli_registry.register(app)


def _parse_gpus(gpus: str | None, cpu: bool) -> tuple[list[int], str, bool]:
    """Parse a ``--gpus`` value into (gpu_ids, mode, allow_cpu).

    Modes: ``auto``, ``balanced``, ``most-free``, ``least-used``, ``cpu``, or an
    explicit comma-separated id list (e.g. ``0,1``) which yields mode
    ``explicit``.
    """
    value = (gpus or "auto").strip().lower()
    if cpu or value == "cpu":
        return [], "cpu", True
    if value in {"auto", "balanced", "most-free", "least-used"}:
        return [], value, False
    ids = [int(part) for part in value.split(",") if part.strip()]
    return ids, "explicit", False


def _session() -> Session:
    settings = load_settings()
    init_db(settings.database_url)
    return Session(get_engine(settings.database_url))


@app.command()
def scan(
    do_import: Annotated[
        bool,
        typer.Option(
            "--import/--dry-run",
            help="Persist discovered models (default: --dry-run only previews).",
        ),
    ] = False,
) -> None:
    """Scan configured model directories and runtime registries.

    Default behavior is a dry run that lists discovered candidates without
    touching the registry. Pass ``--import`` to register everything that was
    found.
    """
    with _session() as db:
        service = RegistryService(db)
        if do_import:
            models = service.scan()
            console.print(
                f"[bold cyan]Scan + import complete.[/bold cyan] "
                f"{len(models)} models currently registered."
            )
            return
        discovered = service.scan_discovered_only()
    if not discovered:
        console.print("[yellow]No new models discovered.[/yellow]")
        return
    table = Table(title=f"Discovered models ({len(discovered)})")
    table.add_column("Name", style="cyan")
    table.add_column("Backend")
    table.add_column("Path")
    table.add_column("Format")
    for model in discovered:
        table.add_row(
            model.name,
            model.runtime.value,
            model.path or "",
            model.format or "",
        )
    console.print(table)
    console.print(
        "[dim]Re-run with --import to register these into the registry.[/dim]"
    )


@app.command("models")
def models_cmd() -> None:
    """List registered models."""
    with _session() as db:
        models = RegistryService(db).list_models()
    table = Table(title="Models")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Runtime")
    table.add_column("Status")
    for model in models:
        table.add_row(model.id or "", model.name, model.runtime.value, model.status.value)
    console.print(table)


@app.command()
def gpus() -> None:
    """Show NVIDIA GPU telemetry."""
    gpus_info = get_gpu_info()
    if not gpus_info:
        console.print(
            "[yellow]No NVIDIA GPU telemetry available or "
            "NVML could not initialize.[/yellow]"
        )
        return
    table = Table(title="GPUs")
    table.add_column("Idx")
    table.add_column("Name")
    table.add_column("Memory (used/total)")
    table.add_column("Util")
    table.add_column("Temp")
    table.add_column("Power")
    table.add_column("Procs")
    for gpu in gpus_info:
        memory = "unknown"
        if gpu.memory_used_mb is not None and gpu.memory_total_mb is not None:
            memory = f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB"
        util = (
            "unknown"
            if gpu.utilization_gpu_percent is None
            else f"{gpu.utilization_gpu_percent}%"
        )
        temp = "n/a" if gpu.temperature_c is None else f"{gpu.temperature_c}C"
        power = "n/a" if gpu.power_draw_watts is None else f"{gpu.power_draw_watts:.0f}W"
        table.add_row(
            str(gpu.index),
            gpu.name,
            memory,
            util,
            temp,
            power,
            str(len(gpu.processes)),
        )
    console.print(table)


@app.command("sessions")
def sessions_cmd() -> None:
    """List runtime sessions."""
    with _session() as db:
        sessions = SessionService(db).list_sessions()
    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Runtime")
    table.add_column("Status")
    table.add_column("Model")
    for session in sessions:
        table.add_row(
            session.id or "",
            session.runtime.value,
            session.status.value,
            session.model_id or "",
        )
    console.print(table)


@app.command("add-model")
def add_model(
    name: Annotated[str, typer.Option(help="Display name for the model.")],
    runtime: Annotated[
        str,
        typer.Option(help="Runtime: vllm, llama_cpp, lmstudio, ollama, python_script"),
    ],
    path: Annotated[Path | None, typer.Option(help="Optional local path.")] = None,
    source: Annotated[str | None, typer.Option(help="Optional runtime/source identifier.")] = None,
    estimated_vram: Annotated[
        float | None, typer.Option(help="Estimated VRAM (GB) for scheduling.")
    ] = None,
) -> None:
    """Register a model record."""
    payload = ModelCreate(
        name=name,
        runtime=runtime,
        path=str(path) if path else None,
        source=source,
        estimated_vram_gb=estimated_vram,
    )
    with _session() as db:
        model = RegistryService(db).add_model(payload)
    console.print(f"[green]Registered model[/green] {model.name} ({model.id})")


@app.command("delete-model")
def delete_model(model_id: Annotated[str, typer.Argument(help="Model ID to soft-delete.")]) -> None:
    """Soft-delete a model record."""
    with _session() as db:
        deleted = RegistryService(db).delete_model(model_id)
    if deleted:
        console.print(f"[green]Soft-deleted model[/green] {model_id}")
    else:
        raise typer.BadParameter(f"Model not found: {model_id}")


def _build_start_request(
    db: Session,
    model_id: str,
    profile: str | None,
    gpus: str,
    cpu: bool,
    force: bool,
    dry_run: bool,
) -> SessionStartRequest:
    """Resolve model/profile and build a :class:`SessionStartRequest`."""
    from llmctl.db import ModelRecord
    from llmctl.services.profiles import ProfileService

    gpu_ids, mode, allow_cpu = _parse_gpus(gpus, cpu)
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise typer.BadParameter(f"Model not found: {model_id}")
    profile_id: str | None = None
    if profile:
        resolved = ProfileService(db).get_by_name(profile)
        if resolved is None:
            raise typer.BadParameter(f"Profile not found: {profile}")
        profile_id = resolved.id
    return SessionStartRequest(
        model_id=model_id,
        profile_id=profile_id,
        runtime=model.runtime,
        gpu_ids=gpu_ids,
        gpu_mode=mode,
        gpus_auto=mode in {"auto", "balanced", "most-free", "least-used"},
        allow_cpu=allow_cpu,
        force=force,
        dry_run=dry_run,
    )


def _print_plan_warnings(plan: object) -> None:
    """Print any warnings and refusal reasons attached to a launch plan."""
    for warning in getattr(plan, "warnings", []) or []:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for reason in getattr(plan, "refusal_reasons", []) or []:
        console.print(f"[red]refusal:[/red] {reason}")


@app.command()
def start(
    model_id: Annotated[str, typer.Argument(help="Model ID to launch.")],
    profile: Annotated[
        str | None, typer.Option(help="Profile name (see `llmctl profiles`).")
    ] = None,
    gpus: Annotated[
        str,
        typer.Option(
            help="GPU selection: auto, balanced, most-free, least-used, cpu, or a list like '0,1'."
        ),
    ] = "auto",
    cpu: Annotated[bool, typer.Option(help="Force CPU-only mode (hide GPUs).")] = False,
    force: Annotated[bool, typer.Option(help="Launch even when safety checks refuse.")] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Plan only; do not launch a process.")
    ] = False,
) -> None:
    """Launch a model session (or plan it with --dry-run)."""
    from llmctl.services.scheduler import SchedulerError

    with _session() as db:
        request = _build_start_request(db, model_id, profile, gpus, cpu, force, dry_run)
        try:
            session = SessionService(db).start(request)
        except SchedulerError as exc:
            raise typer.BadParameter(str(exc)) from exc
    if session.launch_plan is not None:
        _print_plan_warnings(session.launch_plan)
    if session.status.value == "running":
        console.print(
            f"[green]Started session[/green] {session.id} "
            f"pid={session.pid} -> {session.endpoint_url}"
        )
    elif session.status.value == "planned":
        console.print(
            f"[cyan]Planned session[/cyan] {session.id} "
            f"({session.runtime.value}); no process launched."
        )
    else:
        console.print(
            f"[red]Session {session.id} {session.status.value}[/red]: {session.error}"
        )


@app.command()
def plan(
    model_id: Annotated[str, typer.Argument(help="Model ID to plan a launch for.")],
    profile: Annotated[str | None, typer.Option(help="Profile name.")] = None,
    gpus: Annotated[
        str,
        typer.Option(
            help="GPU selection: auto, balanced, most-free, least-used, cpu, or a list like '0,1'."
        ),
    ] = "auto",
    cpu: Annotated[bool, typer.Option(help="Plan CPU-only mode (hide GPUs).")] = False,
) -> None:
    """Print the launch plan for a model without launching anything."""
    with _session() as db:
        request = _build_start_request(db, model_id, profile, gpus, cpu, force=False, dry_run=True)
        launch_plan = SessionService(db).plan(request)

    table = Table(title="Launch Plan", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Model", f"{launch_plan.model_name or launch_plan.model_id} ")
    table.add_row("Backend", launch_plan.runtime.value)
    table.add_row("Profile", launch_plan.profile_name or "-")
    table.add_row("GPU mode", launch_plan.gpu_selection_mode)
    table.add_row("Selected GPUs", ",".join(map(str, launch_plan.gpu_ids)) or "cpu")
    table.add_row("Tensor parallel", str(launch_plan.tensor_parallel_size))
    table.add_row("Port", str(launch_plan.port or "-"))
    table.add_row("Endpoint", launch_plan.endpoint_url or "-")
    est = (
        "unknown"
        if launch_plan.estimated_vram_gb is None
        else f"{launch_plan.estimated_vram_gb:.1f} GB"
    )
    free = "n/a" if launch_plan.free_vram_gb is None else f"{launch_plan.free_vram_gb:.1f} GB"
    table.add_row("Estimated VRAM", est)
    table.add_row("Free VRAM", free)
    table.add_row("Command", launch_plan.command_preview)
    console.print(table)
    _print_plan_warnings(launch_plan)
    if launch_plan.refusal_reasons:
        console.print(
            "[red]This launch would be refused (use --force on `start` to override).[/red]"
        )
    else:
        console.print("[green]This launch is allowed.[/green]")


@app.command()
def doctor() -> None:
    """Report backend binaries, GPU telemetry, and scheduler configuration."""
    from llmctl.services.backends import detect_backends
    from llmctl.telemetry.gpu import nvml_available

    settings = load_settings()
    backends = detect_backends(settings)
    table = Table(title="Backend Binaries")
    table.add_column("Backend")
    table.add_column("Binary")
    table.add_column("Available")
    table.add_column("Path")
    for entry in backends:
        available = "[green]yes[/green]" if entry["available"] else "[red]no[/red]"
        table.add_row(
            str(entry["backend"]),
            str(entry["binary"]),
            available,
            str(entry["path"] or "-"),
        )
    console.print(table)

    gpus_info = get_gpu_info()
    console.print(
        f"[bold]GPUs:[/bold] {len(gpus_info)}  "
        f"[bold]NVML:[/bold] {nvml_available()}  "
        f"[bold]Safe mode:[/bold] {settings.app.safe_mode}"
    )
    sched = settings.scheduler
    console.print(
        f"[bold]Scheduler:[/bold] policy={sched.gpu_policy} "
        f"safety_margin={sched.safety_margin_gb}GB "
        f"public_bind={sched.allow_public_bind} default_host={sched.default_host}"
    )
    missing = [b["backend"] for b in backends if not b["available"]]
    if missing:
        console.print(f"[yellow]Missing backends:[/yellow] {', '.join(missing)}")


@app.command()
def cleanup(
    remove_stale: Annotated[
        bool, typer.Option(help="Delete stopped/failed session records.")
    ] = False,
) -> None:
    """Detect dead sessions, free their ports, and optionally purge stale records."""
    with _session() as db:
        report = SessionService(db).cleanup(remove_stale=remove_stale)
    console.print(
        f"[green]Cleanup complete.[/green] "
        f"dead_marked={report['dead_marked']} "
        f"stale_removed={report['stale_removed']} "
        f"active_remaining={report['active_remaining']}"
    )
    freed = report["freed_ports"]
    if freed:
        console.print(f"[cyan]Freed ports:[/cyan] {', '.join(map(str, freed))}")


@app.command()
def stop(session_id: Annotated[str, typer.Argument(help="Session ID to stop.")]) -> None:
    """Mark a session stopped safely."""
    with _session() as db:
        session = SessionService(db).stop(session_id)
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(f"[green]Session marked stopped[/green] {session_id}")


@app.command()
def restart(session_id: Annotated[str, typer.Argument(help="Session ID to restart-plan.")]) -> None:
    """Plan a safe session restart."""
    with _session() as db:
        session = SessionService(db).restart(session_id)
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(f"[cyan]Restart planned[/cyan] {session_id}; no process launched.")


@app.command()
def logs(
    session_id: Annotated[
        str | None, typer.Argument(help="Session ID to tail logs for.")
    ] = None,
    lines: Annotated[int, typer.Option(help="Number of log lines / events to show.")] = 50,
) -> None:
    """Tail a session's log file, or show recent events when no ID is given."""
    from llmctl.services.events import list_events

    if session_id:
        with _session() as db:
            content = SessionService(db).tail_log(session_id, lines=lines)
        if content is None:
            raise typer.BadParameter(f"Session not found: {session_id}")
        if not content:
            console.print("[yellow]No log output for this session yet.[/yellow]")
        else:
            console.print(content)
        return

    with _session() as db:
        events = list_events(db, limit=lines)
    if not events:
        console.print("[yellow]No events recorded yet.[/yellow]")
        return
    table = Table(title=f"Recent Events (latest {lines})")
    table.add_column("Time")
    table.add_column("Level")
    table.add_column("Category")
    table.add_column("Message")
    for event in events:
        table.add_row(
            event.created_at.isoformat(timespec="seconds"),
            event.level.value,
            event.category,
            event.message,
        )
    console.print(table)


@app.command()
def profiles() -> None:
    """List available launch profiles."""
    from llmctl.services.profiles import ProfileService

    with _session() as db:
        items = ProfileService(db).list_profiles()
    table = Table(title="Profiles")
    table.add_column("Name")
    table.add_column("Runtime")
    table.add_column("Description")
    for item in items:
        table.add_row(item.name, item.runtime.value, item.description or "")
    console.print(table)


@app.command()
def health() -> None:
    """Show overall and per-runtime health."""
    from llmctl.services.health import HealthService

    data = HealthService(load_settings()).get_health()
    console.print(
        f"[bold]State:[/bold] {data['state']}  "
        f"[bold]Safe mode:[/bold] {data['safe_mode']}  "
        f"[bold]GPUs:[/bold] {data['gpu_count']}  "
        f"[bold]NVML:[/bold] {data['nvml_available']}"
    )
    table = Table(title="Runtime Health")
    table.add_column("Runtime")
    table.add_column("State")
    table.add_column("Message")
    for name, info in data["runtimes"].items():
        table.add_row(name, info["state"], info["message"])
    console.print(table)


@app.command()
def bench(
    name: Annotated[str, typer.Option(help="Benchmark name.")] = "smoke",
    model_id: Annotated[str | None, typer.Option(help="Model ID to benchmark.")] = None,
    session_id: Annotated[str | None, typer.Option(help="Running session ID to target.")] = None,
    prompt: Annotated[
        list[str] | None, typer.Option(help="Prompt to send (repeatable).")
    ] = None,
    max_tokens: Annotated[int, typer.Option(help="Max completion tokens per prompt.")] = 64,
    concurrency: Annotated[
        int, typer.Option(help="Parallel in-flight requests (load level).")
    ] = 1,
    sweep: Annotated[
        str | None,
        typer.Option(help="Concurrency sweep, e.g. '1,2,4,8' (overrides --concurrency)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Synthetic mock run vs. real execution against a live endpoint.",
        ),
    ] = True,
) -> None:
    """Benchmark a model: real execution when reachable, else a mock fallback."""
    sweep_levels = (
        [int(part) for part in sweep.split(",") if part.strip()] if sweep else []
    )
    payload = BenchmarkRunRequest(
        name=name,
        model_id=model_id,
        session_id=session_id,
        prompts=list(prompt) if prompt else [],
        parameters={"max_tokens": max_tokens},
        concurrency=concurrency,
        sweep=sweep_levels,
        dry_run=dry_run,
    )
    if sweep_levels:
        with _session() as db:
            results = BenchmarkService(db).run_sweep(payload)
        table = Table(title=f"Benchmark sweep: {name}")
        table.add_column("Concurrency", style="cyan")
        table.add_column("Mode")
        table.add_column("Throughput")
        table.add_column("TTFT")
        for item in results:
            level = item.parameters.get("concurrency", "?")
            mode = str(item.parameters.get("mode", "?"))
            tps = "n/a" if item.tokens_per_second is None else f"{item.tokens_per_second:.1f} tok/s"
            ttft = (
                "n/a"
                if item.time_to_first_token_ms is None
                else f"{item.time_to_first_token_ms:.1f} ms"
            )
            table.add_row(str(level), mode, tps, ttft)
        console.print(table)
        return
    with _session() as db:
        result = BenchmarkService(db).run(payload)
    mode = str(result.parameters.get("mode", "?"))
    reason = result.parameters.get("reason")
    table = Table(title=f"Benchmark: {result.name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    mode_color = "green" if mode == "live" else "yellow"
    table.add_row("Mode", f"[{mode_color}]{mode}[/{mode_color}]")
    if reason:
        table.add_row("Reason", str(reason))
    table.add_row("Concurrency", str(result.parameters.get("concurrency", 1)))
    table.add_row("Prompt tokens", str(result.prompt_tokens))
    table.add_row("Completion tokens", str(result.completion_tokens))
    table.add_row("Total tokens", str(result.total_tokens))
    latency = "n/a" if result.latency_ms is None else f"{result.latency_ms:.1f} ms"
    tps = "n/a" if result.tokens_per_second is None else f"{result.tokens_per_second:.1f} tok/s"
    table.add_row("Latency", latency)
    table.add_row("Throughput", tps)
    ttft = (
        "n/a"
        if result.time_to_first_token_ms is None
        else f"{result.time_to_first_token_ms:.1f} ms"
    )
    table.add_row("Time to first token", ttft)
    console.print(table)
    if mode == "mock":
        console.print(
            "[yellow]Mock fallback used[/yellow] (no live runtime). "
            "Run with [bold]--no-dry-run[/bold] against a started session for real metrics."
        )


@app.command()
def tui() -> None:
    """Launch the Textual TUI skeleton."""
    from llmctl.tui.app import MissionControlApp

    MissionControlApp().run()


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind host; defaults to settings.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port; defaults to settings.")] = None,
) -> None:
    """Serve the FastAPI scaffold."""
    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.api.host,
        port=port or settings.api.port,
    )


@app.command("generate-systemd")
def generate_systemd(
    user: Annotated[bool, typer.Option(help="Generate a user service unit.")] = True,
    output: Annotated[
        Path | None, typer.Option(help="Write the unit to this file instead of stdout.")
    ] = None,
) -> None:
    """Generate a systemd unit for the Mission Control API."""
    from llmctl.services.systemd import render_api_unit

    unit = render_api_unit(load_settings(), user=user)
    _emit_unit(unit, output)


@app.command("generate-systemd-session")
def generate_systemd_session(
    session_id: Annotated[str, typer.Argument(help="Session ID to persist.")],
    user: Annotated[bool, typer.Option(help="Generate a user service unit.")] = True,
    output: Annotated[
        Path | None, typer.Option(help="Write the unit to this file instead of stdout.")
    ] = None,
) -> None:
    """Generate a systemd unit that relaunches a session on boot."""
    from llmctl.services.systemd import render_session_unit

    with _session() as db:
        session = SessionService(db).get_session(session_id)
    if session is None:
        raise typer.BadParameter(f"Session not found: {session_id}")
    unit = render_session_unit(session, user=user)
    _emit_unit(unit, output)


@app.command("install-systemd")
def install_systemd(
    session_id: Annotated[
        str | None, typer.Option(help="Install a session unit instead of the API unit.")
    ] = None,
    all_sessions: Annotated[
        bool, typer.Option("--all", help="Install a unit for every active session.")
    ] = False,
    user: Annotated[bool, typer.Option(help="Install as a user service unit.")] = True,
    enable: Annotated[bool, typer.Option(help="Enable and start the unit after writing.")] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Preview the install actions without writing or enabling anything.",
        ),
    ] = True,
) -> None:
    """Write and enable a systemd unit for the API or a session (safe dry-run default)."""
    from llmctl.services.systemd import (
        install_unit,
        render_api_unit,
        render_session_unit,
    )

    def _install(unit: object) -> None:
        for warning in unit.warnings:  # type: ignore[attr-defined]
            console.print(f"[yellow]warning:[/yellow] {warning}")
        report = install_unit(unit, dry_run=dry_run, enable=enable)  # type: ignore[arg-type]
        header = "DRY RUN - would run" if report.dry_run else "Install report"
        color = "yellow" if report.dry_run else "green"
        console.print(
            f"[{color}]{header}[/{color}] for {unit.name} -> {report.unit_path}"  # type: ignore[attr-defined]
        )
        for action in report.actions:
            console.print(f"  - {action}")
        for message in report.messages:
            console.print(f"  [cyan]{message}[/cyan]")

    if all_sessions:
        active_states = {"running", "starting", "planned"}
        with _session() as db:
            sessions = SessionService(db).list_sessions()
        active = [s for s in sessions if s.status.value in active_states]
        if not active:
            console.print("[yellow]No active sessions to persist.[/yellow]")
            return
        console.print(f"[cyan]Persisting {len(active)} active session(s).[/cyan]")
        for session in active:
            _install(render_session_unit(session, user=user))
        return

    if session_id:
        with _session() as db:
            session = SessionService(db).get_session(session_id)
        if session is None:
            raise typer.BadParameter(f"Session not found: {session_id}")
        unit = render_session_unit(session, user=user)
    else:
        unit = render_api_unit(load_settings(), user=user)
    _install(unit)



def _emit_unit(unit: object, output: Path | None) -> None:
    """Print a rendered systemd unit (and warnings) or write it to a file."""
    name = getattr(unit, "name", "unit.service")
    content = getattr(unit, "content", "")
    for warning in getattr(unit, "warnings", []) or []:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if output is not None:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Wrote {name}[/green] -> {output}")
        return
    console.print(content)
    console.print("[cyan]Install with:[/cyan]")
    for command in unit.install_commands():  # type: ignore[attr-defined]
        console.print(f"  {command}")


# ---------------------------------------------------------------------------
# Phase 5: gpu-models replacement verbs (vllm, slot, presets, status).
# These wrap the high-level orchestrator in services/vllm_orchestrator.py;
# the orchestrator owns lifecycle (preset load -> TQ -> preflight ->
# adapter restart -> hermes verify) so the CLI is purely arg-parsing +
# result formatting.
# ---------------------------------------------------------------------------


def _build_options(
    tq: bool | None,
    no_tq: bool,
    dry_run: bool,
    no_wait: bool,
) -> OrchestratorOptions:
    """Resolve the tri-state TQ flag and turn CLI flags into options."""
    if no_tq and tq:
        raise typer.BadParameter("--tq and --no-tq are mutually exclusive")
    tq_override: bool | None = None
    if tq:
        tq_override = True
    if no_tq:
        tq_override = False
    return OrchestratorOptions(
        tq_override=tq_override,
        dry_run=dry_run,
        wait_for_ready=not no_wait,
    )


@app.command("vllm")
def vllm_cmd(
    preset: Annotated[
        str,
        typer.Argument(help="Preset alias from ~/.config/llm-models/."),
    ],
    tq: Annotated[
        bool,
        typer.Option("--tq", help="Force TurboQuant KV cache on for this start."),
    ] = False,
    no_tq: Annotated[
        bool,
        typer.Option("--no-tq", help="Force TurboQuant KV cache off for this start."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the env file + planned actions, no changes."),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Skip waiting for /v1/models readiness after restart."),
    ] = False,
) -> None:
    """Start the TP-fleet vLLM unit on PRESET (replaces ``gpu-models vllm``)."""
    settings = load_settings()
    options = _build_options(tq if tq else None, no_tq, dry_run, no_wait)
    try:
        result = start_vllm_tp(
            preset,
            managed_unit=settings.managed_units.vllm_tp,
            defaults=settings.vllm.defaults,
            fleet=settings.managed_units.fleet,
            options=options,
        )
    except UnknownPresetError as exc:
        raise typer.Exit(2) from exc

    if result.dry_run:
        return
    if not result.ok:
        console.print("[red]vLLM start did not complete cleanly.[/red]")
        if result.fleet_failed:
            console.print(f"  fleet preflight failed on: {', '.join(result.fleet_failed)}")
        if result.restart is not None and result.restart.error:
            console.print(f"  restart: {result.restart.error}")
        raise typer.Exit(1)
    console.print(
        f"[green]vLLM ready[/green] — serving {result.spec.served_name} "
        f"on port {result.spec.port}"
    )


@app.command("slot")
def slot_cmd(
    slot: Annotated[
        str,
        typer.Argument(help="Slot name (e.g. coder, reasoner)."),
    ],
    preset: Annotated[
        str,
        typer.Argument(help="Preset alias from ~/.config/llm-models/."),
    ],
    tq: Annotated[
        bool,
        typer.Option("--tq", help="Force TurboQuant KV cache on for this start."),
    ] = False,
    no_tq: Annotated[
        bool,
        typer.Option("--no-tq", help="Force TurboQuant KV cache off for this start."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the env file + planned actions, no changes."),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Skip waiting for /v1/models readiness after restart."),
    ] = False,
) -> None:
    """Apply PRESET to a per-GPU slot (replaces ``gpu-models slot``)."""
    settings = load_settings()
    slot_config = settings.managed_units.slots.get(slot)
    if slot_config is None:
        available = ", ".join(sorted(vars(settings.managed_units.slots)))
        console.print(f"[red]unknown slot[/red] {slot!r}. Available: {available}")
        raise typer.Exit(2)

    options = _build_options(tq if tq else None, no_tq, dry_run, no_wait)
    try:
        result = start_slot(
            slot,
            preset,
            slot_config=slot_config,
            defaults=settings.vllm.defaults,
            fleet=settings.managed_units.fleet,
            options=options,
        )
    except UnknownPresetError as exc:
        raise typer.Exit(2) from exc

    if result.dry_run:
        return
    if not result.ok:
        console.print(f"[red]Slot {slot} start did not complete cleanly.[/red]")
        if result.fleet_failed:
            console.print(f"  fleet preflight failed on: {', '.join(result.fleet_failed)}")
        if result.restart is not None and result.restart.error:
            console.print(f"  restart: {result.restart.error}")
        raise typer.Exit(1)
    console.print(
        f"[green]Slot {slot} ready[/green] — serving as '{slot}' "
        f"(model {result.spec.served_name}) on port {result.spec.port}"
    )


@app.command("presets")
def presets_cmd() -> None:
    """List preset aliases known to llmctl."""
    views = load_preset_views()
    if not views:
        console.print(
            "[yellow]No presets found.[/yellow] "
            "Write one to ~/.config/llmctl/presets/<alias>.yaml"
        )
        return
    table = Table(title="Presets")
    table.add_column("alias", style="cyan")
    table.add_column("served name")
    table.add_column("model id")
    table.add_column("family")
    table.add_column("size (B)", justify="right")
    table.add_column("tp", justify="right")
    table.add_column("quant")
    for v in views:
        table.add_row(
            v.alias,
            v.served_name,
            v.model_id,
            v.family or "-",
            f"{v.param_count_b:.0f}" if v.param_count_b else "-",
            str(v.tensor_parallel),
            v.quantization,
        )
    console.print(table)


@app.command("status")
def status_cmd() -> None:
    """Quick overview of managed units and the presets they could serve."""
    settings = load_settings()
    table = Table(title="Managed units")
    table.add_column("role", style="cyan")
    table.add_column("unit name")
    table.add_column("env file")
    table.add_column("default port", justify="right")
    for role, unit in (
        ("vllm-tp", settings.managed_units.vllm_tp),
        ("vllm-coder", settings.managed_units.vllm_coder),
        ("vllm-reasoner", settings.managed_units.vllm_reasoner),
    ):
        table.add_row(
            role,
            unit.unit_name,
            str(unit.resolve_env_file()),
            str(unit.default_port),
        )
    console.print(table)
    slots = settings.managed_units.slots
    slot_table = Table(title="Slots")
    slot_table.add_column("name", style="cyan")
    slot_table.add_column("gpu", justify="right")
    slot_table.add_column("port", justify="right")
    slot_table.add_column("unit name")
    slot_table.add_column("env file")
    for name in ("coder", "reasoner"):
        cfg = slots.get(name)
        if cfg is None:
            continue
        slot_table.add_row(
            name,
            cfg.gpu,
            str(cfg.port),
            cfg.unit_name,
            str(cfg.resolve_env_file(name)),
        )
    console.print(slot_table)


# ---------------------------------------------------------------------------
# OpenAI-compatible router/gateway commands.
#
# The gateway runs as a separate FastAPI app from the control-plane API
# served by `llmctl serve` — different port, different surface, different
# blast radius. `gateway` runs the proxy; `router-status` probes it;
# `aliases` and `set-alias` manage the role -> session mapping the proxy
# uses for the `local-<role>` virtual model ids.
# ---------------------------------------------------------------------------


def _router_url(settings: object) -> str:
    """Compose the local URL the CLI uses to talk to the gateway."""
    router = settings.router  # type: ignore[attr-defined]
    host = router.host if router.host not in ("0.0.0.0", "") else "127.0.0.1"
    return f"http://{host}:{router.port}"


@app.command("gateway")
def gateway_cmd(
    host: Annotated[str | None, typer.Option(help="Bind host; defaults to router.host.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port; defaults to router.port.")] = None,
) -> None:
    """Serve the OpenAI-compatible router gateway."""
    from llmctl.api.gateway import create_gateway_app

    settings = load_settings()
    bind_host = host or settings.router.host
    bind_port = port or settings.router.port

    if bind_host not in ("127.0.0.1", "localhost") and not settings.router.allow_public_bind:
        raise typer.BadParameter(
            f"Refusing to bind gateway to {bind_host}: set router.allow_public_bind=true "
            "in settings.yaml to override. The recommended way to expose the gateway is "
            "via `tailscale serve --bg --https=<port> http://127.0.0.1:<router.port>` "
            "instead of binding the process publicly."
        )

    uvicorn.run(create_gateway_app(settings), host=bind_host, port=bind_port)


@app.command("router-status")
def router_status_cmd() -> None:
    """Probe the gateway and show its current routing view."""
    import httpx

    settings = load_settings()
    url = f"{_router_url(settings)}/health"
    headers = {}
    if settings.router.auth_token:
        headers["Authorization"] = f"Bearer {settings.router.auth_token}"
    try:
        response = httpx.get(url, headers=headers, timeout=2.0)
    except httpx.HTTPError as exc:
        console.print(f"[red]Gateway unreachable[/red] at {url}: {exc}")
        console.print(
            "[cyan]Start it with[/cyan] `llmctl gateway` (default 127.0.0.1:9000)."
        )
        raise typer.Exit(1) from exc

    if response.status_code == 401:
        console.print(
            "[red]Auth required.[/red] Set router.auth_token in settings.yaml or unset it."
        )
        raise typer.Exit(1)
    if response.status_code != 200:
        console.print(f"[red]Gateway returned HTTP {response.status_code}.[/red]")
        raise typer.Exit(1)

    payload = response.json()
    router = payload.get("router", {})
    console.print(
        f"[green]Gateway up[/green] at {url}  "
        f"auth={'on' if router.get('auth_required') else 'off'}  "
        f"auto_start={router.get('auto_start')}  "
        f"fallback={router.get('fallback_policy')}"
    )
    aliases = payload.get("aliases", [])
    table = Table(title="Aliases")
    table.add_column("alias", style="cyan")
    table.add_column("target")
    table.add_column("session")
    table.add_column("served name")
    table.add_column("healthy")
    for entry in aliases:
        healthy = entry.get("healthy")
        color = "green" if healthy else "yellow" if entry.get("target") else "red"
        table.add_row(
            entry.get("name", ""),
            entry.get("target") or "-",
            entry.get("session_id") or "-",
            entry.get("served_name") or "-",
            f"[{color}]{'yes' if healthy else 'no'}[/{color}]",
        )
    console.print(table)


@app.command("aliases")
def aliases_cmd() -> None:
    """List configured router aliases and their resolved targets."""
    from llmctl.services.gateway import GatewayService

    settings = load_settings()
    with _session() as db:
        rows = GatewayService(db, settings).alias_view()

    table = Table(title="Router aliases")
    table.add_column("alias", style="cyan")
    table.add_column("public id")
    table.add_column("target")
    table.add_column("session")
    table.add_column("served name")
    table.add_column("status")
    for row in rows:
        if row.target is None:
            status_text = "[red]unbound[/red]"
        elif row.healthy:
            status_text = "[green]healthy[/green]"
        else:
            status_text = "[yellow]bound, no active session[/yellow]"
        table.add_row(
            row.name,
            f"local-{row.name}",
            row.target or "-",
            row.resolved_session_id or "-",
            row.resolved_served_name or "-",
            status_text,
        )
    console.print(table)


@app.command("set-alias")
def set_alias_cmd(
    alias: Annotated[str, typer.Argument(help="Alias role (e.g. coding, reasoning).")],
    target: Annotated[
        str | None,
        typer.Argument(
            help="Session id, profile name, or served model name to bind. "
            "Pass '-' to clear the alias.",
        ),
    ] = None,
) -> None:
    """Bind ALIAS to a session/profile/served name (overlay JSON, not YAML)."""
    from llmctl.services.gateway import GatewayService

    settings = load_settings()
    resolved_target = None if target in (None, "-") else target
    with _session() as db:
        service = GatewayService(db, settings)
        service.set_alias(alias, resolved_target)
        rows = {row.name: row for row in service.alias_view()}
    if resolved_target is None:
        console.print(f"[yellow]Cleared alias[/yellow] {alias}.")
    else:
        view = rows.get(alias)
        if view is None or not view.healthy:
            console.print(
                f"[yellow]Bound[/yellow] {alias} -> {resolved_target}; "
                "no active session resolves yet (will activate when a matching session starts)."
            )
        else:
            console.print(
                f"[green]Bound[/green] {alias} -> {resolved_target} "
                f"(session {view.resolved_session_id}, served as {view.resolved_served_name})."
            )
