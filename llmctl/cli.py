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
from llmctl.config import ManagedUnitConfig, load_settings
from llmctl.db import (
    BenchmarkKind,
    ModelRecord,
    RuntimeName,
    SessionRecord,
    get_engine,
    init_db,
)
from llmctl.schemas import (
    BenchmarkRunRequest,
    HealthState,
    ModelCreate,
    SessionStartRequest,
)
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.preset_loader import load_preset_views
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import AdoptError, SessionService
from llmctl.services.vllm_orchestrator import (
    OrchestratorOptions,
    UnknownPresetError,
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


def _confirm_state_change(action: str, *, required: bool, assume_yes: bool) -> None:
    """TTY-gated confirmation for a state-changing action.

    Prompts only when the matching ``scheduler.require_confirmation_for_*``
    setting is on, ``--yes`` was not passed, and stdin is a TTY (scripts and
    pipelines are never blocked by a prompt). Declining is a clean exit 0 —
    the user saying "no" is not an error.
    """
    import sys

    if not required or assume_yes or not sys.stdin.isatty():
        return
    if not typer.confirm(f"{action} — continue?"):
        console.print("[yellow]Aborted; nothing changed.[/yellow]")
        raise typer.Exit(0)


def _emit_json(payload: object) -> None:
    """Print machine-readable JSON to stdout with no decoration.

    Uses plain ``print`` (not Rich) so output is stable for pipes/scripts:
    no ANSI codes, no wrapping, keys straight from the schemas.
    """
    import json

    print(json.dumps(payload, indent=2, default=str, sort_keys=False))


_JSON_OPT = Annotated[
    bool, typer.Option("--json", help="Emit machine-readable JSON instead of a table.")
]


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
def models_cmd(json_out: _JSON_OPT = False) -> None:
    """List registered models."""
    with _session() as db:
        models = RegistryService(db).list_models()
    if json_out:
        _emit_json([model.model_dump(mode="json") for model in models])
        return
    table = Table(title="Models")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Runtime")
    table.add_column("Status")
    for model in models:
        table.add_row(model.id or "", model.name, model.runtime.value, model.status.value)
    console.print(table)


@app.command()
def gpus(json_out: _JSON_OPT = False) -> None:
    """Show NVIDIA GPU telemetry."""
    gpus_info = get_gpu_info()
    if json_out:
        _emit_json([gpu.model_dump(mode="json") for gpu in gpus_info])
        return
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
def sessions_cmd(
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh/--no-fresh",
            help="Reconcile liveness (PID + endpoint probes) before listing.",
        ),
    ] = True,
    json_out: _JSON_OPT = False,
) -> None:
    """List runtime sessions."""
    with _session() as db:
        service = SessionService(db)
        if fresh:
            service.reconcile()
        sessions = service.list_sessions()
    if json_out:
        _emit_json([session.model_dump(mode="json") for session in sessions])
        return
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
def delete_model(
    model_id: Annotated[str, typer.Argument(help="Model ID to soft-delete.")],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Soft-delete a model record."""
    settings = load_settings()
    _confirm_state_change(
        f"Soft-delete model {model_id}",
        required=settings.scheduler.require_confirmation_for_delete,
        assume_yes=yes,
    )
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
    from llmctl.services.registry import RegistryService

    gpu_ids, mode, allow_cpu = _parse_gpus(gpus, cpu)
    # Resolve MODEL_ID by name OR id so `llmctl start/plan <name>` works, not
    # just the UUID. Reuses the registry resolver (raises on ambiguous names).
    try:
        found = RegistryService(db).find(model_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if found is None or found.id is None:
        raise typer.BadParameter(f"Model not found: {model_id}")
    model_id = found.id
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
    elif session.status.value == "starting":
        console.print(
            f"[yellow]Session {session.id} is starting[/yellow] pid={session.pid}; "
            "the endpoint is not ready yet (large models load for minutes). "
            "`llmctl sessions` will show it running once it responds."
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
def doctor(json_out: _JSON_OPT = False) -> None:
    """Inspect the environment: storage, runtimes, GPUs, sessions, drift.

    Read-only. Groups results into passed checks, warnings, and failures
    with suggested remediation. Exits 1 when any check fails outright.
    """
    from llmctl.services.doctor import run_doctor

    report = run_doctor(load_settings())
    if json_out:
        _emit_json(report)
        if not report["ok"]:
            raise typer.Exit(code=1)
        return

    for verdict, style, entries in (
        ("PASS", "green", report["passed"]),
        ("WARN", "yellow", report["warnings"]),
        ("FAIL", "red", report["failures"]),
    ):
        for check in entries:
            console.print(f"[{style}]{verdict}[/{style}] {check['name']}: {check['detail']}")
            if check.get("remediation") and verdict != "PASS":
                console.print(f"       ↳ {check['remediation']}")
    console.print(
        f"\n{len(report['passed'])} passed, {len(report['warnings'])} warning(s), "
        f"{len(report['failures'])} failure(s)."
    )
    if not report["ok"]:
        raise typer.Exit(code=1)


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
def reconcile() -> None:
    """Synchronize tracked sessions with reality.

    For OWNED sessions: PID-check; mark STOPPED when the process is gone.
    For ADOPTED sessions: HTTP-probe ``{endpoint}/v1/models``; flip
    RUNNING↔STOPPED based on whether the upstream responds. The gateway
    runs the same pass periodically (``router.reconcile_interval_s``);
    this command is the explicit, on-demand equivalent.
    """
    with _session() as db:
        changed = SessionService(db).reconcile()
    if changed:
        console.print(f"[green]Reconcile complete.[/green] {changed} session(s) transitioned.")
    else:
        console.print("[cyan]Reconcile complete.[/cyan] no changes.")


@app.command()
def stop(
    session_id: Annotated[str, typer.Argument(help="Session ID to stop.")],
    systemd: Annotated[
        bool,
        typer.Option(
            "--systemd",
            "-s",
            help="For an adopted session, stop its backing systemd unit (sudo).",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Mark a session stopped safely."""
    settings = load_settings()
    _confirm_state_change(
        f"Stop session {session_id}" + (" and its systemd unit" if systemd else ""),
        required=settings.scheduler.require_confirmation_for_stop,
        assume_yes=yes,
    )
    with _session() as db:
        try:
            session = SessionService(db).stop(session_id, stop_unit=systemd)
        except AdoptError as exc:
            console.print(f"[red]Stop refused:[/red] {exc}")
            raise typer.Exit(1) from exc
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(f"[green]Session marked stopped[/green] {session_id}")


@app.command()
def restart(session_id: Annotated[str, typer.Argument(help="Session ID to restart-plan.")]) -> None:
    """Plan a safe session restart."""
    with _session() as db:
        try:
            session = SessionService(db).restart(session_id)
        except AdoptError as exc:
            console.print(f"[red]Restart refused:[/red] {exc}")
            raise typer.Exit(1) from exc
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(f"[cyan]Restart planned[/cyan] {session_id}; no process launched.")


@app.command()
def pull(
    model: Annotated[str, typer.Argument(help="Ollama model tag, e.g. qwen3:32b.")],
) -> None:
    """Pull a model into the local Ollama library (streaming progress)."""
    import asyncio

    from llmctl.adapters.ollama import OllamaAdapter
    from llmctl.services.router import RuntimeRouter

    settings = load_settings()
    adapter = RuntimeRouter(settings).get_adapter(RuntimeName.OLLAMA)
    if not isinstance(adapter, OllamaAdapter):  # stripped asserts can't guard
        console.print("[red]The ollama runtime adapter does not support pull.[/red]")
        raise typer.Exit(1)

    # Print each phase once, then byte progress at most every 10% so long
    # pulls stay readable in scripts and terminals alike.
    progress_state: dict[str, object] = {"status": None, "decade": -1}

    def on_progress(status: str, completed: int | None, total: int | None) -> None:
        if status and status != progress_state["status"]:
            progress_state["status"] = status
            progress_state["decade"] = -1
            console.print(f"[cyan]{status}[/cyan]")
        if total and completed is not None:
            decade = min(10, completed * 10 // total)
            if decade > progress_state["decade"]:  # type: ignore[operator]
                progress_state["decade"] = decade
                gib = 1024**3
                console.print(
                    f"  {completed / gib:.2f}/{total / gib:.2f} GiB ({decade * 10}%)"
                )

    status = asyncio.run(adapter.pull_model(model, on_progress=on_progress))
    if status.state != HealthState.OK:
        console.print(f"[red]{status.message}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{status.message}[/green] Run `llmctl scan` to register it.")


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


config_app = typer.Typer(name="config", help="Show and validate llmctl configuration.")
app.add_typer(config_app, name="config")


def _settings_file_path() -> Path:
    """Return the settings.yaml path llmctl reads (may not exist yet)."""
    from llmctl.config import settings_file_path

    return settings_file_path()


@config_app.command("path")
def config_path() -> None:
    """Print the settings file path (whether or not it exists)."""
    path = _settings_file_path()
    exists = "" if path.exists() else "  (not present; defaults in effect)"
    print(f"{path}{exists}")


def _redact_secrets(value: object) -> object:
    """Return a copy of ``value`` with secret-looking fields masked."""
    sensitive = (
        "token",
        "secret",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "auth",
        "bearer",
        "credential",
        "private_key",
        "cookie",
    )
    if isinstance(value, dict):
        return {
            key: (
                "********"
                if any(marker in key.lower() for marker in sensitive)
                and val
                and not isinstance(val, bool)
                else _redact_secrets(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


@config_app.command("show")
def config_show(json_out: _JSON_OPT = False) -> None:
    """Show the fully-resolved settings (defaults + file + env overrides).

    Secret-looking fields (tokens, keys, passwords) are redacted.
    """
    settings = load_settings()
    payload = _redact_secrets(settings.model_dump(mode="json"))
    if json_out:
        _emit_json(payload)
        return
    import yaml

    console.print(f"[dim]# resolved from {_settings_file_path()}[/dim]")
    print(yaml.safe_dump(payload, sort_keys=False))


@config_app.command("validate")
def config_validate() -> None:
    """Validate the settings file; exit 1 with the offending field on error."""
    try:
        load_settings()
    except Exception as exc:
        console.print(f"[red]Configuration invalid:[/red] {_settings_file_path()}")
        console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Configuration valid.[/green] ({_settings_file_path()})")


runtimes_app = typer.Typer(
    name="runtimes",
    help="Inspect configured runtimes: install state, version, capabilities.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(runtimes_app, name="runtimes")


@runtimes_app.callback()
def runtimes_main(ctx: typer.Context, json_out: _JSON_OPT = False) -> None:
    """List every runtime with health, version, endpoint, and loaded models."""
    if ctx.invoked_subcommand is not None:
        # `llmctl runtimes --json inspect X`: carry the group-level flag into
        # the subcommand instead of silently dropping it.
        ctx.obj = {"json": json_out}
        return
    from llmctl.services.runtimes import runtime_inventory

    inventory = runtime_inventory(load_settings())
    if json_out:
        _emit_json(inventory)
        return
    table = Table(title="Runtimes")
    table.add_column("runtime", style="cyan")
    table.add_column("state")
    table.add_column("version")
    table.add_column("endpoint")
    table.add_column("loaded models")
    for row in table_rows_from_inventory(inventory):
        table.add_row(*row)
    console.print(table)
    console.print("[dim]Details: llmctl runtimes inspect <runtime>[/dim]")


def table_rows_from_inventory(inventory: list[dict]) -> list[tuple[str, ...]]:
    """Format inventory rows for the runtimes table (split out for tests)."""
    styles = {"ok": "green", "degraded": "yellow"}
    rows: list[tuple[str, ...]] = []
    for entry in inventory:
        style = styles.get(entry["state"], "red")
        loaded = entry["loaded"]
        rows.append(
            (
                entry["runtime"],
                f"[{style}]{entry['state']}[/{style}]",
                entry["version"] or "-",
                entry["endpoint"] or "-",
                "n/a" if loaded is None else (", ".join(loaded) or "(none)"),
            )
        )
    return rows


@runtimes_app.command("inspect")
def runtimes_inspect(
    ctx: typer.Context,
    runtime: Annotated[str, typer.Argument(help="Runtime name (e.g. ollama, vllm).")],
    json_out: _JSON_OPT = False,
) -> None:
    """Show one runtime's full record, including its capability flags."""
    from llmctl.services.runtimes import runtime_inventory

    json_out = json_out or bool(ctx.obj and ctx.obj.get("json"))
    inventory = runtime_inventory(load_settings())
    match = next((row for row in inventory if row["runtime"] == runtime), None)
    if match is None:
        known = ", ".join(row["runtime"] for row in inventory)
        raise typer.BadParameter(f"Unknown runtime '{runtime}'. Known: {known}")
    if json_out:
        _emit_json(match)
        return
    console.print(f"[bold]{match['display_name']}[/bold] ({match['runtime']})")
    console.print(f"  state:    {match['state']} — {match['message']}")
    console.print(f"  version:  {match['version'] or 'unknown'}")
    console.print(f"  endpoint: {match['endpoint'] or '-'}")
    loaded = match["loaded"]
    console.print(
        "  loaded:   " + ("n/a" if loaded is None else (", ".join(loaded) or "(none)"))
    )
    console.print("  capabilities:")
    for key, supported in match["capabilities"].items():
        mark = "[green]yes[/green]" if supported else "[dim]no[/dim]"
        console.print(f"    {key}: {mark}")


@app.command()
def health(json_out: _JSON_OPT = False) -> None:
    """Show overall and per-runtime health."""
    from llmctl.services.health import HealthService

    data = HealthService(load_settings()).get_health()
    if json_out:
        _emit_json(data)
        return
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


def _resolve_bench_target(
    target: str | None,
    model_id: str | None,
    session_id: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the positional ``target`` to (model_id, session_id).

    Lookup order: session_id (running takes precedence so live sessions are
    convenient to bench) -> model_id. Returns the explicit flag values when
    no positional ``target`` was supplied.
    """
    if target is None:
        return model_id, session_id
    if model_id or session_id:
        raise typer.BadParameter(
            "Pass a positional target OR --model-id/--session-id, not both."
        )
    with _session() as db:
        if db.get(SessionRecord, target) is not None:
            return None, target
        if db.get(ModelRecord, target) is not None:
            return target, None
    raise typer.BadParameter(
        f"Target '{target}' does not match any session or model ID."
    )


@app.command()
def bench(
    target: Annotated[
        str | None,
        typer.Argument(
            help="Model ID or session ID to benchmark (auto-detected).",
        ),
    ] = None,
    name: Annotated[str, typer.Option(help="Benchmark name.")] = "smoke",
    kind: Annotated[
        BenchmarkKind,
        typer.Option(
            help="Benchmark kind: chat (default), completion, health, long_context.",
            case_sensitive=False,
        ),
    ] = BenchmarkKind.CHAT,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile ID to associate with the run."),
    ] = None,
    model_id: Annotated[
        str | None, typer.Option(help="Model ID (alternative to positional target).")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option(help="Session ID (alternative to positional target).")
    ] = None,
    prompt: Annotated[
        list[str] | None, typer.Option(help="Prompt to send (repeatable).")
    ] = None,
    max_tokens: Annotated[int, typer.Option(help="Max completion tokens per prompt.")] = 256,
    context_length: Annotated[
        int | None,
        typer.Option(help="Target context length in tokens (long_context kind)."),
    ] = None,
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
            help="Force a synthetic mock run instead of hitting the live endpoint.",
        ),
    ] = False,
) -> None:
    """Benchmark a model or session against the configured runtime.

    Pass the positional ``TARGET`` as either a model ID or session ID --
    llmctl will look it up automatically. On live failures the result is
    persisted with ``success=False`` for inspection. Use ``--dry-run`` to
    force a synthetic mock run (e.g. on CI hosts without a runtime).
    """
    resolved_model, resolved_session = _resolve_bench_target(
        target, model_id, session_id
    )
    sweep_levels = (
        [int(part) for part in sweep.split(",") if part.strip()] if sweep else []
    )
    payload = BenchmarkRunRequest(
        name=name,
        kind=kind,
        model_id=resolved_model,
        session_id=resolved_session,
        profile_id=profile,
        context_length=context_length,
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
    _print_benchmark_result(result)


def _print_benchmark_result(result: object) -> None:
    """Render a single benchmark result row-by-row."""
    mode = str(result.parameters.get("mode", "?"))
    reason = result.parameters.get("reason")
    table = Table(title=f"Benchmark: {result.name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    success_color = "green" if result.success else "red"
    table.add_row("Success", f"[{success_color}]{result.success}[/{success_color}]")
    if result.error:
        table.add_row("Error", f"[red]{result.error}[/red]")
    mode_color = "green" if mode == "live" else "yellow"
    table.add_row("Mode", f"[{mode_color}]{mode}[/{mode_color}]")
    if reason:
        table.add_row("Reason", str(reason))
    if result.kind:
        table.add_row("Kind", result.kind.value)
    if result.backend:
        table.add_row("Backend", result.backend)
    if result.context_length:
        table.add_row("Context length", str(result.context_length))
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
    if result.peak_vram_mb is not None:
        table.add_row("Peak VRAM", f"{result.peak_vram_mb} MB")
    if result.avg_gpu_util_pct is not None:
        table.add_row(
            "GPU util (avg / max)",
            f"{result.avg_gpu_util_pct:.1f}% / {result.max_gpu_util_pct:.0f}%",
        )
    console.print(table)
    if mode == "mock" and not result.error:
        console.print(
            "[yellow]Mock fallback used[/yellow] (no live runtime). "
            "Run against a started session for real metrics."
        )


@app.command("benchmarks")
def benchmarks_cmd(
    benchmark_id: Annotated[
        str | None,
        typer.Argument(help="Show details for a single benchmark id."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Filter by model ID.")
    ] = None,
    session: Annotated[
        str | None, typer.Option("--session", help="Filter by session ID.")
    ] = None,
    kind: Annotated[
        BenchmarkKind | None,
        typer.Option(help="Filter by kind.", case_sensitive=False),
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum rows to return.")] = 20,
) -> None:
    """List benchmark history (newest first) or show one benchmark in detail."""
    with _session() as db:
        service = BenchmarkService(db)
        if benchmark_id:
            result = service.get_result(benchmark_id)
            if result is None:
                raise typer.Exit(code=1)
            _print_benchmark_result(result)
            return
        results = service.list_results(
            model_id=model, session_id=session, kind=kind, limit=limit
        )
    if not results:
        console.print("[yellow]No benchmarks recorded yet.[/yellow]")
        return
    table = Table(title=f"Benchmarks ({len(results)})")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Backend")
    table.add_column("Tok/s", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("Peak VRAM", justify="right")
    table.add_column("GPU%", justify="right")
    table.add_column("OK")
    table.add_column("When")
    for result in results:
        tps = "-" if result.tokens_per_second is None else f"{result.tokens_per_second:.1f}"
        ttft = (
            "-"
            if result.time_to_first_token_ms is None
            else f"{result.time_to_first_token_ms:.0f} ms"
        )
        peak = "-" if result.peak_vram_mb is None else f"{result.peak_vram_mb}"
        util = (
            "-"
            if result.max_gpu_util_pct is None
            else f"{result.max_gpu_util_pct:.0f}"
        )
        when = result.created_at.strftime("%Y-%m-%d %H:%M") if result.created_at else "-"
        ok_color = "green" if result.success else "red"
        table.add_row(
            (result.id or "")[:8],
            result.name,
            result.kind.value if result.kind else "-",
            result.backend or "-",
            tps,
            ttft,
            peak,
            util,
            f"[{ok_color}]{'y' if result.success else 'n'}[/{ok_color}]",
            when,
        )
    console.print(table)


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
    bind_host = host or settings.api.host
    if bind_host not in ("127.0.0.1", "localhost") and not settings.scheduler.allow_public_bind:
        raise typer.BadParameter(
            f"Refusing to bind the control-plane API to {bind_host}: its mutating "
            "routes are unauthenticated. Set scheduler.allow_public_bind=true in "
            "settings.yaml to override, or expose it via a reverse proxy instead."
        )
    if settings.scheduler.require_auth_token:
        from llmctl.config import resolve_api_auth_token

        if not resolve_api_auth_token(settings):
            raise typer.BadParameter(
                "scheduler.require_auth_token is on but no token is configured. "
                "Set api.auth_token in settings.yaml or export LLMCTL_API_TOKEN."
            )
    uvicorn.run(
        create_app(settings),
        host=bind_host,
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
    try:
        unit = render_session_unit(session, user=user)
    except AdoptError as exc:
        console.print(f"[red]generate-systemd-session refused:[/red] {exc}")
        raise typer.Exit(1) from exc
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
        active_states = {"running", "starting", "degraded", "planned"}
        with _session() as db:
            sessions = SessionService(db).list_sessions()
        active = [s for s in sessions if s.status.value in active_states]
        # ADOPTED sessions don't have an llmctl-issued launch command;
        # their lifecycle belongs to the upstream systemd unit. Skip them
        # rather than abort the whole --all batch.
        owned = [s for s in active if s.kind.value != "adopted"]
        skipped = [s for s in active if s.kind.value == "adopted"]
        for s in skipped:
            console.print(
                f"[yellow]Skip adopted session[/yellow] {s.id} "
                f"({s.systemd_unit or s.endpoint_url}); already managed by systemd."
            )
        if not owned:
            console.print("[yellow]No owned sessions to persist.[/yellow]")
            return
        console.print(f"[cyan]Persisting {len(owned)} owned session(s).[/cyan]")
        for session in owned:
            _install(render_session_unit(session, user=user))
        return

    if session_id:
        with _session() as db:
            session = SessionService(db).get_session(session_id)
        if session is None:
            raise typer.BadParameter(f"Session not found: {session_id}")
        try:
            unit = render_session_unit(session, user=user)
        except AdoptError as exc:
            console.print(f"[red]install-systemd refused:[/red] {exc}")
            raise typer.Exit(1) from exc
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
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Start the TP-fleet vLLM unit on PRESET (replaces ``gpu-models vllm``)."""
    settings = load_settings()
    if not dry_run:
        _confirm_state_change(
            f"Restart {settings.managed_units.vllm_tp.unit_name} with preset '{preset}' "
            "(interrupts whatever it is currently serving)",
            required=settings.scheduler.require_confirmation_for_start,
            assume_yes=yes,
        )
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


@app.command("presets")
def presets_cmd(json_out: _JSON_OPT = False) -> None:
    """List preset aliases known to llmctl."""
    views = load_preset_views()
    if json_out:
        import dataclasses

        _emit_json([dataclasses.asdict(v) for v in views])
        return
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
def status_cmd(json_out: _JSON_OPT = False) -> None:
    """Quick overview of managed units: config, live state, and served models."""
    from llmctl.services.backends import probe_openai_v1_models

    settings = load_settings()
    rows = []
    for role, unit in (
        ("vllm-tp", settings.managed_units.vllm_tp),
    ):
        served = probe_openai_v1_models(f"http://127.0.0.1:{unit.default_port}", 1.5)
        rows.append(
            {
                "role": role,
                "unit_name": unit.unit_name,
                "env_file": str(unit.resolve_env_file()),
                "port": unit.default_port,
                "serving": served is not None,
                "served_models": served or [],
            }
        )
    if json_out:
        _emit_json(rows)
        return
    table = Table(title="Managed units")
    table.add_column("role", style="cyan")
    table.add_column("unit name")
    table.add_column("env file")
    table.add_column("port", justify="right")
    table.add_column("serving")
    for row in rows:
        serving = (
            f"[green]{', '.join(row['served_models'])}[/green]"
            if row["serving"] and row["served_models"]
            else ("[green]yes (empty list)[/green]" if row["serving"] else "[red]no[/red]")
        )
        table.add_row(
            row["role"], row["unit_name"], row["env_file"], str(row["port"]), serving
        )
    console.print(table)


@app.command("validate")
def validate_cmd(json_out: _JSON_OPT = False) -> None:
    """Check that everything llmctl records still exists where it records it.

    Four read-only checks: preset ``model_id`` targets, registry row
    paths, dangling symlinks in the configured model roots, and managed
    units that are active but serve nothing on their registered port.
    Exits non-zero when anything is found, so it can gate a script.
    """
    from llmctl.config import load_model_dirs
    from llmctl.presets.store import load_all as load_all_presets
    from llmctl.services import validate as validate_svc

    settings = load_settings()
    with _session() as db:
        models = RegistryService(db).list_models(include_inactive=True)

    findings = [
        *validate_svc.check_preset_model_ids(load_all_presets()),
        *validate_svc.check_registry_paths(models),
        *validate_svc.check_model_root_symlinks(load_model_dirs()),
        *validate_svc.check_managed_unit_ports([settings.managed_units.vllm_tp]),
    ]

    if json_out:
        _emit_json(
            [
                {"check": f.check, "target": f.target, "detail": f.detail}
                for f in findings
            ]
        )
        if findings:
            raise typer.Exit(code=1)
        return

    if not findings:
        console.print("[green]Validation passed.[/green] No drift found.")
        return

    table = Table(title=f"Validation findings ({len(findings)})")
    table.add_column("Check", style="yellow")
    table.add_column("Target", style="cyan")
    table.add_column("Detail")
    for finding in findings:
        table.add_row(finding.check, finding.target, finding.detail)
    console.print(table)
    raise typer.Exit(code=1)


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


# ---------------------------------------------------------------------------
# Adopt flow — track externally-managed runtime endpoints (systemd vLLM units,
# manually-started llama.cpp servers, etc.) as kind=ADOPTED sessions so the
# gateway can route to them without llmctl ever spawning a process.
# ---------------------------------------------------------------------------


#: CLI shorthand -> ManagedUnitsConfig attribute name. Lets the user type
#: ``llmctl adopt-managed vllm-tp`` instead of ``vllm_tp``.
_MANAGED_ROLES: dict[str, str] = {
    "vllm-tp": "vllm_tp",
}


def _print_adopted(session: object, *, prefix: str = "Adopted") -> None:
    """Render an adopt confirmation line shared by adopt + adopt-managed."""
    console.print(
        f"[green]{prefix}[/green] {session.id}  "  # type: ignore[attr-defined]
        f"runtime={session.runtime.value}  "  # type: ignore[attr-defined]
        f"served_name={session.served_name or '-'}  "  # type: ignore[attr-defined]
        f"endpoint={session.endpoint_url}  "  # type: ignore[attr-defined]
        f"unit={session.systemd_unit or '-'}"  # type: ignore[attr-defined]
    )


@app.command("adopt")
def adopt_cmd(
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Base URL of the running upstream, e.g. http://127.0.0.1:8003."),
    ],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime", "-r",
            help=(
                "Runtime adapter id (vllm, llama_cpp, lmstudio, ollama, "
                "python_script), or 'openai' for a generic OpenAI-compatible "
                "endpoint llmctl routes to but does not manage."
            ),
        ),
    ] = "vllm",
    unit: Annotated[
        str | None,
        typer.Option(
            "--unit",
            help="Optional systemd unit name to associate (e.g. vllm-tp.service).",
        ),
    ] = None,
    served_name: Annotated[
        str | None,
        typer.Option(
            "--served-name",
            help=(
                "OpenAI model id the upstream answers to. "
                "Defaults to the first id from /v1/models."
            ),
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(help="Probe timeout in seconds against {endpoint}/v1/models."),
    ] = 1.5,
) -> None:
    """Track an externally-managed runtime endpoint as a kind=ADOPTED session."""
    try:
        runtime_enum = RuntimeName(runtime)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Unknown runtime '{runtime}'. Choose one of: "
            f"{', '.join(r.value for r in RuntimeName)}."
        ) from exc

    with _session() as db:
        service = SessionService(db)
        try:
            session = service.adopt(
                runtime_enum,
                endpoint,
                served_name=served_name,
                systemd_unit=unit,
                timeout_s=timeout,
            )
        except AdoptError as exc:
            console.print(f"[red]Adopt failed:[/red] {exc}")
            raise typer.Exit(1) from exc
    _print_adopted(session)


@app.command("adopt-managed")
def adopt_managed_cmd(
    role: Annotated[
        str | None,
        typer.Argument(
            help="Managed-unit role to adopt (vllm-tp). "
            "Omit and pass --all to adopt every running role.",
        ),
    ] = None,
    all_roles: Annotated[
        bool,
        typer.Option("--all", help="Probe and adopt every managed-unit role that responds."),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(help="Probe timeout in seconds per role."),
    ] = 1.5,
) -> None:
    """Adopt one or all managed systemd units declared in settings.managed_units.*."""
    if role is None and not all_roles:
        raise typer.BadParameter("Pass a role (vllm-tp) or --all.")
    if role is not None and all_roles:
        raise typer.BadParameter("Pass a role OR --all, not both.")

    settings = load_settings()
    if all_roles:
        targets = list(_MANAGED_ROLES.items())
    else:
        attr = _MANAGED_ROLES.get(role or "")
        if attr is None:
            raise typer.BadParameter(
                f"Unknown role '{role}'. Choose: {', '.join(sorted(_MANAGED_ROLES))}."
            )
        targets = [(role or "", attr)]

    adopted = 0
    skipped = 0
    with _session() as db:
        service = SessionService(db)
        for cli_role, attr in targets:
            unit_cfg: ManagedUnitConfig = getattr(settings.managed_units, attr)
            endpoint = f"http://127.0.0.1:{unit_cfg.default_port}"
            try:
                session = service.adopt(
                    RuntimeName.VLLM,
                    endpoint,
                    systemd_unit=f"{unit_cfg.unit_name}.service",
                    timeout_s=timeout,
                )
            except AdoptError as exc:
                skipped += 1
                console.print(
                    f"[yellow]Skip {cli_role}[/yellow] ({endpoint}): {exc}"
                )
                continue
            adopted += 1
            _print_adopted(session)

    summary_color = "green" if adopted else "yellow"
    console.print(
        f"[{summary_color}]Done.[/{summary_color}] adopted={adopted}, skipped={skipped}."
    )
    if adopted == 0 and skipped > 0:
        raise typer.Exit(1)


@app.command("detach")
def detach_cmd(
    session_id: Annotated[str, typer.Argument(help="ADOPTED session id to remove from tracking.")],
) -> None:
    """Remove an adopted session from llmctl tracking (upstream untouched)."""
    with _session() as db:
        service = SessionService(db)
        try:
            session = service.detach(session_id)
        except AdoptError as exc:
            console.print(f"[red]Detach refused:[/red] {exc}")
            raise typer.Exit(1) from exc
    if session is None:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(
        f"[green]Detached[/green] {session_id} "
        f"(endpoint={session.endpoint_url}, unit={session.systemd_unit or '-'}). "
        "The upstream process was not touched."
    )
