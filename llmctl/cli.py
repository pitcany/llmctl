"""Typer CLI skeleton for LLM Mission Control."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import click
import typer
import uvicorn
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table
from sqlmodel import Session

from llmctl import cli_swap
from llmctl.api.app import create_app
from llmctl.config import VLLM_ROLE, ManagedUnitConfig, Settings, load_settings, resolve_role
from llmctl.db import (
    BenchmarkKind,
    ModelRecord,
    RuntimeName,
    SessionRecord,
    get_engine,
    init_db,
)
from llmctl.gpu_arg import GPU_MODES, parse_gpus
from llmctl.integrations.launcher_env import LauncherEnvError
from llmctl.schemas import (
    BenchmarkRunRequest,
    ModelCreate,
    SessionStartRequest,
)
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.lanes import (
    SwapTarget,
    close_matches,
    gather_state,
    record_load,
    resolve,
)
from llmctl.services.preset_loader import load_preset_views
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import AdoptError, SessionService
from llmctl.services.vllm_orchestrator import (
    MissingModelPathError,
    OrchestratorOptions,
    UnknownPresetError,
    start_vllm_tp,
)
from llmctl.telemetry.gpu import get_gpu_info


class _SuggestingGroup(typer.core.TyperGroup):
    """Print exactly one "Did you mean" hint for a mistyped command.

    Click 8.4 appends its own suggestion to NoSuchCommand and Typer 0.24 added
    a second; Typer 0.27 (vendored click) appends none. Supplying the hint
    here, only when the error carries none, gives one hint on every version.
    """

    #: Front-page order: the everyday verbs in the order a new user meets them.
    FRONT = (
        "setup", "add", "run", "settings", "ls", "stop", "rm",
        "status", "logs", "doctor", "tui", "help",
    )

    def list_commands(self, ctx: Any) -> list[str]:
        names = list(super().list_commands(ctx))
        rank = {name: i for i, name in enumerate(self.FRONT)}
        return sorted(names, key=lambda n: (rank.get(n, len(rank)), names.index(n)))

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        try:
            return super().resolve_command(ctx, args)
        except Exception as exc:
            message = getattr(exc, "message", None)
            # click 8.4 raises NoSuchCommand, which renders its own hint.
            if (
                type(exc).__name__ != "UsageError"
                or not isinstance(message, str)
                or "Did you mean" in message
            ):
                raise
            import difflib

            wanted = str(args[0]) if args else ""
            close = difflib.get_close_matches(wanted, self.list_commands(ctx), n=1)
            if close:
                exc.message = f"{message.rstrip('.')}. Did you mean '{close[0]}'?"
            raise


app = typer.Typer(
    name="llmctl",
    cls=_SuggestingGroup,
    help=(
        "Run models on vLLM, SGLang, FreeToken, Strata, llama.cpp, Ollama or LM Studio "
        "from one place."
    ),
    epilog="More commands: llmctl help --all",
    no_args_is_help=True,
    # -h everywhere: Click contexts inherit help_option_names from the parent,
    # so this one setting reaches every subcommand and sub-typer.
    context_settings={"help_option_names": ["-h", "--help"]},
    # One suggestion comes from _SuggestingGroup; Typer's own would double it.
    suggest_commands=False,
)
console = Console()

#: Distribution name; ``llmctl`` on PyPI belongs to someone else.
DIST_NAME = "pitcany-llmctl"


def _version_string() -> str:
    """``llmctl X.Y.Z``, plus ``+<sha8>`` when installed from a git URL.

    pip and uv record the commit of a VCS install in ``direct_url.json``
    (``vcs_info.commit_id``); an editable or source install has no
    ``vcs_info``. Any metadata failure falls back to the bare version.
    """
    import json
    from importlib import metadata

    from llmctl import __version__

    try:
        raw = metadata.distribution(DIST_NAME).read_text("direct_url.json")
        commit = str(json.loads(raw or "{}").get("vcs_info", {}).get("commit_id") or "")
    except Exception:  # noqa: BLE001 - --version must never crash
        commit = ""
    return f"llmctl {__version__}+{commit[:8]}" if commit else f"llmctl {__version__}"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(_version_string())
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the version (and the git commit, if known) and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    pass


# Registry/profile management commands (CRUD, preview, export/import) live in
# their own module to keep cli.py focused on launch/orchestration. ``register``
# attaches sub-typers (``model``, ``profile``) plus top-level commands
# (``profiles``, ``preview``, ``export-registry``, ``import-registry``).
from llmctl import cli_registry  # noqa: E402

cli_registry.register(app)

# `llmctl hf`: read-only Hugging Face search and sizing (ported from vserve).
from llmctl import cli_hf  # noqa: E402

cli_hf.register(app)

# Plugins add their own verbs last (see llmctl.plugins); a plugin that fails
# to import is reported on stderr and skipped so `llmctl` still runs.
from llmctl.plugins import registry as _plugin_registry  # noqa: E402

for _hook in _plugin_registry().commands:
    _hook(app)


def _session_model_cell(session: object, names: dict[str, str]) -> str:
    """What this session serves, as a name the operator recognises.

    ``served_name`` is what the endpoint answers with and what every other
    surface (`status`, `swap --list`) prints, so it wins. A registry model_id
    is resolved to its name; the raw uuid is never shown, because nothing can
    be done with it here. The column used to print ``model_id`` alone, which
    is None for adopted sessions and for the vLLM unit -- an empty column on
    every row of a table whose whole subject is what is running.
    """
    served = getattr(session, "served_name", None)
    if served:
        return str(served)
    model_id = getattr(session, "model_id", None)
    if model_id:
        return names.get(str(model_id), str(model_id))
    return "—"


def _id_column(table: Table, header: str, **kwargs: object) -> None:
    """Add a column holding a value the operator types back into a command.

    Rich's default column overflow is ``ellipsis``: once a table is wider than
    the terminal every column is squeezed and characters are dropped silently.
    That is fine for prose, but an ellipsised alias, unit name, path or id is
    unusable -- it cannot be copied into the next command, and nothing on
    screen says it was shortened. ``fold`` wraps onto extra lines instead, so
    the value survives at any terminal width. Descriptive columns keep the
    default; ``--json`` remains the lossless machine-readable path.
    """
    table.add_column(header, overflow="fold", **kwargs)  # type: ignore[arg-type]


def _session() -> Session:
    settings = load_settings()
    init_db(settings.database_url)
    return Session(get_engine(settings.database_url))


def _confirm_state_change(action: str, *, required: bool, assume_yes: bool) -> None:
    """Confirmation gate for a state-changing action.

    Applies when the matching ``scheduler.require_confirmation_for_*``
    setting is on and ``--yes`` was not passed. On a TTY it prompts; off a
    TTY it REFUSES (exit 1) rather than proceeding unasked -- a script that
    means to restart the GPU fleet says so with ``--yes``. It used to fail
    open, so any pipe or cron job silently skipped the confirmation the
    setting asked for. Declining the prompt is a clean exit 0 -- the user
    saying "no" is not an error.
    """
    import sys

    if not required or assume_yes:
        return
    if not sys.stdin.isatty():
        console.print(
            f"[red]Refusing:[/red] {escape(action)} needs confirmation and stdin is "
            "not a terminal. Pass --yes to run it non-interactively."
        )
        raise typer.Exit(1)
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


@app.command(hidden=True)
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
    _id_column(table, "Name", style="cyan")
    table.add_column("Backend")
    _id_column(table, "Path")
    table.add_column("Format")
    for model in discovered:
        table.add_row(
            escape(str(model.name)),
            model.runtime.value,
            escape(str(model.path or "")),
            model.format or "",
        )
    console.print(table)
    console.print(
        "[dim]Re-run with --import to register these into the registry.[/dim]"
    )


def models_cmd(json_out: _JSON_OPT = False) -> None:
    """List registered models."""
    with _session() as db:
        models = RegistryService(db).list_models()
    if json_out:
        _emit_json([model.model_dump(mode="json") for model in models])
        return
    table = Table(title="Models")
    _id_column(table, "ID")
    _id_column(table, "Name")
    table.add_column("Runtime")
    _id_column(table, "Host")
    table.add_column("Status")
    for model in models:
        table.add_row(
            model.id or "",
            escape(str(model.name)),
            model.runtime.value,
            model.host or "",
            model.status.value,
        )
    console.print(table)


cli_registry.model_app.command("list", help="List registered models.")(models_cmd)
app.command("models", hidden=True, help="Deprecated: use `model list`.")(
    cli_registry.deprecated_alias(models_cmd, "models", "model list")
)


@app.command(hidden=True)
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
            escape(str(gpu.name)),
            memory,
            util,
            temp,
            power,
            str(len(gpu.processes)),
        )
    console.print(table)


@app.command("sessions", hidden=True)
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
        # Registry names, looked up only when some session carries a model_id:
        # served_name answers for every other row, and an unconditional query
        # would make listing sessions depend on the registry for nothing.
        names: dict[str, str] = {}
        if any(session.model_id for session in sessions):
            names = {
                model.id: model.name
                for model in RegistryService(db).list_models()
                if model.id is not None
            }
    if json_out:
        _emit_json([session.model_dump(mode="json") for session in sessions])
        return
    table = Table(title="Sessions")
    _id_column(table, "ID")
    table.add_column("Runtime")
    table.add_column("Status")
    _id_column(table, "Model")
    for session in sessions:
        # A stop that did not take is not a status — reconcile legitimately
        # reports the process as running again — but the operator still needs
        # to see that their stop never landed.
        status_cell = session.status.value
        if session.terminate_failed_at is not None:
            status_cell = f"{status_cell} [yellow](stop failed)[/yellow]"
        table.add_row(
            session.id or "",
            session.runtime.value,
            status_cell,
            escape(_session_model_cell(session, names)),
        )
    console.print(table)


@app.command("add-model", hidden=True)
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
    """Deprecated: use `llmctl model add` (non-interactive when all flags are given)."""
    cli_registry.err_console.print(
        "[dim]note: `llmctl add-model` is deprecated; use `llmctl model add` "
        "(--backend instead of --runtime).[/dim]"
    )
    payload = ModelCreate(
        name=name,
        runtime=runtime,
        path=str(path) if path else None,
        source=source,
        estimated_vram_gb=estimated_vram,
    )
    with _session() as db:
        model = RegistryService(db).add_model(payload)
    console.print(f"[green]Registered model[/green] {escape(str(model.name))} ({model.id})")


@app.command("delete-model", hidden=True)
def delete_model(
    model_id: Annotated[str, typer.Argument(help="Model ID to soft-delete.")],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Deprecated: use `llmctl model delete` (also accepts a name)."""
    cli_registry.err_console.print(
        "[dim]note: `llmctl delete-model` is deprecated; use `llmctl model delete`.[/dim]"
    )
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
    parameters: dict[str, Any] | None = None,
) -> SessionStartRequest:
    """Resolve model/profile and build a :class:`SessionStartRequest`."""
    from llmctl.db import ModelRecord
    from llmctl.services.profiles import ProfileService
    from llmctl.services.registry import RegistryService

    gpu_ids, mode, allow_cpu = parse_gpus(gpus, cpu=cpu)
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
        gpus_auto=mode in GPU_MODES,
        allow_cpu=allow_cpu,
        force=force,
        dry_run=dry_run,
        parameters=dict(parameters or {}),
    )


def _print_session_state(session: object, *, verb: str) -> bool:
    """Print what actually happened to ``session``; ``False`` when it failed.

    One producer for ``start`` and ``restart`` so the two commands cannot
    drift into describing the same states differently. ``restart`` used to
    print "Restart planned <id>; no process launched." unconditionally --
    while `SessionService.restart` terminated the process *and relaunched it*
    -- so the one surface an operator checks after a restart reported the
    opposite of what happened.
    """
    status = session.status.value
    if status == "running":
        console.print(
            f"[green]{verb} session[/green] {session.id} "
            f"pid={session.pid} -> {escape(str(session.endpoint_url))}"
        )
        return True
    if status == "planned":
        console.print(
            f"[cyan]Planned session[/cyan] {session.id} "
            f"({session.runtime.value}); no process launched."
        )
        return True
    if status == "starting":
        # An adopted unit restarted via systemctl has no pid llmctl owns,
        # so print one only when there is one rather than "pid=None".
        pid_part = f" pid={session.pid}" if session.pid is not None else ""
        console.print(
            f"[yellow]Session {session.id} is starting[/yellow]{pid_part}; "
            "the endpoint is not ready yet (large models load for minutes). "
            "`llmctl sessions` will show it running once it responds."
        )
        return True
    console.print(
        f"[red]Session {session.id} {status}[/red]: {escape(str(session.error))}"
    )
    return False


def _print_plan_warnings(plan: object) -> None:
    """Print any warnings and refusal reasons attached to a launch plan."""
    for warning in getattr(plan, "warnings", []) or []:
        console.print(f"[yellow]warning:[/yellow] {escape(str(warning))}")
    for reason in getattr(plan, "refusal_reasons", []) or []:
        console.print(f"[red]refusal:[/red] {escape(str(reason))}")


@app.command(hidden=True)
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
    """Launch a model session (or plan it with --dry-run).

    --dry-run records a PLANNED session row without launching (visible in
    `sessions`, purged by `cleanup --remove-stale`). To look without writing
    anything, use `plan` (verdict + command) or `preview` (adds env/health).
    """
    with _session() as db:
        request = _build_start_request(db, model_id, profile, gpus, cpu, force, dry_run)
        _start_request(db, request)


def _start_request(db: Session, request: SessionStartRequest) -> None:
    """Start ``request``, print the outcome, exit 1 when the launch failed."""
    from llmctl.services.scheduler import SchedulerError

    try:
        session = SessionService(db).start(request)
    except SchedulerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if session.launch_plan is not None:
        _print_plan_warnings(session.launch_plan)
    # Exit 1 on a failed launch, like `restart`. It used to return 0 (kept for
    # scripts that might depend on it; none on this box did), which also hid
    # failures from the generated systemd units' Restart=on-failure.
    if not _print_session_state(session, verb="Started"):
        raise typer.Exit(1)


@app.command(hidden=True)
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
    """Print the launch plan and allow/refuse verdict; writes nothing.

    Siblings: `preview` shows the same plan plus env vars, health URL and
    notes; `start --dry-run` also records a PLANNED session row.
    """
    with _session() as db:
        request = _build_start_request(db, model_id, profile, gpus, cpu, force=False, dry_run=True)
        launch_plan = SessionService(db).plan(request)
    _print_launch_plan(launch_plan)


def _print_launch_plan(launch_plan: Any) -> None:
    """The ``plan`` table, warnings and verdict."""
    table = Table(title="Launch Plan", show_header=False)
    table.add_column("Field", style="cyan")
    _id_column(table, "Value")
    model_label = launch_plan.model_name or launch_plan.model_id
    table.add_row("Model", escape(f"{model_label} "))
    table.add_row("Backend", launch_plan.runtime.value)
    table.add_row("Profile", escape(str(launch_plan.profile_name or "-")))
    table.add_row("GPU mode", launch_plan.gpu_selection_mode)
    table.add_row("Selected GPUs", ",".join(map(str, launch_plan.gpu_ids)) or "cpu")
    table.add_row("Tensor parallel", str(launch_plan.tensor_parallel_size))
    table.add_row("Port", str(launch_plan.port or "-"))
    table.add_row("Endpoint", escape(str(launch_plan.endpoint_url or "-")))
    est = (
        "unknown"
        if launch_plan.estimated_vram_gb is None
        else f"{launch_plan.estimated_vram_gb:.1f} GB"
    )
    free = "n/a" if launch_plan.free_vram_gb is None else f"{launch_plan.free_vram_gb:.1f} GB"
    table.add_row("Estimated VRAM", est)
    table.add_row("Free VRAM", free)
    # The command preview carries model paths and launch args verbatim.
    table.add_row("Command", escape(launch_plan.command_preview))
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
    """Full environment checkup with remediation hints: storage, runtimes, GPUs, drift.

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
            console.print(
                f"[{style}]{verdict}[/{style}] {escape(str(check['name']))}: "
                f"{escape(str(check['detail']))}"
            )
            if check.get("remediation") and verdict != "PASS":
                console.print(f"       ↳ {escape(str(check['remediation']))}")
    console.print(
        f"\n{len(report['passed'])} passed, {len(report['warnings'])} warning(s), "
        f"{len(report['failures'])} failure(s)."
    )
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command(hidden=True)
def cleanup(
    remove_stale: Annotated[
        bool,
        typer.Option(
            help="Delete stopped/failed session records and leftover dry-run plans."
        ),
    ] = False,
) -> None:
    """Mark dead sessions and report free ports; --remove-stale deletes stale rows.

    Always writes to the session DB: runs a reconcile that marks sessions
    whose process/endpoint is gone as STOPPED. Never kills a process and
    never frees a port -- it only reports which stale sessions' ports are
    already free. With --remove-stale it also deletes STOPPED/FAILED rows and
    leftover `start --dry-run` plans.
    """
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
        console.print(
            f"[cyan]Ports already free (stale sessions):[/cyan] {', '.join(map(str, freed))}"
        )


@app.command(hidden=True)
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


@app.command("start-unit", hidden=True)
def start_unit(
    unit: Annotated[str, typer.Argument(help="systemd unit name, e.g. gpt-oss-120b.")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve the unit and its scope; start nothing."),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Start a systemd unit llmctl does not own.

    The counterpart to `stop <id> --systemd`. That verb takes a *session*, but
    stopping one of these units typically removes its session row, leaving
    nothing to name -- so this takes the unit instead. Scope is detected, so a
    user unit is started with `systemctl --user` and no sudo.

    No session row is created: these units adopt themselves from their own
    ExecStartPost, and inventing a row here would race that hook.
    """
    settings = load_settings()
    if not dry_run:
        _confirm_state_change(
            f"Start systemd unit {unit}",
            required=settings.scheduler.require_confirmation_for_start,
            assume_yes=yes,
        )
    with _session() as db:
        try:
            scope, was_active = SessionService(db).start_unit(unit, dry_run=dry_run)
        except AdoptError as exc:
            console.print(f"[red]Start refused:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    if dry_run:
        flag = " --user" if scope == "user" else ""
        state = "already active — nothing would happen" if was_active else "inactive"
        console.print(
            f"[cyan]dry-run:[/cyan] {escape(unit)} resolves to {scope} scope ({state}).\n"
            f"  would run: systemctl{flag} start {escape(unit)}"
        )
        return
    if was_active:
        console.print(f"[cyan]{escape(str(unit))} is already active[/cyan] ({scope} scope).")
        return
    console.print(
        f"[green]Started[/green] {escape(str(unit))} ({scope} scope). "
        "It will appear in `llmctl sessions` once it adopts itself and answers."
    )


@app.command()
def stop(
    target: Annotated[
        str | None,
        typer.Argument(
            help="Session ID or a name from `llmctl ls`. Omit to stop what is serving.",
            show_default=False,
        ),
    ] = None,
    systemd: Annotated[
        bool,
        typer.Option(
            "--systemd",
            "-s",
            help="For an adopted session, stop its backing systemd unit (sudo).",
        ),
    ] = False,
    all_: Annotated[
        bool, typer.Option("--all", help="Stop everything that is serving.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Stop what is serving: the vLLM unit, a lane, or a model llmctl started.

    With no argument it stops the one thing serving (if several are, it lists
    them; name one, or pass --all). A session ID or a name from `llmctl ls`
    picks one.
    """
    settings = load_settings()
    if target is not None and not all_ and _is_live_session(target):
        _stop_session(target, systemd=systemd, yes=yes, settings=settings)
        return
    if systemd:
        raise typer.BadParameter("--systemd applies to a session ID (see `llmctl sessions`)")
    from llmctl.services.stop_serving import pick, real_deps, serving_now, stop_items

    items = serving_now(gather_state(settings))
    if target is not None:
        items = pick(items, target)
        if not items:
            with _session() as db:
                known = SessionService(db).get_session(target) is not None
            if known:  # a session that is not serving: keep the old behaviour
                _stop_session(target, systemd=False, yes=yes, settings=settings)
                return
            raise typer.BadParameter(
                f"Nothing serving is called {target}. `llmctl ls` shows what serves."
            )
    if not items:
        console.print("Nothing llmctl can stop is serving.")
        return
    if len(items) > 1 and not all_ and target is None:
        console.print("More than one thing is serving:")
        for item in items:
            console.print(f"  - {escape(item.label)}")
        console.print("Stop one: llmctl stop NAME   ·   stop all: llmctl stop --all")
        raise typer.Exit(1)
    _confirm_state_change(
        "Stop " + "; ".join(item.label for item in items),
        required=settings.scheduler.require_confirmation_for_stop,
        assume_yes=yes,
    )
    failed = False
    for item, ok in stop_items(items, real_deps(settings)):
        if ok:
            console.print(f"[green]Stopped[/green] {escape(item.label)}")
        else:
            failed = True
            console.print(f"[red]Could not stop[/red] {escape(item.label)}")
    if failed:
        raise typer.Exit(1)


def _is_live_session(selector: str) -> bool:
    """Whether ``selector`` names a session that is running (or ambiguous)."""
    from llmctl.services.sessions import AmbiguousSessionError

    with _session() as db:
        try:
            session = SessionService(db).get_session(selector)
        except AmbiguousSessionError:
            return True  # let the session path report the ambiguity
    return session is not None and session.status.value != "stopped"


def _stop_session(session_id: str, *, systemd: bool, yes: bool, settings: Any) -> None:
    """Stop one session by ID (the original `llmctl stop ID`)."""
    _confirm_state_change(
        f"Stop session {session_id}" + (" and its systemd unit" if systemd else ""),
        required=settings.scheduler.require_confirmation_for_stop,
        assume_yes=yes,
    )
    with _session() as db:
        try:
            session = SessionService(db).stop(session_id, stop_unit=systemd)
        except AdoptError as exc:
            console.print(f"[red]Stop refused:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    if session.status.value != "stopped":
        # The process outlived SIGTERM and SIGKILL. It still holds its GPU
        # memory and its port, and the row keeps its pid so reconcile can
        # settle it later -- but this is not a successful stop.
        console.print(
            f"[red]Session {session_id} not stopped[/red] "
            f"(pid={session.pid}): {escape(str(session.error))}"
        )
        raise typer.Exit(1)
    console.print(f"[green]Session marked stopped[/green] {session_id}")


@app.command(hidden=True)
def restart(
    session_id: Annotated[str, typer.Argument(help="Session ID to restart.")],
    systemd: Annotated[
        bool,
        typer.Option(
            "--systemd",
            "-s",
            help="For an adopted session, restart its backing systemd unit.",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Stop a session's process and relaunch it from its stored launch plan.

    With --systemd, restart an adopted session's backing unit instead; this
    also starts a unit that is currently down.

    Confirmation is gated on scheduler.require_confirmation_for_stop -- the
    risky half is terminating whatever is currently serving.
    """
    settings = load_settings()
    _confirm_state_change(
        f"Restart session {session_id}"
        + (
            " and its systemd unit"
            if systemd
            else " (stops its process if running, then relaunches)"
        ),
        required=settings.scheduler.require_confirmation_for_stop,
        assume_yes=yes,
    )
    with _session() as db:
        try:
            session = SessionService(db).restart(session_id, restart_unit=systemd)
        except AdoptError as exc:
            console.print(f"[red]Restart refused:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    if not session:
        raise typer.BadParameter(f"Session not found: {session_id}")
    if not _print_session_state(session, verb="Restarted"):
        # The old process was terminated before the relaunch was attempted,
        # so a failed relaunch means this session is serving nothing.
        console.print(
            "  [yellow]The previous process was stopped; nothing is serving "
            f"this session.[/yellow]\n  next:\n    llmctl logs {session.id}"
            f"  # why the relaunch failed\n    llmctl restart {session.id}"
            "  # retry once fixed"
        )
        raise typer.Exit(1)


@app.command(hidden=True)
def pull(
    model: Annotated[str, typer.Argument(help="Ollama model tag, e.g. qwen3:32b.")],
) -> None:
    """Pull an Ollama model and register it (`model add` covers every engine)."""
    from llmctl.cli_registry import _add_from_source

    source = model if model.lower().startswith("ollama:") else f"ollama:{model}"
    _add_from_source(source, None, None, yes=True)


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
            # Log text is bracket-rich: `[/INST]` chat tokens raise MarkupError,
            # `[rank0]:` torch prefixes are silently deleted. Print the tail
            # verbatim -- the TUI escapes the same bytes (see tui/_base.esc).
            console.print(content, markup=False, highlight=False)
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
            escape(event.category),
            escape(event.message),
        )
    console.print(table)


config_app = typer.Typer(
    name="config", help="Show and validate llmctl configuration.", suggest_commands=False
)
app.add_typer(config_app, name="config", hidden=True)


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

    console.print(f"[dim]# resolved from {escape(str(_settings_file_path()))}[/dim]")
    print(yaml.safe_dump(payload, sort_keys=False))


@config_app.command("validate")
def config_validate() -> None:
    """Validate the settings file; exit 1 with the offending field on error."""
    try:
        load_settings()
    except Exception as exc:
        console.print(f"[red]Configuration invalid:[/red] {escape(str(_settings_file_path()))}")
        console.print(escape(str(exc)))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Configuration valid.[/green] ({escape(str(_settings_file_path()))})")


runtimes_app = typer.Typer(
    name="runtimes",
    suggest_commands=False,
    help="Inspect configured runtimes: install state, version, capabilities.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(runtimes_app, name="runtimes", hidden=True)


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
    _id_column(table, "runtime", style="cyan")
    table.add_column("state")
    table.add_column("version")
    _id_column(table, "endpoint")
    _id_column(table, "loaded models")
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
    console.print(f"[bold]{escape(str(match['display_name']))}[/bold] ({match['runtime']})")
    console.print(f"  state:    {escape(str(match['state']))} — {escape(str(match['message']))}")
    console.print(f"  version:  {escape(str(match['version'] or 'unknown'))}")
    console.print(f"  endpoint: {escape(str(match['endpoint'] or '-'))}")
    loaded = match["loaded"]
    console.print(
        "  loaded:   " + ("n/a" if loaded is None else (", ".join(loaded) or "(none)"))
    )
    console.print("  capabilities:")
    for key, supported in match["capabilities"].items():
        mark = "[green]yes[/green]" if supported else "[dim]no[/dim]"
        console.print(f"    {key}: {mark}")


@app.command(hidden=True)
def health(json_out: _JSON_OPT = False) -> None:
    """Live runtime probe: is each backend (vllm, ollama, ...) answering right now?

    Siblings: `status` = managed systemd units; `doctor` = whole-environment
    checkup with fixes; `validate` = recorded paths still exist;
    `router-status` = the llmctl gateway's routing table.
    """
    from llmctl.services.health import HealthService

    with _session() as db:
        data = HealthService(load_settings(), db=db).get_health()
    if json_out:
        _emit_json(data)
        return
    console.print(
        f"[bold]State:[/bold] {escape(str(data['state']))}  "
        f"[bold]Safe mode:[/bold] {data['safe_mode']}  "
        f"[bold]GPUs:[/bold] {data['gpu_count']}  "
        f"[bold]NVML:[/bold] {data['nvml_available']}"
    )
    table = Table(title="Runtime Health")
    _id_column(table, "Runtime")
    table.add_column("State")
    _id_column(table, "Serving")
    table.add_column("Message")
    for name, info in data["runtimes"].items():
        sessions = info.get("sessions", [])
        if sessions:
            labels = []
            for s in sessions:
                label = escape(
                    str(s["served_name"] or s["endpoint_url"] or (s["id"] or "")[:8])
                )
                if s.get("lane"):
                    label = f"{label} (lane {escape(str(s['lane']))})"
                elif not s.get("managed", True):
                    # Started outside llmctl: real, but not ours to stop or track.
                    label = f"{label} (unmanaged)"
                labels.append(label)
            serving = f"[green]{', '.join(labels)}[/green]"
        else:
            # "binary found" with nothing serving is the case that used to read
            # as healthy; say so plainly instead.
            serving = f"[{'red' if info['state'] == 'ok' else 'dim'}]nothing[/]"
        cells = [escape(str(name)), escape(str(info["state"])), escape(str(info["message"]))]
        if info["state"] == "not_used":
            cells = [f"[dim]{c}[/]" for c in cells]
        table.add_row(cells[0], cells[1], serving, cells[2])
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


@app.command(hidden=True)
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
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic/--live",
            help=(
                "Run and record a synthetic mock benchmark instead of hitting "
                "the live endpoint (e.g. on CI hosts without a runtime)."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool | None,
        typer.Option(
            "--dry-run/--no-dry-run",
            hidden=True,
            help="Deprecated alias of --synthetic/--live.",
        ),
    ] = None,
) -> None:
    """Run a benchmark against a model or session (history: `llmctl benchmarks`).

    Pass the positional ``TARGET`` as either a model ID or session ID --
    llmctl will look it up automatically. On live failures the result is
    persisted with ``success=False`` for inspection. Use ``--synthetic`` to
    run a mock benchmark instead (it still runs and records a result, so it
    is not a no-op preview like ``start --dry-run``).
    """
    if dry_run is not None:
        # `start --dry-run` executes nothing; `bench --dry-run` ran a fake
        # benchmark and persisted it. Same flag, opposite safety meaning.
        console.print(
            "[yellow]warning:[/yellow] `bench --dry-run` is deprecated -- it runs a "
            "synthetic benchmark, it is not a preview. Use --synthetic."
        )
        synthetic = dry_run
    resolved_model, resolved_session = _resolve_bench_target(
        target, model_id, session_id
    )
    try:
        sweep_levels = (
            [int(part) for part in sweep.split(",") if part.strip()] if sweep else []
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"--sweep {sweep!r}: expected comma-separated integers like '1,2,4,8'."
        ) from exc
    if any(level < 1 for level in sweep_levels):
        raise typer.BadParameter(f"--sweep {sweep!r}: concurrency levels must be >= 1.")
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
        dry_run=synthetic,
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
    _id_column(table, "Value")
    success_color = "green" if result.success else "red"
    table.add_row("Success", f"[{success_color}]{result.success}[/{success_color}]")
    if result.error:
        table.add_row("Error", f"[red]{escape(str(result.error))}[/red]")
    mode_color = "green" if mode == "live" else "yellow"
    table.add_row("Mode", f"[{mode_color}]{mode}[/{mode_color}]")
    if reason:
        table.add_row("Reason", escape(str(reason)))
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


@app.command("benchmarks", hidden=True)
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
    """List benchmark history (newest first) or show one in detail (run one: `llmctl bench`)."""
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
    _id_column(table, "ID", style="dim")
    _id_column(table, "Name")
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
            # Full id, not a [:8] prefix: `llmctl bench <id>` resolves by
            # primary key (BenchmarkService.get_result), so a shortened id is
            # not something the operator can act on. _id_column folds it.
            result.id or "-",
            escape(str(result.name)),
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
    """Launch the interactive TUI (the primary UI: presets, units, sessions, GPUs)."""
    from llmctl.tui.app import MissionControlApp

    MissionControlApp().run()


@app.command(hidden=True)
def serve(
    host: Annotated[str | None, typer.Option(help="Bind host; defaults to settings.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port; defaults to settings.")] = None,
) -> None:
    """Serve the FastAPI scaffold."""
    from llmctl.config import resolve_api_auth_token

    settings = load_settings()
    bind_host = host or settings.api.host
    public = bind_host not in ("127.0.0.1", "localhost", "::1")

    # `api.allow_public_bind`, NOT `scheduler.allow_public_bind`: the latter is
    # about serving a *model* on a public host, and reusing it here meant that
    # opting into that silently also exposed this API.
    if public and not settings.api.allow_public_bind:
        raise typer.BadParameter(
            f"Refusing to bind the control-plane API to {bind_host}: its mutating "
            "routes include POST /sessions/start, which runs local commands. Set "
            "api.allow_public_bind=true in settings.yaml (and configure "
            "api.auth_token) to override, or put it behind a reverse proxy instead."
        )

    if public:
        # A public bind forces authentication on regardless of
        # `require_auth_token`, which defaults to False -- otherwise a token can
        # be configured while the bearer middleware is never installed, leaving
        # every mutating route open.
        if not resolve_api_auth_token(settings):
            raise typer.BadParameter(
                f"Refusing to bind the control-plane API to {bind_host} without an "
                "auth token. Set api.auth_token in settings.yaml or export "
                "LLMCTL_API_TOKEN."
            )
        settings = settings.model_copy(deep=True)
        settings.scheduler.require_auth_token = True
    elif settings.scheduler.require_auth_token and not resolve_api_auth_token(settings):
        raise typer.BadParameter(
            "scheduler.require_auth_token is on but no token is configured. "
            "Set api.auth_token in settings.yaml or export LLMCTL_API_TOKEN."
        )

    uvicorn.run(
        create_app(settings),
        host=bind_host,
        port=port or settings.api.port,
    )


@app.command("generate-systemd", hidden=True)
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


@app.command("generate-systemd-session", hidden=True)
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
    except (AdoptError, ValueError) as exc:
        console.print(f"[red]generate-systemd-session refused:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    _emit_unit(unit, output)


@app.command("install-vllm-unit", hidden=True)
def install_vllm_unit(
    role: Annotated[
        str, typer.Option(help="managed_units role to write the unit for (vllm-tp = vllm-engine).")
    ] = VLLM_ROLE,
    enable: Annotated[
        bool, typer.Option(help="Enable the unit after writing (it starts on the first swap).")
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Print the unit and the install actions without writing anything.",
        ),
    ] = True,
) -> None:
    """Write the systemd unit `llmctl run` restarts (shipped launcher; sudo)."""
    from llmctl.services.systemd import install_unit, render_vllm_unit

    settings = load_settings()
    roles = settings.managed_units.roles()
    role = resolve_role(role)
    if role not in roles:
        raise typer.BadParameter(f"no managed_units role {role!r}; have: {', '.join(roles)}")
    unit = render_vllm_unit(roles[role])
    for warning in unit.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(str(warning))}")
    if dry_run:
        console.print(unit.content, markup=False, highlight=False)
    report = install_unit(unit, dry_run=dry_run, enable=enable)
    header = "DRY RUN - would run" if report.dry_run else "Install report"
    color = "yellow" if report.dry_run else "green"
    console.print(
        f"[{color}]{header}[/{color}] for {escape(unit.name)} -> {escape(report.unit_path)}"
    )
    for action in report.actions:
        console.print(f"  - {escape(str(action))}")
    for message in report.messages:
        console.print(f"  [cyan]{escape(str(message))}[/cyan]")
    if dry_run:
        console.print("Re-run with --no-dry-run to install it.")
    else:
        env_file = roles[role].resolve_env_file()
        console.print(f"Next: llmctl run <preset>  # writes {escape(str(env_file))}")


@app.command("install-systemd", hidden=True)
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
            console.print(f"[yellow]warning:[/yellow] {escape(str(warning))}")
        report = install_unit(unit, dry_run=dry_run, enable=enable)  # type: ignore[arg-type]
        header = "DRY RUN - would run" if report.dry_run else "Install report"
        color = "yellow" if report.dry_run else "green"
        console.print(
            f"[{color}]{header}[/{color}] for {escape(str(unit.name))} -> "
            f"{escape(str(report.unit_path))}"  # type: ignore[attr-defined]
        )
        for action in report.actions:
            console.print(f"  - {escape(str(action))}")
        for message in report.messages:
            console.print(f"  [cyan]{escape(str(message))}[/cyan]")

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
                f"({escape(str(s.systemd_unit or s.endpoint_url))}); already managed by systemd."
            )
        if not owned:
            console.print("[yellow]No owned sessions to persist.[/yellow]")
            return
        console.print(f"[cyan]Persisting {len(owned)} owned session(s).[/cyan]")
        for session in owned:
            try:
                unit = render_session_unit(session, user=user)
            except ValueError as exc:
                console.print(f"[red]Skip session[/red] {session.id}: {escape(str(exc))}")
                continue
            _install(unit)
        return

    if session_id:
        with _session() as db:
            session = SessionService(db).get_session(session_id)
        if session is None:
            raise typer.BadParameter(f"Session not found: {session_id}")
        try:
            unit = render_session_unit(session, user=user)
        except (AdoptError, ValueError) as exc:
            console.print(f"[red]install-systemd refused:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    else:
        unit = render_api_unit(load_settings(), user=user)
    _install(unit)



def _emit_unit(unit: object, output: Path | None) -> None:
    """Print a rendered systemd unit (and warnings) or write it to a file."""
    name = getattr(unit, "name", "unit.service")
    content = getattr(unit, "content", "")
    for warning in getattr(unit, "warnings", []) or []:
        console.print(f"[yellow]warning:[/yellow] {escape(str(warning))}")
    if output is not None:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Wrote {escape(str(name))}[/green] -> {escape(str(output))}")
        return
    console.print(content)
    console.print("[cyan]Install with:[/cyan]")
    for command in unit.install_commands():  # type: ignore[attr-defined]
        console.print(f"  {command}")


# ---------------------------------------------------------------------------
# Phase 5: gpu-models replacement verbs (vllm, slot, presets, status).
# These wrap the high-level orchestrator in services/vllm_orchestrator.py;
# the orchestrator owns lifecycle (preset load -> TQ -> preflight ->
# adapter restart -> post-start checks) so the CLI is purely arg-parsing +
# result formatting.
# ---------------------------------------------------------------------------


def _build_options(
    tq: bool | None,
    no_tq: bool,
    dry_run: bool,
    no_wait: bool,
    force: bool = False,
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
        force=force,
        wait_for_ready=not no_wait,
    )


@app.command("help")
def help_cmd(
    ctx: typer.Context,
    command: Annotated[
        list[str] | None, typer.Argument(help="Command (and subcommand) to explain.")
    ] = None,
    show_all: Annotated[
        bool, typer.Option("--all", "-a", help="Also list the advanced and legacy commands.")
    ] = False,
) -> None:
    """Show help for llmctl or for COMMAND (same as `llmctl COMMAND --help`)."""
    root = ctx.find_root()
    group = root.command
    target_ctx = root
    if show_all and not command:
        console.print(group.get_help(target_ctx), highlight=False, markup=False)
        console.print("\nAdvanced and legacy commands (all still work):")
        names = group.list_commands(target_ctx)
        for name in names:
            sub = group.get_command(target_ctx, name)
            if sub is None or not getattr(sub, "hidden", False):
                continue
            summary = " ".join((sub.help or "").split())
            summary = summary.split(". ")[0].rstrip(".")
            if len(summary) > 70:
                summary = summary[:67] + "..."
            console.print(f"  {name:<26} {summary}", highlight=False, markup=False)
        return
    for name in command or []:
        # Typer >= 0.27 vendors click, so its groups are not click.Group instances.
        get = getattr(group, "get_command", None)
        sub = get(target_ctx, name) if get is not None else None
        if sub is None:
            raise typer.BadParameter(f"no such command: {' '.join(command or [])}")
        target_ctx = click.Context(sub, info_name=name, parent=target_ctx)
        group = sub
    console.print(group.get_help(target_ctx), highlight=False, markup=False)


_RUN_HELP = (
    "Run a model: the one verb for every engine. NAME is anything `llmctl ls` lists "
    "(a model you added, a vLLM preset, a lane). Whatever holds the GPUs is stopped "
    "first, NAME is started, and a failed start puts the previous model back. "
    "Ollama and LM Studio models need nothing started: they load on first request. "
    "With no NAME, pick from the list. Shows what stops and starts before asking. "
    "--dry-run previews without changing anything."
)
_OLD_RUN_NAMES = ("swap", "switch", "vllm")


@app.command(
    "run",
    help=_RUN_HELP,
    short_help="Run a model on its engine: stops what holds the GPUs, starts it, waits.",
)
@app.command("swap", help=_RUN_HELP, hidden=True)
@app.command("switch", help=_RUN_HELP, hidden=True)
@app.command("vllm", help=_RUN_HELP, hidden=True)
def vllm_cmd(
    ctx: typer.Context,
    preset: Annotated[
        str | None,
        typer.Argument(
            help="Preset, lane, or registered model name (see --list). Omit to pick."
        ),
    ] = None,
    list_only: Annotated[
        bool, typer.Option("--list", "-l", help="Print the swap list and exit.")
    ] = False,
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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Restart even if the preset's local model path is missing.",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
    tp: Annotated[
        int | None,
        typer.Option("--tp", min=1, help="GPUs to split the model across (this run)."),
    ] = None,
    max_len: Annotated[
        int | None,
        typer.Option("--max-len", min=1, help="Context length in tokens (this run)."),
    ] = None,
    gpu_mem: Annotated[
        float | None,
        typer.Option(
            "--gpu-mem", min=0.05, max=1.0, help="Fraction of GPU memory to use (this run)."
        ),
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save", help="Keep --tp/--max-len/--gpu-mem for future runs."),
    ] = False,
) -> None:
    """Run NAME on its engine (``run``; ``swap``, ``switch`` and ``vllm`` are old names)."""
    from llmctl.services.run_overrides import RunOverrides

    overrides = RunOverrides(tp=tp, max_len=max_len, gpu_mem=gpu_mem)
    if save and not overrides:
        raise typer.BadParameter("--save keeps --tp/--max-len/--gpu-mem; give at least one")
    if ctx.info_name in _OLD_RUN_NAMES:
        cli_registry.err_console.print(
            f"[dim]note: `llmctl {escape(str(ctx.info_name))}` is now `llmctl run` (same command); "
            "the old name keeps working.[/dim]"
        )
    settings = load_settings()
    state = gather_state(settings)
    if list_only:
        cli_swap.render_table(state, console)
        return
    target = cli_swap.pick(state, console) if preset is None else resolve(preset, state.catalog)
    if target is None and preset is not None and preset.isdecimal():
        # `llmctl ls` numbers its rows; an exact key/model/alias already won above.
        number = int(preset)
        if not 1 <= number <= len(state.catalog):
            console.print(
                f"[red]No row {escape(preset)}.[/red] Pick a number in "
                f"1-{len(state.catalog)}, or run `llmctl ls` to see what you can run."
            )
            raise typer.Exit(2)
        target = state.catalog[number - 1]
    if target is None and preset is not None:
        # Not a preset or lane: maybe a registered model of another engine.
        routed = _swap_registry_model(
            preset, state, settings, dry_run=dry_run, yes=yes, overrides=overrides, save=save
        )
        if routed is not None:
            target, state = routed
    session_parameters: dict[str, Any] = {}
    if target is not None and target.kind == "session" and overrides:
        target, session_parameters = _session_overrides(target, overrides, save=save)
    if target is None:
        # Not in the list: hand it to the vLLM path, which owns preset lookup
        # and fails with the available presets (exit 2) if it is unknown.
        target = SwapTarget(
            key=str(preset), kind="vllm", model=str(preset), engine="vLLM", upstream=""
        )
    elif target.kind in ("lane", "session") and (tq or no_tq or force or no_wait):
        raise typer.BadParameter("--tq/--no-tq/--force/--no-wait apply to vLLM presets only")
    elif target.kind == "lane" and overrides:
        raise typer.BadParameter(
            f"{target.key} is a router lane started by its own script, so llmctl cannot "
            "change its settings. --tp/--max-len/--gpu-mem apply to vLLM presets and "
            "models llmctl starts."
        )
    elif not cli_swap.print_preview(state, target, console):
        return
    if overrides:
        console.print(f"  for this run: {escape(overrides.describe())}")
    if target.kind in ("lane", "session") and dry_run:
        if save and target.kind == "session":
            _save_run_settings(target, overrides, dry_run=True)
        return
    if not dry_run:
        _confirm_state_change(
            f"Swap to {target.model} ({target.engine})",
            required=settings.scheduler.require_confirmation_for_start,
            assume_yes=yes,
        )
    if target.kind == "session":
        cli_swap.run_session_swap(state, target, console, parameters=session_parameters)
        if save:
            _save_run_settings(target, overrides, dry_run=False)
        return
    if target.kind == "lane":
        cli_swap.run_lane_swap(state, target, console)
        return
    preset = target.key
    # Only a name from the list frees the GPUs up front: an unlisted one is
    # almost always a typo, and start_vllm_tp rejects it before touching
    # anything. Stopping FreeToken first would turn a typo into an outage.
    listed = target in state.catalog
    stopped_lanes = cli_swap.stop_lanes_for_vllm(state, console) if listed and not dry_run else []
    started_at = time.monotonic()
    options = _build_options(tq if tq else None, no_tq, dry_run, no_wait, force)
    options.preset_overrides = overrides.fields()
    try:
        result = start_vllm_tp(
            preset,
            managed_unit=settings.managed_units.vllm_engine,
            defaults=settings.vllm.defaults,
            fleet=settings.managed_units.fleet,
            options=options,
        )
    except UnknownPresetError as exc:
        unknown = exc.args[0] if exc.args else preset
        console.print(f"[red]No model named '{escape(str(unknown))}'.[/red]")
        close = close_matches(str(unknown), state.catalog)
        if close:
            console.print(f"Did you mean: {escape(', '.join(close))}?")
        console.print("Run `llmctl ls` to see what you can run.")
        cli_swap.restore_lanes(stopped_lanes, console)
        raise typer.Exit(2) from exc
    except MissingModelPathError as exc:
        console.print(f"[red]Refusing to start {escape(str(preset))}:[/red] {escape(str(exc))}")
        cli_swap.restore_lanes(stopped_lanes, console)
        raise typer.Exit(2) from exc
    except LauncherEnvError as exc:
        # Nothing was touched: the env file is rendered before any stop.
        console.print(f"[red]Cannot render the launch env:[/red] {escape(str(exc))}")
        cli_swap.restore_lanes(stopped_lanes, console)
        raise typer.Exit(2) from exc

    if result.dry_run:
        if save:
            _save_run_settings(target, overrides, dry_run=True)
        return
    if not result.ok:
        console.print("[red]vLLM start did not complete cleanly.[/red]")
        cli_swap.restore_lanes(stopped_lanes, console)
        if result.fleet_failed:
            console.print(
                f"  fleet preflight failed on: {escape(', '.join(result.fleet_failed))}"
            )
        if result.restart is not None and result.restart.error:
            console.print(f"  restart: {escape(result.restart.error)}")
        _print_rollback(result, settings.managed_units.vllm_engine.unit_name)
        raise typer.Exit(1)
    if result.restart is not None and result.restart.ready is None:
        # --no-wait: the restart was issued and nobody probed /v1/models.
        # Announcing "ready" here would be a claim no code checked.
        console.print(
            f"[cyan]Restart issued[/cyan] for {escape(result.spec.served_name)} "
            f"on port {result.spec.port}; readiness not checked (--no-wait). "
            "Run `llmctl status` once it has loaded."
        )
        _print_companions(result)
        if save:
            _save_run_settings(target, overrides, dry_run=False)
        return
    record_load(preset, time.monotonic() - started_at)
    console.print(
        f"[green]vLLM ready[/green] — serving {escape(result.spec.served_name)} "
        f"on port {result.spec.port}"
    )
    _print_companions(result)
    if save:
        _save_run_settings(target, overrides, dry_run=False)


#: Engines whose launch settings llmctl owns (it starts their servers).
_STARTABLE_ENGINES = ("vllm", "sglang", "freetoken", "strata", "llama_cpp")


@app.command(
    "settings",
    short_help="Show (or --edit) the settings `llmctl run NAME` starts a model with.",
)
def settings_cmd(
    name: Annotated[str, typer.Argument(help="Preset or model name (see `llmctl ls`).")],
    edit: Annotated[
        bool, typer.Option("--edit", help="Open the preset file in $VISUAL/$EDITOR.")
    ] = False,
) -> None:
    """Show the GPUs, context length and memory share NAME runs with, and where
    they are stored. Change them for one run with `llmctl run NAME --tp N
    --max-len N --gpu-mem F`; add --save to keep them."""
    import os
    import shlex
    import subprocess

    from llmctl.services.run_overrides import FIELDS, FLAGS, preset_source, profile_source
    from llmctl.services.serve import AmbiguousModel, find_model

    source = preset_source(name)
    if source is None:
        with _session() as db:
            try:
                model = find_model(db, name)
            except AmbiguousModel as exc:
                raise typer.BadParameter(str(exc)) from exc
            if model is None:
                raise typer.BadParameter(f"No preset or model named {name} (see `llmctl ls`).")
            from llmctl.db import ModelRecord

            record = db.get(ModelRecord, model.id) if model.id else None
            if record is None or record.runtime.value == "vllm":
                console.print(
                    f"{escape(name)} has no preset yet; `llmctl run {escape(name)}` writes one."
                )
                return
            if record.runtime.value not in _STARTABLE_ENGINES:
                console.print(
                    f"{escape(name)} is served by {escape(record.runtime.value)}, which keeps "
                    "its own settings; llmctl does not change them."
                )
                return
            source = profile_source(db, record)
    labels = {
        "tensor_parallel_size": "GPUs (tensor parallel)",
        "max_model_len": "context length",
        "gpu_memory_utilization": "GPU memory share",
    }
    table = Table(title=f"{name} settings", show_header=True)
    table.add_column("setting")
    table.add_column("run flag")
    table.add_column("value")
    for key in FIELDS:
        value = source.values.get(key)
        table.add_row(labels[key], FLAGS[key], "engine default" if value is None else str(value))
    console.print(table)
    where = f"{source.label}: {source.path}" if source.path else source.label
    console.print(f"Stored in {escape(where)}")
    console.print(
        f"One run: llmctl run {escape(name)} --max-len 32768   ·   keep it: add --save"
    )
    if not edit:
        return
    if source.path is None:
        raise typer.BadParameter(
            f"{name}'s settings live in llmctl's database, not a file. Change them with "
            f"`llmctl run {name} --tp/--max-len/--gpu-mem ... --save`."
        )
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
    subprocess.call([*shlex.split(editor), str(source.path)])
    if preset_source(name) is None:
        console.print(
            f"[red]{escape(str(source.path))} no longer loads as a preset.[/red] "
            "Fix it, or `llmctl run` will not find it."
        )
        raise typer.Exit(1)


def _session_overrides(
    target: SwapTarget, overrides: Any, *, save: bool
) -> tuple[SwapTarget, dict[str, Any]]:
    """Translate run flags for a model llmctl starts as a session (llama.cpp)."""
    import dataclasses

    from llmctl.db import ModelRecord
    from llmctl.services.run_overrides import OverrideError, session_parameters
    from llmctl.services.session_swap import Placement

    with _session() as db:
        record = db.get(ModelRecord, target.model_id) if target.model_id else None
        if record is None:
            raise typer.BadParameter(f"model {target.key} is no longer registered")
        runtime = record.runtime
    try:
        parameters = session_parameters(runtime, overrides)
    except OverrideError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if overrides.tp is not None:
        if save:
            raise typer.BadParameter(
                "llama.cpp picks its GPUs from the model's size on every run, so --tp "
                "cannot be saved. Drop --save to use it for this run."
            )
        parameters.pop("tensor_parallel_size", None)
        target = dataclasses.replace(
            target, placement=Placement(cards=overrides.tp)
        )
    return target, parameters


def _save_run_settings(target: SwapTarget, overrides: Any, *, dry_run: bool) -> None:
    """``run --save``: write the flags into the preset or profile NAME runs from."""
    from llmctl.db import ModelRecord
    from llmctl.presets import load_all_records
    from llmctl.services.run_overrides import (
        OverrideError,
        save_to_preset,
        save_to_profile,
        session_parameters,
    )

    try:
        if target.kind == "vllm":
            record = load_all_records().get(target.key)
            if record is None:
                raise OverrideError(f"Not saved: no preset named {target.key}.")
            where = str(record.source_path)
            if dry_run:
                console.print(f"Would save {escape(overrides.describe())} to {escape(where)}")
                return
            changes = save_to_preset(record.source_path, overrides.fields())
        else:
            with _session() as db:
                model = db.get(ModelRecord, target.model_id) if target.model_id else None
                if model is None:
                    raise OverrideError(f"Not saved: model {target.key} is not registered.")
                parameters = session_parameters(model.runtime, overrides)
                if dry_run:
                    console.print(
                        f"Would save {escape(overrides.describe())} to {escape(model.name)}'s "
                        "profile"
                    )
                    return
                name, changes = save_to_profile(db, model, parameters)
                where = f"profile {name}"
    except OverrideError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if changes:
        console.print(f"[green]Saved[/green] to {escape(where)}: {escape('; '.join(changes))}")
    else:
        console.print(f"Already saved in {escape(where)}; nothing changed.")


def _swap_registry_model(
    name: str,
    state: Any,
    settings: Any,
    *,
    dry_run: bool,
    yes: bool,
    overrides: Any = None,
    save: bool = False,
) -> tuple[SwapTarget, Any] | None:
    """Serve a registered model that is not itself a preset or lane.

    Returns the swap-list row to continue with (a vLLM preset, or a llama.cpp
    model that swap frees the GPUs for), or None when NAME is not a registered
    model. The rest exit here: Ollama and LM Studio need nothing started, and
    the others say why llmctl cannot start them.
    """
    from llmctl.services.preset_scaffold import ensure_starter_preset
    from llmctl.services.serve import AmbiguousModel, find_model, route

    with _session() as db:
        try:
            model = find_model(db, name)
        except AmbiguousModel as exc:
            raise typer.BadParameter(str(exc)) from exc
        if model is None:
            return None
        served = route(db, model, state.catalog)
        if overrides and served.kind in ("on-demand", "cannot"):
            raise typer.BadParameter(
                f"{model.name} runs on {model.runtime.value}, which llmctl does not start, "
                "so --tp/--max-len/--gpu-mem cannot apply."
            )
        console.print(escape(served.message))
        if served.kind == "preset" and served.target is not None:
            return served.target, state
        if served.kind in ("session", "lane") and served.target is not None:
            # The normal preview -> confirm -> swap path takes it from here.
            return served.target, state
        if served.kind == "on-demand":
            raise typer.Exit(0)
        if served.kind == "cannot":
            raise typer.Exit(1)
        if served.kind == "start":
            from llmctl.db import ProfileRecord
            from llmctl.services.run_overrides import OverrideError, session_parameters

            try:
                parameters = session_parameters(model.runtime, overrides) if overrides else {}
            except OverrideError as exc:
                raise typer.BadParameter(str(exc)) from exc
            # The model's own profile, so settings `run --save` stored apply.
            own = (
                db.get(ProfileRecord, model.default_profile_id)
                if model.default_profile_id
                else None
            )
            request = _build_start_request(
                db,
                model.id or name,
                own.name if own else None,
                "auto",
                False,
                False,
                dry_run,
                parameters=parameters,
            )
            row = SwapTarget(
                key=model.name,
                kind="session",
                model=model.name,
                engine=model.runtime.value,
                upstream="",
                model_id=model.id,
            )
            if overrides:
                console.print(f"  for this run: {escape(overrides.describe())}")
            if dry_run:
                # A plan, not `start --dry-run`: a swap preview writes nothing.
                _print_launch_plan(SessionService(db).plan(request))
                if save:
                    _save_run_settings(row, overrides, dry_run=True)
                raise typer.Exit(0)
            _confirm_state_change(
                f"Start {model.name} ({model.runtime.value})",
                required=settings.scheduler.require_confirmation_for_start,
                assume_yes=yes,
            )
            _start_request(db, request)
            if save:
                _save_run_settings(row, overrides, dry_run=False)
            raise typer.Exit(0)
        # no-preset: a vLLM checkpoint swap cannot serve until a preset exists.
        if dry_run:
            console.print("Would write a starter preset for it, then swap to it.")
            raise typer.Exit(0)
        if not yes and not Confirm.ask("Write a starter preset for it now?", default=True):
            console.print("[yellow]Nothing changed.[/yellow]")
            raise typer.Exit(0)
        try:
            starter = ensure_starter_preset(db, model)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(1) from exc
    console.print(f"[green]Wrote starter preset[/green] {escape(starter.alias)}")
    for note in starter.notes:
        console.print(f"  {escape(note)}")
    fresh = gather_state(settings)
    target = resolve(starter.alias, fresh.catalog)
    if target is None:
        console.print(f"[red]Preset {escape(starter.alias)} was written but is not listed.[/red]")
        raise typer.Exit(1)
    return target, fresh


def _print_companions(result: object) -> None:
    """Say which paused services (ollama, companions) came back, or how to."""
    restarted = list(getattr(result, "companions_restarted", []) or [])
    failed = list(getattr(result, "companions_failed", []) or [])
    pending = list(getattr(result, "companions_pending", []) or [])
    if restarted:
        console.print(f"  started again: {escape(', '.join(restarted))}")
    hints = dict(getattr(result, "start_hints", {}) or {})
    for name in failed + pending:
        how = hints.get(name, f"sudo systemctl start {name}")
        why = "could not start" if name in failed else "not restarted (--no-wait)"
        console.print(f"  [yellow]{escape(name)} {why}[/yellow] — run: {escape(how)}")


def _print_rollback(result: object, tp_unit: str) -> None:
    """Say what a failed swap put back and what the operator must still do.

    The orchestrator rolls back on failure (previous env restored, stopped
    fleet units started again); anything it could not undo is printed as a
    command to run, so a failed swap never silently leaves nothing serving.
    """
    rollback = getattr(result, "rollback", None)
    stopped: list[str] = list(getattr(result, "fleet_stopped", []) or [])
    stopped += list(getattr(result, "companions_stopped", []) or [])
    hints = dict(getattr(result, "start_hints", {}) or {})

    def how(name: str) -> str:
        return hints.get(name, f"sudo systemctl start {name}")

    todo: list[str] = []
    if rollback is None:
        if stopped:
            console.print(
                f"  [yellow]still stopped:[/yellow] {escape(', '.join(stopped))}"
            )
            todo += [how(unit) for unit in stopped]
    else:
        if rollback.env_restored:
            console.print(f"  rollback: restored previous {escape(str(rollback.env_path))}")
        elif rollback.env_restored is False:
            console.print(
                f"  [yellow]rollback: previous env NOT restored[/yellow] "
                f"({escape(str(rollback.env_path))} holds the failed preset)"
            )
            todo.append("llmctl vllm <previous-preset>  # rewrite a known-good env")
        if rollback.tp_stopped:
            console.print(f"  rollback: stopped half-started {escape(tp_unit)}")
        if rollback.restarted:
            console.print(
                f"  [green]rollback: started again:[/green] "
                f"{escape(', '.join(rollback.restarted))}"
            )
        if rollback.restart_failed:
            console.print(
                f"  [red]rollback: could not start:[/red] "
                f"{escape(', '.join(rollback.restart_failed))}"
            )
            todo += [how(unit) for unit in rollback.restart_failed]
    todo.append(f"journalctl -u {tp_unit} -n 80 --no-pager  # why the launch failed")
    console.print("  next:")
    for cmd in todo:
        console.print(f"    {escape(cmd)}")


@app.command("setup", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def setup_cmd(
    ctx: typer.Context,
    engine: Annotated[
        str | None,
        typer.Argument(help="vllm, sglang, freetoken, strata, llama_cpp or ollama. Omit to list."),
    ] = None,
    python: Annotated[
        str | None,
        typer.Option(help="Python to build the engine's env from (default: uv's choice)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the steps and do nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    from_source: Annotated[
        bool,
        typer.Option(
            "--from-source",
            help="llama.cpp: build it with git + cmake instead of downloading a build "
            "(automatic when glibc is older than the prebuilt builds need).",
        ),
    ] = False,
) -> None:
    """Install an engine so `llmctl run` can use it (`llmctl setup` alone shows them).

    Extra flags after `--` go to the engine's own installer (Strata:
    `llmctl setup strata -- --family coder --model IQ1_M`).
    """
    from llmctl.services import engines as eng

    settings = load_settings()
    if engine is None:
        table = Table(title="Engines", title_justify="left")
        for column in ("engine", "state", "binary", "version"):
            table.add_column(column)
        for row in eng.engine_statuses(settings):
            if row.source == "lane":
                state = f"[green]via lane {escape(row.binary or '?')}[/green]"
                table.add_row(row.engine, state, "—", escape(row.version or "—"))
                continue
            state = {
                "llmctl": "[green]installed by llmctl[/green]",
                "path": "[green]found[/green]",
                "missing": "[yellow]missing[/yellow]",
            }[row.source]
            table.add_row(row.engine, state, escape(row.binary or "—"), escape(row.version or "—"))
        console.print(table)
        console.print("Install one: llmctl setup ENGINE")
        return
    try:
        plan = eng.plan_setup(
            engine,
            settings=settings,
            python=python,
            extra=list(ctx.args),
            from_source=from_source,
        )
    except eng.SetupError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc
    if plan.problem:
        console.print(f"[red]Cannot set up {escape(plan.engine)}:[/red] {escape(plan.problem)}")
        raise typer.Exit(1)
    console.print(f"[bold]Set up {escape(plan.engine)}[/bold]")
    for step in plan.steps:
        console.print(f"  - {escape(step)}")
    if plan.needs_sudo:
        console.print("  [yellow]this installer asks for sudo[/yellow]")
    if dry_run:
        console.print("Dry run: nothing was done.")
        return
    if not yes and not Confirm.ask("Proceed?", default=True):
        console.print("[yellow]Nothing changed.[/yellow]")
        raise typer.Exit(0)
    deps = eng.SetupDeps(log=lambda m: console.print(escape(m)))
    try:
        record = eng.run_setup(plan, settings=settings, deps=deps)
    except eng.SetupError as exc:
        console.print(f"[red]Setup failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    binary = escape(str(record.get("binary")))
    version = escape(str(record.get("version") or ""))
    suffix = f" (version {version})" if version else ""
    console.print(f"[green]{escape(plan.engine)} ready[/green] — {binary}{suffix}")
    if plan.engine == "strata" and record.get("config"):
        _register_strata_config(Path(str(record["config"])))
    console.print(f"Recorded in {escape(str(eng.engines_file(settings)))}.")
    console.print(
        "Next: llmctl add MODEL  (then llmctl run MODEL)"
        if plan.engine != "strata"
        else "Next: llmctl run  (the model setup installed is listed)"
    )


def _register_strata_config(config: Path) -> None:
    """Register the model a Strata setup produced, so `run` can serve it."""
    from llmctl.services.model_ops import ModelOps, ModelOpsError, plan_add

    try:
        plan = plan_add(str(config), "strata")
        with _session() as db:
            result = ModelOps(db).add(plan)
        console.print(f"  registered {escape(result.model.name)} ({escape(str(config))})")
    except (ModelOpsError, Exception) as exc:  # noqa: BLE001 - setup succeeded; report only
        console.print(
            f"  [yellow]could not register {escape(str(config))}: {escape(str(exc))}[/yellow] "
            f"— run: llmctl add {escape(str(config))} --engine strata"
        )


@app.command("ls")
def ls_cmd() -> None:
    """List every model you can run and which one is serving now."""
    settings = load_settings()
    cli_swap.render_table(gather_state(settings), console)


@app.command("presets", hidden=True)
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
    _id_column(table, "alias", style="cyan")
    _id_column(table, "served name")
    # Deliberately NOT an id column. It is the longest field in a seven-column
    # table, so folding it wraps every single row and buries the alias list
    # this command exists to show. It is reference-only here; `--json` has it.
    table.add_column("model id")
    table.add_column("family")
    table.add_column("size (B)", justify="right")
    table.add_column("tp", justify="right")
    table.add_column("quant")
    for v in views:
        table.add_row(
            v.alias,
            escape(str(v.served_name)),
            v.model_id,
            v.family or "-",
            f"{v.param_count_b:.0f}" if v.param_count_b else "-",
            str(v.tensor_parallel),
            v.quantization,
        )
    console.print(table)


@app.command("status")
def status_cmd(json_out: _JSON_OPT = False) -> None:
    """Managed systemd units (vllm-tp, ...): unit state, env config, served models.

    Siblings: `health` = per-runtime live probe; `doctor` = full checkup;
    `validate` = recorded paths still exist; `router-status` = gateway routes.
    """
    from llmctl.services.units import unit_rows

    # Shared with the TUI Units screen -- see services/units.py. The JSON shape
    # below is the published contract and is kept byte-identical.
    rows = [
        {
            "role": r.role,
            "unit_name": r.unit_name,
            "env_file": str(r.env_file),
            "env_file_exists": r.env_file_exists,
            "port": r.port,
            "source": r.source,
            "serving": r.serving,
            "served_models": r.served or [],
        }
        for r in unit_rows(load_settings())
    ]
    if json_out:
        _emit_json(rows)
        return
    table = Table(title="Managed units")
    _id_column(table, "role", style="cyan")
    _id_column(table, "unit name")
    _id_column(table, "env file")
    table.add_column("port", justify="right")
    table.add_column("source")
    _id_column(table, "serving")
    for row in rows:
        # Served model ids and env paths are process/config derived and may
        # contain brackets; escape them so a name cannot eat the styling.
        serving = (
            f"[green]{escape(', '.join(row['served_models']))}[/green]"
            if row["serving"] and row["served_models"]
            else ("[green]yes (empty list)[/green]" if row["serving"] else "[red]no[/red]")
        )
        env_cell = escape(row["env_file"]) if row["env_file_exists"] else "[dim]—[/dim]"
        source = (
            row["source"]
            if row["source"] == "settings"
            else f"[cyan]{escape(row['source'])}[/cyan]"
        )
        table.add_row(
            row["role"],
            escape(row["unit_name"]),
            env_cell,
            str(row["port"]),
            source,
            serving,
        )
    console.print(table)


@app.command("validate", hidden=True)
def validate_cmd(json_out: _JSON_OPT = False) -> None:
    """Registry integrity: does everything llmctl records still exist where recorded?

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
        *validate_svc.check_managed_unit_ports(
            list(settings.managed_units.roles().values())
        ),
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
    _id_column(table, "Target", style="cyan")
    table.add_column("Detail")
    for finding in findings:
        table.add_row(finding.check, finding.target, escape(str(finding.detail)))
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


@app.command("gateway", hidden=True)
def gateway_cmd(
    host: Annotated[str | None, typer.Option(help="Bind host; defaults to router.host.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port; defaults to router.port.")] = None,
) -> None:
    """Serve llmctl's own OpenAI-compatible alias router (:9000).

    Not the LiteLLM front door on :4000 that every client uses; for that see
    `ccl --status` and ~/AI/docs/GATEWAY-QUICKSTART.md.
    """
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


@app.command("router-status", hidden=True)
def router_status_cmd() -> None:
    """The llmctl gateway (`llmctl gateway`): is it up, and which alias routes where?

    Not the LiteLLM gateway on :4000. Siblings: `health`, `status`, `doctor`,
    `validate`.
    """
    import httpx

    settings = load_settings()
    url = f"{_router_url(settings)}/health"
    headers = {}
    if settings.router.auth_token:
        headers["Authorization"] = f"Bearer {settings.router.auth_token}"
    try:
        response = httpx.get(url, headers=headers, timeout=2.0)
    except httpx.HTTPError as exc:
        console.print(f"[red]Gateway unreachable[/red] at {escape(str(url))}: {escape(str(exc))}")
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
        f"[green]Gateway up[/green] at {escape(str(url))}  "
        f"auth={'on' if router.get('auth_required') else 'off'}  "
        f"auto_start={router.get('auto_start')}  "
        f"fallback={router.get('fallback_policy')}"
    )
    aliases = payload.get("aliases", [])
    table = Table(title="Aliases")
    _id_column(table, "alias", style="cyan")
    _id_column(table, "target")
    _id_column(table, "session")
    _id_column(table, "served name")
    table.add_column("healthy")
    for entry in aliases:
        healthy = entry.get("healthy")
        color = "green" if healthy else "yellow" if entry.get("target") else "red"
        table.add_row(
            escape(str(entry.get("name", ""))),
            entry.get("target") or "-",
            entry.get("session_id") or "-",
            escape(str(entry.get("served_name") or "-")),
            f"[{color}]{'yes' if healthy else 'no'}[/{color}]",
        )
    console.print(table)


@app.command("aliases", hidden=True)
def aliases_cmd() -> None:
    """List configured router aliases and their resolved targets."""
    from llmctl.services.gateway import GatewayService

    settings = load_settings()
    with _session() as db:
        rows = GatewayService(db, settings).alias_view()

    table = Table(title="Router aliases")
    _id_column(table, "alias", style="cyan")
    _id_column(table, "public id")
    _id_column(table, "target")
    _id_column(table, "session")
    _id_column(table, "served name")
    table.add_column("status")
    for row in rows:
        if row.target is None:
            status_text = "[red]unbound[/red]"
        elif row.healthy:
            status_text = "[green]healthy[/green]"
        else:
            status_text = "[yellow]bound, no active session[/yellow]"
        table.add_row(
            escape(str(row.name)),
            f"local-{escape(str(row.name))}",
            row.target or "-",
            row.resolved_session_id or "-",
            escape(str(row.resolved_served_name or "-")),
            status_text,
        )
    console.print(table)


@app.command("set-alias", hidden=True)
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
                f"(session {view.resolved_session_id}, "
                f"served as {escape(str(view.resolved_served_name))})."
            )


# ---------------------------------------------------------------------------
# Adopt flow — track externally-managed runtime endpoints (systemd vLLM units,
# manually-started llama.cpp servers, etc.) as kind=ADOPTED sessions so the
# gateway can route to them without llmctl ever spawning a process.
# ---------------------------------------------------------------------------


def _managed_roles(settings: Settings) -> dict[str, ManagedUnitConfig]:
    """Adoptable managed-unit roles by CLI name.

    The built-in ``vllm-engine`` plus every role in ``managed_units.units`` —
    so ``llmctl adopt-managed <role>`` and ``--all`` cover units of any
    runtime, not only the vLLM TP pair.
    """
    return settings.managed_units.roles()


def _print_adopted(session: object, *, prefix: str = "Adopted") -> None:
    """Render an adopt confirmation line shared by adopt + adopt-managed."""
    console.print(
        f"[green]{prefix}[/green] {session.id}  "  # type: ignore[attr-defined]
        f"runtime={session.runtime.value}  "  # type: ignore[attr-defined]
        f"served_name={escape(str(session.served_name or '-'))}  "  # type: ignore[attr-defined]
        f"endpoint={escape(str(session.endpoint_url))}  "  # type: ignore[attr-defined]
        f"unit={escape(str(session.systemd_unit or '-'))}"  # type: ignore[attr-defined]
    )


@app.command("adopt", hidden=True)
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
            console.print(f"[red]Adopt failed:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    _print_adopted(session)


@app.command("adopt-managed", hidden=True)
def adopt_managed_cmd(
    role: Annotated[
        str | None,
        typer.Argument(
            help="Managed-unit role to adopt (vllm-engine, or any role from "
            "managed_units.units). Omit and pass --all to adopt every "
            "running role.",
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
        raise typer.BadParameter("Pass a role (e.g. vllm-engine) or --all.")
    if role is not None and all_roles:
        raise typer.BadParameter("Pass a role OR --all, not both.")

    settings = load_settings()
    roles = _managed_roles(settings)
    if all_roles:
        targets = list(roles.items())
    else:
        role = resolve_role(role or "")
        unit = roles.get(role)
        if unit is None:
            raise typer.BadParameter(
                f"Unknown role '{role}'. Choose: {', '.join(sorted(roles))}."
            )
        targets = [(role, unit)]

    adopted = 0
    skipped = 0
    with _session() as db:
        service = SessionService(db)
        for cli_role, unit_cfg in targets:
            try:
                unit_runtime = RuntimeName(unit_cfg.runtime)
            except ValueError:
                skipped += 1
                console.print(
                    f"[yellow]Skip {cli_role}[/yellow]: configured runtime "
                    f"'{escape(str(unit_cfg.runtime))}' is not a known RuntimeName."
                )
                continue
            endpoint = f"http://127.0.0.1:{unit_cfg.default_port}"
            try:
                session = service.adopt(
                    unit_runtime,
                    endpoint,
                    systemd_unit=f"{unit_cfg.unit_name}.service",
                    timeout_s=timeout,
                )
            except AdoptError as exc:
                skipped += 1
                console.print(
                    f"[yellow]Skip {cli_role}[/yellow] "
                    f"({escape(str(endpoint))}): {escape(str(exc))}"
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


@app.command("detach", hidden=True)
def detach_cmd(
    session_id: Annotated[str, typer.Argument(help="ADOPTED session id to remove from tracking.")],
) -> None:
    """Remove an adopted session from llmctl tracking (upstream untouched)."""
    with _session() as db:
        service = SessionService(db)
        try:
            session = service.detach(session_id)
        except AdoptError as exc:
            console.print(f"[red]Detach refused:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
    if session is None:
        raise typer.BadParameter(f"Session not found: {session_id}")
    console.print(
        f"[green]Detached[/green] {session_id} "
        f"(endpoint={escape(str(session.endpoint_url))}, "
        f"unit={escape(str(session.systemd_unit or '-'))}). "
        "The upstream process was not touched."
    )
