"""CLI commands for the model registry and profile management.

Wired into the main Typer app via :func:`register`. Kept in a separate module
so the management surface stays comprehensible while the legacy ``cli.py``
keeps its existing structure for launch/orchestration commands.

Sub-typers exposed:

* ``llmctl model <action>``     — CRUD over registered models
* ``llmctl profile <action>``   — CRUD over launch profiles

Top-level commands added:

* ``llmctl profiles``           — list profiles
* ``llmctl preview``            — dry-run launch plan for a model+profile
* ``llmctl export-registry``    — write registry bundle to JSON
* ``llmctl import-registry``    — read registry bundle from JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sqlmodel import Session

from llmctl.config import load_settings
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import (
    ModelCreate,
    ModelUpdate,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    RegistryExport,
    SessionStartRequest,
    ValidationIssue,
)
from llmctl.services.profiles import ProfileService
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import SessionService


console = Console()

model_app = typer.Typer(
    name="model",
    help="Manage registered models.",
    no_args_is_help=True,
)
profile_app = typer.Typer(
    name="profile",
    help="Manage launch profiles.",
    no_args_is_help=True,
)


_RUNTIME_CHOICES = [r.value for r in RuntimeName]


def _session() -> Session:
    settings = load_settings()
    init_db(settings.database_url)
    return Session(get_engine(settings.database_url))


def _resolve_model_or_die(service: RegistryService, name_or_id: str) -> str:
    """Return the model id for ``name_or_id`` or raise a CLI error."""
    try:
        model = service.find(name_or_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if model is None or model.id is None:
        raise typer.BadParameter(f"Model not found: {name_or_id}")
    return model.id


def _resolve_profile_or_die(service: ProfileService, name_or_id: str) -> Profile:
    profile = service.find(name_or_id)
    if profile is None:
        raise typer.BadParameter(f"Profile not found: {name_or_id}")
    return profile


def _print_validation(issues: list[ValidationIssue]) -> None:
    for issue in issues:
        color = "red" if issue.severity == "error" else "yellow"
        prefix = f"[{color}]{issue.severity}[/{color}]"
        field = f" {issue.field}" if issue.field else ""
        console.print(f"{prefix}{field}: {issue.message}")


def _prompt_optional(label: str, default: str | None = None) -> str | None:
    """Prompt for an optional string. Empty input means ``None``."""
    raw = Prompt.ask(label, default=default or "")
    raw = raw.strip()
    return raw or None


def _prompt_optional_int(label: str, default: int | None = None) -> int | None:
    raw = _prompt_optional(label, str(default) if default is not None else None)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"{label}: expected integer") from exc


def _prompt_optional_float(label: str, default: float | None = None) -> float | None:
    raw = _prompt_optional(label, str(default) if default is not None else None)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"{label}: expected number") from exc


def _resolve_gpu_mode(gpus: str) -> str:
    """Classify a ``--gpus`` argument as a named mode or an explicit id list.

    Accepts forms like ``"auto"``, ``"0"``, ``"0,1"``, or ``"0, 1"``. The
    previous one-liner only stripped commas, so ``"0, 1"`` failed the
    ``isdigit`` check and fell through as a literal mode name (later
    rejected by the scheduler). Strip whitespace too.
    """
    return "explicit" if gpus.replace(",", "").replace(" ", "").isdigit() else gpus


def _prompt_runtime(default: RuntimeName | None = None) -> RuntimeName:
    raw = Prompt.ask(
        "backend",
        choices=_RUNTIME_CHOICES,
        default=default.value if default else None,
    )
    return RuntimeName(raw)


def _prompt_tags(default: list[str] | None = None) -> list[str]:
    base = ",".join(default or [])
    raw = Prompt.ask("tags (comma-separated)", default=base)
    return [t.strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# model subcommands
# ---------------------------------------------------------------------------


@model_app.command("show")
def model_show(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
) -> None:
    """Print full details for a single model."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        model = service.get_model(resolved)
    if model is None:
        raise typer.BadParameter(f"Model not found: {model_id}")
    table = Table(title=f"Model: {model.name}", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("id", model.id or "")
    table.add_row("runtime", model.runtime.value)
    table.add_row("status", model.status.value)
    table.add_row("active", "yes" if model.active else "no")
    table.add_row("path", model.path or "")
    table.add_row("source", model.source or "")
    table.add_row("format", model.format or "")
    table.add_row("quantization", model.quantization or "")
    table.add_row(
        "estimated VRAM (GB)",
        f"{model.estimated_vram_gb:.1f}" if model.estimated_vram_gb else "",
    )
    table.add_row(
        "max context", str(model.max_context) if model.max_context else ""
    )
    table.add_row(
        "parameters", f"{model.parameter_count:,}" if model.parameter_count else ""
    )
    table.add_row("notes", model.notes or "")
    table.add_row("default profile id", model.default_profile_id or "")
    table.add_row("tags", ", ".join(model.tags))
    console.print(table)


@model_app.command("add")
def model_add(
    name: Annotated[str | None, typer.Option(help="Display name (skips prompt).")] = None,
    backend: Annotated[
        str | None,
        typer.Option(help=f"Backend: {', '.join(_RUNTIME_CHOICES)} (skips prompt)."),
    ] = None,
    path: Annotated[Path | None, typer.Option(help="Local filesystem path.")] = None,
    quantization: Annotated[str | None, typer.Option(help="Quantization label.")] = None,
    max_context: Annotated[int | None, typer.Option(help="Maximum context length.")] = None,
    estimated_vram: Annotated[float | None, typer.Option(help="Estimated VRAM (GB).")] = None,
    tags: Annotated[
        str | None,
        typer.Option(help="Comma-separated tags."),
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="Fail instead of prompting."),
    ] = False,
) -> None:
    """Register a model, prompting for any unspecified fields."""
    if non_interactive:
        if not name or not backend:
            raise typer.BadParameter(
                "--name and --backend are required in --non-interactive mode"
            )
        runtime = RuntimeName(backend)
        resolved_path = str(path) if path else None
        resolved_tags = [t.strip() for t in (tags or "").split(",") if t.strip()]
        payload = ModelCreate(
            name=name,
            runtime=runtime,
            path=resolved_path,
            source=resolved_path,
            quantization=quantization,
            max_context=max_context,
            estimated_vram_gb=estimated_vram,
            tags=resolved_tags,
        )
    else:
        name_val = name or Prompt.ask("model name")
        runtime_val = (
            RuntimeName(backend) if backend else _prompt_runtime()
        )
        path_val = str(path) if path else _prompt_optional("path")
        quant_val = quantization or _prompt_optional("quantization")
        max_ctx_val = (
            max_context if max_context is not None else _prompt_optional_int("max context")
        )
        vram_val = (
            estimated_vram
            if estimated_vram is not None
            else _prompt_optional_float("estimated VRAM (GB)")
        )
        tags_val = (
            [t.strip() for t in tags.split(",") if t.strip()]
            if tags is not None
            else _prompt_tags()
        )
        payload = ModelCreate(
            name=name_val,
            runtime=runtime_val,
            path=path_val,
            source=path_val,
            quantization=quant_val,
            max_context=max_ctx_val,
            estimated_vram_gb=vram_val,
            tags=tags_val,
        )
    with _session() as db:
        model = RegistryService(db).add_model(payload)
    console.print(f"[green]Registered model[/green] {model.name} ({model.id})")


@model_app.command("edit")
def model_edit(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
    name: Annotated[str | None, typer.Option(help="New display name.")] = None,
    path: Annotated[str | None, typer.Option(help="New local path.")] = None,
    quantization: Annotated[str | None, typer.Option(help="New quantization.")] = None,
    max_context: Annotated[int | None, typer.Option(help="New max context.")] = None,
    estimated_vram: Annotated[
        float | None, typer.Option(help="New estimated VRAM (GB).")
    ] = None,
    notes: Annotated[str | None, typer.Option(help="New notes.")] = None,
    tags: Annotated[
        str | None, typer.Option(help="Comma-separated tags (replaces existing).")
    ] = None,
) -> None:
    """Apply a partial update to a model record."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        updates_dict: dict[str, Any] = {}
        if name is not None:
            updates_dict["name"] = name
        if path is not None:
            updates_dict["path"] = path
        if quantization is not None:
            updates_dict["quantization"] = quantization
        if max_context is not None:
            updates_dict["max_context"] = max_context
        if estimated_vram is not None:
            updates_dict["estimated_vram_gb"] = estimated_vram
        if notes is not None:
            updates_dict["notes"] = notes
        if tags is not None:
            updates_dict["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if not updates_dict:
            raise typer.BadParameter("provide at least one --field to update")
        updated = service.update_model(resolved, ModelUpdate(**updates_dict))
    if updated is None:
        raise typer.BadParameter(f"Model not found: {model_id}")
    console.print(f"[green]Updated[/green] {updated.name} ({updated.id})")


@model_app.command("delete")
def model_delete(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
    delete_files: Annotated[
        bool,
        typer.Option(
            "--delete-files",
            help="Also remove the on-disk artifact at the model's path.",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")
    ] = False,
) -> None:
    """Soft-delete a model. ``--delete-files`` additionally removes the artifact."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        model = service.get_model(resolved)
        assert model is not None
        if delete_files and not yes:
            ok = Confirm.ask(
                f"This will permanently delete files at {model.path!r}. Continue?",
                default=False,
            )
            if not ok:
                raise typer.Exit(code=1)
        deleted = service.delete_model(resolved, delete_files=delete_files)
    if deleted:
        suffix = " (files removed)" if delete_files else ""
        console.print(f"[green]Soft-deleted model[/green] {model_id}{suffix}")
    else:
        raise typer.BadParameter(f"Model not found: {model_id}")


@model_app.command("clone")
def model_clone(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name to clone.")],
    new_name: Annotated[str, typer.Argument(help="Name for the clone.")],
) -> None:
    """Duplicate a model record under a new name."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        clone = service.clone_model(resolved, new_name)
    if clone is None:
        raise typer.BadParameter(f"Model not found: {model_id}")
    console.print(f"[green]Cloned[/green] -> {clone.name} ({clone.id})")


@model_app.command("disable")
def model_disable(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
) -> None:
    """Mark a model as inactive (hidden from default listings)."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        ok = service.disable_model(resolved)
    if not ok:
        raise typer.BadParameter(f"Model not found: {model_id}")
    console.print(f"[yellow]Disabled[/yellow] {model_id}")


@model_app.command("enable")
def model_enable(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
) -> None:
    """Mark a model as active."""
    with _session() as db:
        service = RegistryService(db)
        resolved = _resolve_model_or_die(service, model_id)
        ok = service.enable_model(resolved)
    if not ok:
        raise typer.BadParameter(f"Model not found: {model_id}")
    console.print(f"[green]Enabled[/green] {model_id}")


# ---------------------------------------------------------------------------
# profile subcommands
# ---------------------------------------------------------------------------


@profile_app.command("show")
def profile_show(
    profile: Annotated[str, typer.Argument(help="Profile id or name.")],
) -> None:
    """Print full details for a single profile."""
    with _session() as db:
        service = ProfileService(db)
        prof = _resolve_profile_or_die(service, profile)
    table = Table(title=f"Profile: {prof.name}", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("id", prof.id or "")
    table.add_row("runtime", prof.runtime.value)
    table.add_row("description", prof.description or "")
    table.add_row(
        "tensor_parallel_size",
        str(prof.tensor_parallel_size) if prof.tensor_parallel_size else "",
    )
    table.add_row(
        "max_model_len", str(prof.max_model_len) if prof.max_model_len else ""
    )
    table.add_row(
        "gpu_memory_utilization",
        f"{prof.gpu_memory_utilization:.2f}" if prof.gpu_memory_utilization else "",
    )
    table.add_row("dtype", prof.dtype or "")
    table.add_row("quantization", prof.quantization or "")
    if prof.extra_args:
        table.add_row("extra_args", " ".join(prof.extra_args))
    if prof.environment_variables:
        table.add_row(
            "environment_variables",
            ", ".join(f"{k}={v}" for k, v in prof.environment_variables.items()),
        )
    if prof.parameters:
        table.add_row("parameters", json.dumps(prof.parameters, indent=2))
    console.print(table)


@profile_app.command("create")
def profile_create(
    name: Annotated[str | None, typer.Option(help="Profile name (skips prompt).")] = None,
    backend: Annotated[
        str | None,
        typer.Option(help=f"Backend: {', '.join(_RUNTIME_CHOICES)} (skips prompt)."),
    ] = None,
    description: Annotated[str | None, typer.Option(help="Short description.")] = None,
    tensor_parallel: Annotated[
        int | None, typer.Option(help="tensor_parallel_size.")
    ] = None,
    max_model_len: Annotated[int | None, typer.Option(help="max_model_len.")] = None,
    gpu_memory_utilization: Annotated[
        float | None, typer.Option(help="gpu_memory_utilization (0..1].")
    ] = None,
    dtype: Annotated[str | None, typer.Option(help="dtype (auto, bfloat16, ...).")] = None,
    quantization: Annotated[str | None, typer.Option(help="Quantization label.")] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Fail instead of prompting.")
    ] = False,
) -> None:
    """Create a new profile, prompting for any unspecified fields."""
    if non_interactive:
        if not name or not backend:
            raise typer.BadParameter(
                "--name and --backend are required in --non-interactive mode"
            )
        runtime = RuntimeName(backend)
        payload = ProfileCreate(
            name=name,
            runtime=runtime,
            description=description,
            tensor_parallel_size=tensor_parallel,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            quantization=quantization,
        )
    else:
        name_val = name or Prompt.ask("profile name")
        runtime_val = RuntimeName(backend) if backend else _prompt_runtime()
        desc_val = description or _prompt_optional("description")
        tp_val = (
            tensor_parallel
            if tensor_parallel is not None
            else _prompt_optional_int("tensor_parallel_size", 1)
        )
        max_len_val = (
            max_model_len
            if max_model_len is not None
            else _prompt_optional_int("max_model_len")
        )
        gpu_util_val = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else _prompt_optional_float("gpu_memory_utilization", 0.85)
        )
        dtype_val = dtype or _prompt_optional("dtype", "auto")
        quant_val = quantization or _prompt_optional("quantization")
        payload = ProfileCreate(
            name=name_val,
            runtime=runtime_val,
            description=desc_val,
            tensor_parallel_size=tp_val,
            max_model_len=max_len_val,
            gpu_memory_utilization=gpu_util_val,
            dtype=dtype_val,
            quantization=quant_val,
        )

    with _session() as db:
        service = ProfileService(db)
        issues = service.validate(payload)
        _print_validation(issues)
        if any(issue.severity == "error" for issue in issues):
            raise typer.BadParameter("refusing to save profile with validation errors")
        created = service.create_profile(payload)
    console.print(f"[green]Created profile[/green] {created.name} ({created.id})")


@profile_app.command("edit")
def profile_edit(
    profile: Annotated[str, typer.Argument(help="Profile id or name.")],
    description: Annotated[str | None, typer.Option()] = None,
    tensor_parallel: Annotated[int | None, typer.Option()] = None,
    max_model_len: Annotated[int | None, typer.Option()] = None,
    gpu_memory_utilization: Annotated[float | None, typer.Option()] = None,
    dtype: Annotated[str | None, typer.Option()] = None,
    quantization: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Apply a partial update to a profile record."""
    updates_dict: dict[str, Any] = {}
    if description is not None:
        updates_dict["description"] = description
    if tensor_parallel is not None:
        updates_dict["tensor_parallel_size"] = tensor_parallel
    if max_model_len is not None:
        updates_dict["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        updates_dict["gpu_memory_utilization"] = gpu_memory_utilization
    if dtype is not None:
        updates_dict["dtype"] = dtype
    if quantization is not None:
        updates_dict["quantization"] = quantization
    if not updates_dict:
        raise typer.BadParameter("provide at least one --field to update")
    with _session() as db:
        service = ProfileService(db)
        existing = _resolve_profile_or_die(service, profile)
        update = ProfileUpdate(**updates_dict)
        issues = service.validate(update)
        _print_validation(issues)
        if any(issue.severity == "error" for issue in issues):
            raise typer.BadParameter("refusing to save profile with validation errors")
        try:
            updated = service.update_profile(existing.id, update)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    assert updated is not None
    console.print(f"[green]Updated[/green] {updated.name} ({updated.id})")


@profile_app.command("delete")
def profile_delete(
    profile: Annotated[str, typer.Argument(help="Profile id or name.")],
) -> None:
    """Delete a profile."""
    with _session() as db:
        service = ProfileService(db)
        existing = _resolve_profile_or_die(service, profile)
        ok = service.delete_profile(existing.id)
    if not ok:
        raise typer.BadParameter(f"Profile not found: {profile}")
    console.print(f"[green]Deleted profile[/green] {profile}")


@profile_app.command("clone")
def profile_clone(
    profile: Annotated[str, typer.Argument(help="Profile id or name to clone.")],
    new_name: Annotated[str, typer.Argument(help="Name for the clone.")],
) -> None:
    """Duplicate a profile under a new name."""
    with _session() as db:
        service = ProfileService(db)
        existing = _resolve_profile_or_die(service, profile)
        try:
            cloned = service.clone_profile(existing.id, new_name)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    assert cloned is not None
    console.print(f"[green]Cloned profile[/green] -> {cloned.name} ({cloned.id})")


@profile_app.command("export")
def profile_export(
    profile: Annotated[str, typer.Argument(help="Profile id or name.")],
    out_path: Annotated[Path, typer.Argument(help="Output YAML file.")],
) -> None:
    """Write a single profile to a YAML file."""
    with _session() as db:
        service = ProfileService(db)
        existing = _resolve_profile_or_die(service, profile)
        data = service.export_to_dict(existing)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    console.print(f"[green]Wrote profile[/green] {existing.name} -> {out_path}")


@profile_app.command("sync")
def profile_sync() -> None:
    """Re-seed profiles from ``configs/profiles.yaml``.

    Upserts the seven shipped defaults (fast/coding/reasoning/long-context/
    quant/adtech/tutoring) into the database. Existing profiles with the
    same name are updated in place; profiles you created locally and that
    aren't in the YAML are untouched.
    """
    with _session() as db:
        synced = ProfileService(db).sync_from_yaml()
    console.print(f"[green]Synced[/green] {len(synced)} profiles from YAML.")


@profile_app.command("import")
def profile_import(
    in_path: Annotated[Path, typer.Argument(help="Input YAML file.")],
) -> None:
    """Create or update a profile from a YAML file."""
    if not in_path.exists():
        raise typer.BadParameter(f"File not found: {in_path}")
    data = yaml.safe_load(in_path.read_text())
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Expected a YAML mapping in {in_path}")
    with _session() as db:
        service = ProfileService(db)
        try:
            imported = service.import_from_dict(data)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Imported profile[/green] {imported.name} ({imported.id})")


# ---------------------------------------------------------------------------
# top-level commands
# ---------------------------------------------------------------------------


def profiles_cmd() -> None:
    """List profiles."""
    with _session() as db:
        profiles = ProfileService(db).list_profiles()
    table = Table(title="Profiles")
    table.add_column("ID")
    table.add_column("Name", style="cyan")
    table.add_column("Backend")
    table.add_column("TP", justify="right")
    table.add_column("max_model_len", justify="right")
    table.add_column("Description")
    for profile in profiles:
        table.add_row(
            (profile.id or "")[:8],
            profile.name,
            profile.runtime.value,
            str(profile.tensor_parallel_size or ""),
            str(profile.max_model_len or ""),
            profile.description or "",
        )
    console.print(table)


def preview_cmd(
    model_id: Annotated[str, typer.Argument(help="Model id or unique name.")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Profile name or id (defaults to model's default)."),
    ] = None,
    gpus: Annotated[str, typer.Option("--gpus", help="GPU mode or comma-separated ids.")] = "auto",
) -> None:
    """Dry-run a launch plan for ``MODEL_ID`` + ``--profile``.

    Prints backend, command, environment, selected GPUs, expected VRAM,
    context length, port, and health endpoint without starting anything.
    """
    settings = load_settings()
    init_db(settings.database_url)
    with Session(get_engine(settings.database_url)) as db:
        reg = RegistryService(db)
        prof_service = ProfileService(db)
        resolved_id = _resolve_model_or_die(reg, model_id)
        model = reg.get_model(resolved_id)
        assert model is not None
        profile_name_or_id = profile or model.default_profile_id
        profile_id: str | None = None
        if profile_name_or_id:
            resolved_profile = prof_service.find(profile_name_or_id)
            if resolved_profile is None:
                raise typer.BadParameter(
                    f"Profile not found: {profile_name_or_id}"
                )
            profile_id = resolved_profile.id
        request = SessionStartRequest(
            model_id=resolved_id,
            profile_id=profile_id,
            runtime=model.runtime,
            gpu_ids=[int(p) for p in gpus.split(",") if p.strip().isdigit()],
            gpu_mode=_resolve_gpu_mode(gpus),
            dry_run=True,
        )
        plan = SessionService(db).plan(request)
    table = Table(title=f"Launch preview: {model.name}", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("backend", plan.runtime.value)
    table.add_row("port", str(plan.port or ""))
    table.add_row("endpoint", plan.endpoint_url or "")
    table.add_row("health", plan.health_url or "")
    table.add_row("gpu_ids", ",".join(str(g) for g in plan.gpu_ids) or "(auto)")
    table.add_row("tensor_parallel", str(plan.tensor_parallel_size))
    table.add_row(
        "estimated VRAM (GB)",
        f"{plan.estimated_vram_gb:.1f}" if plan.estimated_vram_gb else "",
    )
    table.add_row("command", plan.command_preview)
    if plan.env:
        table.add_row("env", ", ".join(f"{k}={v}" for k, v in plan.env.items()))
    console.print(table)
    for note in plan.notes:
        console.print(f"[blue]note:[/blue] {note}")
    for warning in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for reason in plan.refusal_reasons:
        console.print(f"[red]refusal:[/red] {reason}")


def export_registry_cmd(
    out_path: Annotated[Path, typer.Argument(help="Output JSON file.")],
) -> None:
    """Write models + profiles + settings to a JSON bundle."""
    with _session() as db:
        models = RegistryService(db).list_models(include_inactive=True)
        profiles = ProfileService(db).list_profiles()
    bundle = RegistryExport(models=models, profiles=profiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundle.model_dump_json(indent=2))
    console.print(
        f"[green]Exported registry[/green] -> {out_path} "
        f"({len(models)} models, {len(profiles)} profiles)"
    )


def import_registry_cmd(
    in_path: Annotated[Path, typer.Argument(help="Input JSON file.")],
    replace_profiles: Annotated[
        bool,
        typer.Option(
            "--replace-profiles",
            help="Update profiles by name (default: skip existing).",
        ),
    ] = False,
) -> None:
    """Read a registry bundle and merge it into the local database."""
    if not in_path.exists():
        raise typer.BadParameter(f"File not found: {in_path}")
    data = json.loads(in_path.read_text())
    bundle = RegistryExport.model_validate(data)
    added_models = 0
    added_profiles = 0
    with _session() as db:
        reg = RegistryService(db)
        prof = ProfileService(db)
        existing_models = {
            (m.runtime.value, m.source or m.name): m for m in reg.list_models(include_inactive=True)
        }
        for model in bundle.models:
            key = (model.runtime.value, model.source or model.name)
            if key in existing_models:
                continue
            reg.add_model(
                ModelCreate(
                    name=model.name,
                    runtime=model.runtime,
                    source=model.source,
                    path=model.path,
                    format=model.format,
                    quantization=model.quantization,
                    estimated_vram_gb=model.estimated_vram_gb,
                    max_context=model.max_context,
                    parameter_count=model.parameter_count,
                    notes=model.notes,
                    default_profile_id=model.default_profile_id,
                    tags=list(model.tags),
                    metadata=dict(model.metadata),
                )
            )
            added_models += 1
        for profile in bundle.profiles:
            existing = prof.get_by_name(profile.name)
            if existing is not None and not replace_profiles:
                continue
            prof.import_from_dict(prof.export_to_dict(profile))
            added_profiles += 1
    console.print(
        f"[green]Imported[/green] {added_models} models, "
        f"{added_profiles} profiles "
        f"({'updates allowed' if replace_profiles else 'skipped existing'})"
    )


def register(app: typer.Typer) -> None:
    """Attach the registry/profile sub-typers and top-level commands to ``app``.

    Called from ``llmctl.cli`` so the main module doesn't grow another
    several hundred lines of inline command definitions.
    """
    app.add_typer(model_app, name="model")
    app.add_typer(profile_app, name="profile")
    app.command("profiles", help="List launch profiles.")(profiles_cmd)
    app.command("preview", help="Dry-run a launch plan for MODEL_ID.")(preview_cmd)
    app.command("export-registry", help="Write registry bundle to JSON.")(
        export_registry_cmd
    )
    app.command("import-registry", help="Read registry bundle from JSON.")(
        import_registry_cmd
    )
