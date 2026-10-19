"""`llmctl hf`: find a Hugging Face model and size it for these GPUs before a download.

Read-only. Nothing here downloads weights or starts a server; `llmctl add`
downloads and registers, `llmctl run` serves.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from llmctl.cli_tables import whole_id_table
from llmctl.hf.hardware import HardwareInfo, detect_hardware
from llmctl.hf.inspect import inspect_model
from llmctl.hf.parsing import parse_query
from llmctl.hf.plan import LaunchPlan, build_plan, render_command
from llmctl.hf.search import best_match, search_models

console = Console()

hf_app = typer.Typer(
    name="hf",
    help=(
        "Find a Hugging Face model and check that it fits these GPUs, before you "
        "download it (read-only; `llmctl add` downloads)."
    ),
    no_args_is_help=True,
    suggest_commands=False,
)

_INSPECT_KEYS = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "max_position_embeddings",
    "quantization_config",
    "torch_dtype",
    "vocab_size",
)


def register(app: typer.Typer) -> None:
    app.add_typer(hf_app, name="hf")


@hf_app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help='What you want, e.g. "qwen coder 32b awq".')],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max candidates.")] = 10,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print only a JSON list of candidates.")
    ] = False,
) -> None:
    """Search Hugging Face and rank the matches.

    The query is parsed for family, size (e.g. 70b, "under 35B") and quantization
    (awq/gptq/fp8/gguf/bnb); the "Why" column shows what matched.
    """
    parsed = parse_query(query)
    if not as_json:
        console.print(
            f"[dim]parsed:[/] family={parsed.family} size={parsed.param_count_b} "
            f"quant={parsed.quantization} keywords={escape(str(parsed.keywords))}"
        )
    try:
        results = search_models(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - network / HF API errors
        if as_json:
            typer.echo(f"Search failed: {exc}", err=True)
        else:
            console.print(f"[red]Search failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "model_id": c.model_id,
                        "score": c.score,
                        "downloads": c.downloads,
                        "likes": c.likes,
                        "task": c.pipeline_tag,
                        "reasons": list(c.reasons),
                    }
                    for c in results
                ]
            )
        )
        if not results:
            raise typer.Exit(1)
        return
    if not results:
        console.print("[yellow]No matches.[/]")
        raise typer.Exit(1)

    rows = [
        (
            f"{c.score:.1f}",
            c.model_id,
            f"{c.downloads:,}",
            f"{c.likes:,}",
            c.pipeline_tag or "-",
            ", ".join(c.reasons[:3]) or "-",
        )
        for c in results
    ]
    console.print(
        whole_id_table(
            console,
            ("Score", "Model", "Downloads", "Likes", "Task", "Why"),
            rows,
            ids={"Model", "Score", "Downloads", "Likes"},
            justify={"Score": "right", "Downloads": "right", "Likes": "right"},
            styles={},
        )
    )


@hf_app.command("inspect")
def inspect_cmd(
    model_id: Annotated[str, typer.Argument(help="Exact Hugging Face id (org/name).")],
) -> None:
    """Show a model's config.json and tokenizer facts; never downloads weights."""
    meta = inspect_model(model_id)
    if meta.error:
        console.print(f"[red]{escape(meta.error)}[/]")
    console.print(f"[bold]{escape(meta.model_id)}[/]  gated={meta.gated} private={meta.private}")
    if meta.pipeline_tag:
        console.print(f"task: {escape(meta.pipeline_tag)}")
    if meta.tags:
        console.print(f"tags: {escape(', '.join(meta.tags[:12]))}")
    for key in _INSPECT_KEYS:
        if key in meta.config:
            console.print(f"  {key}: {escape(str(meta.config[key]))}")
    if meta.tokenizer_config:
        present = "present" if meta.tokenizer_config.get("chat_template") else "absent"
        console.print(f"tokenizer.chat_template: {present}")


def _resolve_model_id(query_or_id: str) -> str:
    """Use an org/name verbatim; otherwise search and take the top match."""
    parsed = parse_query(query_or_id)
    if parsed.is_model_id():
        return parsed.raw
    top = best_match(query_or_id)
    console.print(
        f"[dim]selected:[/] {escape(top.model_id)} (score {top.score:.1f}: "
        f"{escape(', '.join(top.reasons[:2]))})"
    )
    return top.model_id


def _print_plan(p: LaunchPlan, hw: HardwareInfo) -> None:
    console.print(f"\n[bold]Plan for {escape(p.model_id)}[/]")
    console.print(f"  family={p.family} params={p.param_count_b} quant={p.quantization}")
    console.print(f"  archs={escape(str(p.architectures))} vllm_supported={p.vllm_supported}")
    console.print(
        f"  tp={p.tensor_parallel_size} max_model_len={p.max_model_len} "
        f"util={p.gpu_memory_utilization:.2f} ships_own_code={p.trust_remote_code}"
    )
    if hw.gpus:
        gpus = ", ".join(f"{g.index}:{g.name} ({g.vram_gib:.1f} GiB)" for g in hw.gpus)
        console.print(f"  gpus: {escape(gpus)}")
    else:
        console.print("  [yellow]gpus: none detected[/]")
    for note in p.notes:
        console.print(f"  • {escape(note)}")
    for warning in p.warnings:
        console.print(f"  [yellow]! {escape(warning)}[/]")
    console.print("\n[bold]vLLM command:[/]")
    console.print("  " + escape(" ".join(render_command(p))))
    console.print(f"\n[bold]To download and register it:[/] llmctl add {escape(p.model_id)}")


@hf_app.command("plan")
def plan_cmd(
    query: Annotated[str, typer.Argument(help="Hugging Face id or a description.")],
    tp: Annotated[int | None, typer.Option("--tp", help="Override tensor_parallel_size.")] = None,
    max_model_len: Annotated[int | None, typer.Option("--max-model-len")] = None,
    gpu_memory_utilization: Annotated[
        float | None, typer.Option("--gpu-memory-utilization")
    ] = None,
) -> None:
    """Size a model for these GPUs and print the `vllm serve` command.

    Reads config.json and the GPU inventory to pick tensor_parallel_size,
    max_model_len, gpu_memory_utilization and dtype, and warns about OOM risk, GGUF,
    gated repos and repos that ship their own code (the command never adds
    --trust-remote-code). Nothing is downloaded or started.
    """
    hw = detect_hardware()
    model_id = _resolve_model_id(query)
    meta = inspect_model(model_id)
    if meta.error and not meta.config:
        console.print(f"[red]{escape(meta.error)}[/]")
        raise typer.Exit(1)
    p = build_plan(meta, hw)
    if tp is not None:
        p.tensor_parallel_size = tp
    if max_model_len is not None:
        p.max_model_len = max_model_len
    if gpu_memory_utilization is not None:
        p.gpu_memory_utilization = gpu_memory_utilization
    _print_plan(p, hw)


@hf_app.command("fit")
def fit_cmd(
    query: Annotated[str, typer.Argument(help='What you want, e.g. "coder under 35B".')],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Candidates to check.")] = 5,
) -> None:
    """Rank candidates and show whether each one fits these GPUs.

    Like `search`, but reads each candidate's config.json and sizes it, so the
    "Fits?" column reflects OOM risk (yes / risk), or says unknown (no GPUs or no
    model size) or error (the repo could not be read). One metadata fetch per row.
    For measured per-role advice on models you already have, see `llmctl recommend`.
    """
    hw = detect_hardware()
    if hw.gpus:
        console.print(f"[dim]hardware:[/] {hw.count} GPU(s), {hw.total_vram_gib:.1f} GiB total")
    try:
        candidates = search_models(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - network / HF API errors
        console.print(f"[red]Search failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    if not candidates:
        console.print(
            "[yellow]No matches.[/] Try a broader query, or `llmctl hf search` for the raw pool."
        )
        raise typer.Exit(1)

    rows = []
    for c in candidates:
        meta = inspect_model(c.model_id)
        p = build_plan(meta, hw)
        oom = any("OOM" in w for w in p.warnings)
        if meta.error:
            fits, note = "error", meta.error
        elif not hw.gpus or p.param_count_b is None:
            # No GPUs or no size: the planner has nothing to judge "yes" by.
            fits = "unknown"
            note = "; ".join(p.warnings[:1]) or "model size unknown"
        else:
            fits = "yes" if p.vllm_supported and not oom else "risk"
            note = "; ".join(p.warnings[:1]) or (p.notes[0] if p.notes else "")
        rows.append((c.model_id, f"{c.score:.1f}", fits, str(p.tensor_parallel_size), note))
    console.print(
        whole_id_table(
            console,
            ("Model", "Score", "Fits?", "tp", "Notes"),
            rows,
            ids={"Model", "Score", "Fits?", "tp"},
            justify={"Score": "right", "Fits?": "center"},
        )
    )
