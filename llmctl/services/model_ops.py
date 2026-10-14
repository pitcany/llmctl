"""Engine-agnostic add and remove for local models.

One verb pair covers every engine, so the operator never has to remember
which tool owns which model:

* **add SOURCE** fetches the model with the engine's own mechanism
  (``ollama pull``, a Hugging Face download, ``lms get``) and registers it.
  A local path is registered in place.
* **remove [purge]** hides the registry row and, with ``purge``, also removes
  the model from the engine (``ollama rm``, the Hugging Face cache entry, the
  LM Studio model folder).

SOURCE forms, first match wins::

    ollama:TAG          Ollama tag, incl. namespaced ``user/model:tag``
    lmstudio:KEY        LM Studio catalog key (``lms:`` also works)
    hf:ORG/REPO[:QUANT] Hugging Face repo (the ``hf:`` is optional)
    /path, ~/path, ./p  local directory or ``.gguf`` file
    hf.co/ORG/REPO:Q    Ollama's own Hugging Face syntax -> Ollama
    ORG/REPO[:QUANT]    Hugging Face repo; GGUF repos -> llama.cpp, else vLLM
    TAG                 anything else is an Ollama tag (``qwen3:8b``)

:func:`plan_add` and :func:`plan_remove` are pure descriptions (no network,
no writes) so the TUI can preview exactly what will happen before the
operator confirms; :class:`ModelOps` executes them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from llmctl._asyncutil import run_sync
from llmctl.adapters.lmstudio import lms_binary
from llmctl.db import ModelRecord, ModelStatus, RuntimeName, SessionRecord, SessionStatus
from llmctl.schemas import HealthState, Model
from llmctl.services.registry import (
    RegistryService,
    effective_host,
    local_host,
    record_to_model,
)
from llmctl.services.router import RuntimeRouter

#: Engines ``add`` can fetch or register for. The rest are lanes or adopted
#: endpoints whose models are declared elsewhere.
ADDABLE_ENGINES: tuple[RuntimeName, ...] = (
    RuntimeName.OLLAMA,
    RuntimeName.VLLM,
    RuntimeName.SGLANG,
    RuntimeName.FREETOKEN,
    RuntimeName.STRATA,
    RuntimeName.LLAMA_CPP,
    RuntimeName.LMSTUDIO,
)

_NOT_ADDABLE = {
    RuntimeName.OPENAI: "OpenAI-compatible endpoints are adopted, not added (`llmctl adopt`).",
    RuntimeName.PYTHON_SCRIPT: "python_script models are registered with `llmctl model add` "
    "without a SOURCE (flags or prompts).",
}

#: Files a vLLM/SGLang download skips: other formats of the same weights.
_HF_SAFETENSORS_IGNORE = [
    "*.gguf",
    "*.onnx",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "original/*",
    "*.pth",
]

#: Free space kept on the cache filesystem after a download.
DISK_MARGIN_BYTES = 10 * 1024**3

_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)
_QUANT_SPLIT_RE = re.compile(r"^(?P<repo>[^:@\s]+?)(?:[:@](?P<quant>[^:@/\s]+))?$")

Progress = Callable[[str], None]


class ModelOpsError(ValueError):
    """A SOURCE that cannot be planned, or an add/remove that failed."""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddPlan:
    """What ``add`` will do, in words the operator can check before it runs."""

    engine: RuntimeName
    #: ``ollama-pull`` | ``hf-download`` | ``lmstudio-get`` | ``register``
    method: str
    #: Normalized source: Ollama tag, HF repo id, LM Studio key, or local path.
    source: str
    name: str
    summary: str
    #: The equivalent shell command, so the operator learns the engine's tool.
    command: str
    quant: str | None = None
    #: HF only: one file inside the repo (``org/repo/file.gguf``).
    filename: str | None = None


def _coerce_engine(engine: RuntimeName | str | None) -> RuntimeName | None:
    if engine is None or engine == "" or engine == "auto":
        return None
    if isinstance(engine, RuntimeName):
        return engine
    value = str(engine).strip().lower().replace("-", "_").replace(".", "_")
    aliases = {"llamacpp": "llama_cpp", "llama": "llama_cpp", "lms": "lmstudio"}
    value = aliases.get(value, value)
    try:
        return RuntimeName(value)
    except ValueError as exc:
        choices = ", ".join(e.value for e in ADDABLE_ENGINES)
        raise ModelOpsError(f"Unknown engine {engine!r}; use one of: {choices}.") from exc


def normalize_ollama_tag(tag: str) -> str:
    """Return the name Ollama lists after a pull (``qwen3`` -> ``qwen3:latest``)."""
    tag = tag.strip()
    last = tag.rsplit("/", 1)[-1]
    return tag if ":" in last else f"{tag}:latest"


def _looks_local(source: str) -> bool:
    return source.startswith(("/", "~", "./", "../")) or source in (".", "..")


def _split_hf(source: str) -> tuple[str, str | None, str | None]:
    """``org/repo[:QUANT|@QUANT|/file.gguf]`` -> (repo, quant, filename)."""
    text = source.strip()
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            # Browser URLs: org/repo/blob/main/file.gguf or org/repo/tree/main
            text = re.sub(r"^([^/]+/[^/]+)/(?:blob|resolve|tree)/[^/]+", r"\1", text)
    text = text.strip("/")
    parts = text.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ModelOpsError(
            f"{source!r} is not a Hugging Face repo id; expected ORG/REPO "
            "(e.g. Qwen/Qwen3-8B or unsloth/Qwen3-8B-GGUF:Q4_K_M)."
        )
    if len(parts) > 2:
        repo = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        if not filename.lower().endswith(".gguf"):
            raise ModelOpsError(
                f"{source!r}: only a .gguf file can be picked out of a repo; "
                "give ORG/REPO to download the whole checkpoint."
            )
        return repo, None, filename
    match = _QUANT_SPLIT_RE.match(text)
    if not match:
        raise ModelOpsError(f"{source!r} is not a Hugging Face repo id (ORG/REPO[:QUANT]).")
    return match.group("repo"), match.group("quant"), None


def _local_gguf_entry(path: Path) -> Path:
    """The file llama.cpp should be pointed at for ``path`` (file or folder)."""
    if path.is_file():
        return path
    ggufs = sorted(p for p in path.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    groups = sorted({_SHARD_RE.sub("", p.name) for p in ggufs})
    if len(groups) > 1:
        listed = ", ".join(groups[:8])
        raise ModelOpsError(
            f"{path} holds several GGUF models ({listed}); give the file itself."
        )
    return ggufs[0]


def _plan_local(source: str, engine: RuntimeName | None, name: str | None) -> AddPlan:
    path = Path(source).expanduser()
    if not path.exists():
        raise ModelOpsError(f"No such file or folder: {path}")
    path = path.resolve()
    is_gguf = path.suffix.lower() == ".gguf" or (
        path.is_dir() and any(path.glob("*.gguf"))
    )
    is_strata = path.is_file() and path.suffix.lower() == ".json" and _is_strata_config(path)
    if engine is None:
        if is_gguf:
            engine = RuntimeName.LLAMA_CPP
        elif path.is_dir() and (path / "config.json").exists():
            engine = RuntimeName.VLLM
        elif is_strata:
            engine = RuntimeName.STRATA
        else:
            raise ModelOpsError(
                f"Can't tell which engine serves {path}: it is neither a .gguf "
                "file, a folder with config.json, nor a Strata config. Pick an engine."
            )
    if engine == RuntimeName.STRATA and not is_strata:
        raise ModelOpsError(
            f"strata serves the config JSON its setup writes (exe + args + tokenizer); "
            f"{path} is not one."
        )
    if engine in (RuntimeName.OLLAMA, RuntimeName.LMSTUDIO):
        tool = "ollama create" if engine == RuntimeName.OLLAMA else "lms import"
        raise ModelOpsError(
            f"{engine.value} keeps its own library; import a local file with "
            f"`{tool}` (lms import MOVES the file), then scan."
        )
    if engine == RuntimeName.LLAMA_CPP:
        if not is_gguf:
            raise ModelOpsError(f"llama.cpp serves .gguf files; {path} has none.")
        path = _local_gguf_entry(path)
    elif is_gguf:
        raise ModelOpsError(
            f"{engine.value} serves safetensors checkpoints, not GGUF; "
            "use the llama_cpp engine for this file."
        )
    display = name or (_SHARD_RE.sub("", path.stem) if path.is_file() else path.name)
    if engine == RuntimeName.STRATA and not name:
        display = _strata_model_name(path) or display
    return AddPlan(
        engine=engine,
        method="register",
        source=str(path),
        name=display,
        summary=f"Register {path} for {engine.value} (nothing is downloaded)"
        + (", and write a starter preset for it." if engine == RuntimeName.VLLM else "."),
        command=f"llmctl model add {path} --engine {engine.value}",
    )


def _is_strata_config(path: Path) -> bool:
    """A Strata server config: a JSON object with ``exe`` and ``args``."""
    import json

    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "exe" in data and isinstance(data.get("args"), list)


def _strata_model_name(path: Path) -> str | None:
    import json

    try:
        name = json.loads(path.read_text()).get("model_name")
    except (OSError, ValueError, AttributeError):
        return None
    return str(name) if name else None


def plan_add(
    source: str,
    engine: RuntimeName | str | None = None,
    *,
    name: str | None = None,
) -> AddPlan:
    """Describe what ``add SOURCE`` will do. Pure: no network, no writes."""
    text = (source or "").strip()
    if not text:
        raise ModelOpsError(
            "Give a source: an Ollama tag (qwen3:8b), a Hugging Face repo "
            "(Qwen/Qwen3-8B), or a local path."
        )
    chosen = _coerce_engine(engine)
    if chosen in _NOT_ADDABLE:
        raise ModelOpsError(_NOT_ADDABLE[chosen])

    lowered = text.lower()
    if lowered.startswith("ollama:"):
        if chosen not in (None, RuntimeName.OLLAMA):
            raise ModelOpsError(f"'ollama:' source conflicts with engine {chosen.value}.")
        text, chosen = text[len("ollama:") :], RuntimeName.OLLAMA
    elif lowered.startswith(("lmstudio:", "lms:")):
        if chosen not in (None, RuntimeName.LMSTUDIO):
            raise ModelOpsError(f"'lmstudio:' source conflicts with engine {chosen.value}.")
        text, chosen = text.split(":", 1)[1], RuntimeName.LMSTUDIO
    elif lowered.startswith("hf:"):
        text = text[3:]
        if chosen == RuntimeName.OLLAMA:
            text = f"hf.co/{text}"
        elif _looks_local(text):
            raise ModelOpsError("'hf:' takes a repo id, not a local path.")
    elif _looks_local(text):
        return _plan_local(text, chosen, name)
    elif lowered.startswith("hf.co/") and chosen in (None, RuntimeName.OLLAMA):
        chosen = RuntimeName.OLLAMA
    elif "/" not in text and chosen is None:
        chosen = RuntimeName.OLLAMA

    text = text.strip()
    if not text:
        raise ModelOpsError("The source after the prefix is empty.")

    if chosen == RuntimeName.OLLAMA:
        tag = normalize_ollama_tag(text)
        return AddPlan(
            engine=RuntimeName.OLLAMA,
            method="ollama-pull",
            source=tag,
            name=name or tag,
            summary=f"Pull {tag} into the Ollama library.",
            command=f"ollama pull {tag}",
        )

    if chosen == RuntimeName.LMSTUDIO:
        key = text[len("hf.co/") :] if text.lower().startswith("hf.co/") else text
        return AddPlan(
            engine=RuntimeName.LMSTUDIO,
            method="lmstudio-get",
            source=key,
            name=name or key,
            summary=f"Download {key} with LM Studio (its default variant for this machine).",
            command=f"lms get {key} --yes",
        )

    if text.lower().startswith("hf.co/"):
        text = text[len("hf.co/") :]
    repo, quant, filename = _split_hf(text)
    wants_gguf = bool(quant or filename) or "gguf" in repo.lower()
    if chosen is None:
        chosen = RuntimeName.LLAMA_CPP if wants_gguf else RuntimeName.VLLM
    if chosen in (RuntimeName.VLLM, RuntimeName.SGLANG, RuntimeName.FREETOKEN) and (
        quant or filename
    ):
        raise ModelOpsError(
            f"{chosen.value} downloads the whole safetensors checkpoint; a quant or "
            ".gguf file selects a GGUF build, which is llama_cpp's job."
        )
    if chosen == RuntimeName.LLAMA_CPP:
        pick = filename or (f"the {quant} files" if quant else "its GGUF file")
        display = name or (f"{repo}:{quant}" if quant else (filename or repo))
        return AddPlan(
            engine=RuntimeName.LLAMA_CPP,
            method="hf-download",
            source=repo,
            quant=quant,
            filename=filename,
            name=display,
            summary=f"Download {pick} from Hugging Face {repo} for llama.cpp.",
            command=(
                f"hf download {repo} {filename}"
                if filename
                else f"hf download {repo} --include '*{quant}*.gguf'"
                if quant
                else f"hf download {repo} --include '*.gguf'"
            ),
        )
    return AddPlan(
        engine=chosen,
        method="hf-download",
        source=repo,
        name=name or repo,
        summary=(
            f"Download the safetensors checkpoint {repo} from Hugging Face, then write "
            "a starter preset so `llmctl run` can serve it."
            if chosen == RuntimeName.VLLM
            else f"Download the safetensors checkpoint {repo} from Hugging Face; "
            f"`llmctl run` then serves it with {chosen.value}."
        ),
        command=f"hf download {repo}",
    )


# ---------------------------------------------------------------------------
# Removal planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovePlan:
    """What ``remove`` does to one model, with and without ``purge``."""

    model: Model
    can_purge: bool
    #: What purge removes, or why it is not possible (shown either way).
    purge_summary: str
    #: The checkbox / flag label, e.g. "Also remove it from Ollama".
    purge_label: str
    command: str | None = None
    targets: tuple[Path, ...] = ()
    size_bytes: int | None = None
    #: Reasons purge is refused right now (a running session, ...).
    blockers: tuple[str, ...] = ()
    #: Starter presets ``add`` wrote for this model; purge deletes them too.
    presets_to_delete: tuple[str, ...] = ()


def human_size(size: int | None) -> str:
    """``31.2 GB`` style, decimal like ``ollama list`` and ``hf``."""
    if size is None:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000  # type: ignore[assignment]
    return f"{size} B"  # pragma: no cover


def _tree_sizes(path: Path) -> tuple[int, int]:
    """``(freed, shared)`` bytes under ``path``; symlinks are not followed.

    A file with other hard links (the model store links GGUFs into Ollama and
    LM Studio) frees nothing when one name goes, so it counts as shared.
    """
    files: list[Path]
    if path.is_symlink() or path.is_file():
        files = [] if path.is_symlink() else [path]
    else:
        files = [
            Path(root) / name
            for root, _dirs, names in os.walk(path, followlinks=False)
            for name in names
        ]
    freed = shared = 0
    for fpath in files:
        try:
            if fpath.is_symlink():
                continue
            st = fpath.stat()
        except OSError:
            continue
        if st.st_nlink > 1:
            shared += st.st_size
        else:
            freed += st.st_size
    return freed, shared


def _tree_size(path: Path) -> int:
    """Total bytes under ``path`` (freed + shared)."""
    return sum(_tree_sizes(path))


def _model_roots() -> list[Path]:
    from llmctl.config import load_model_dirs

    roots = []
    for root in load_model_dirs().model_roots:
        resolved = root.resolve_path()
        if resolved is not None:
            roots.append(resolved.expanduser().resolve())
    return roots


def _hf_hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE).expanduser()


def _inside(path: Path, roots: list[Path]) -> bool:
    probe = path.parent.resolve() / path.name if path.is_symlink() else path.resolve()
    return any(probe == r or probe.is_relative_to(r) for r in roots)


def _hf_repo_dir(path: Path, hub: Path) -> Path | None:
    """The ``models--org--repo`` folder holding ``path``, if it is in the cache."""
    try:
        rel = path.parent.resolve().relative_to(hub.resolve()) if path.is_symlink() else (
            path.resolve().relative_to(hub.resolve())
        )
    except ValueError:
        return None
    if not rel.parts or not rel.parts[0].startswith("models--"):
        return None
    return hub / rel.parts[0]


def _shard_set(entry: Path) -> list[Path]:
    """Every shard of a split GGUF (``-00001-of-00003``), or just ``entry``."""
    if not _SHARD_RE.search(entry.name):
        return [entry]
    stem = _SHARD_RE.sub("", entry.name)
    return sorted(p for p in entry.parent.glob("*.gguf") if _SHARD_RE.sub("", p.name) == stem)


def _lmstudio_path(
    model: Model, roots: list[Path], runner: Callable[..., Any]
) -> tuple[Path | None, str]:
    """Folder of a local LM Studio model from ``lms ls --json``, or why not.

    Entries from LM Link devices (``deviceIdentifier`` set) are skipped: they
    are files on another machine. ``ollama-library/...`` entries are LM
    Studio's view of the Ollama library; the files belong to Ollama.
    """
    fallback = (
        "LM Studio did not report a local folder for it (`lms ls`); delete it "
        "in the LM Studio app. Only the row is hidden."
    )
    try:
        proc = runner(
            [lms_binary() or "lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        entries = json.loads(proc.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, fallback
    keys = {k.lower() for k in (model.source, model.name) if k}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("deviceIdentifier"):
            continue
        ids = {
            str(entry.get(k, "")).lower()
            for k in ("modelKey", "indexedModelIdentifier", "path")
        }
        if not keys & ids:
            continue
        rel = str(entry.get("path") or "")
        if rel.startswith("ollama-library/"):
            return None, (
                "LM Studio lists this from the Ollama library; the files are "
                "Ollama's. Remove the Ollama row with purge instead. Only this "
                "row is hidden."
            )
        for root in roots:
            candidate = root / rel
            if candidate.exists() or candidate.is_symlink():
                # "publisher/model/file.gguf" -> remove the model folder.
                nested = candidate.is_file() and len(Path(rel).parts) > 2
                return (candidate.parent if nested else candidate), ""
    return None, fallback


def _active_sessions(db: Session, model_id: str) -> list[str]:
    live = (SessionStatus.RUNNING, SessionStatus.STARTING)
    rows = db.exec(
        select(SessionRecord).where(
            SessionRecord.model_id == model_id,
            SessionRecord.status.in_(live),  # type: ignore[attr-defined]
        )
    ).all()
    return [(r.id or "")[:8] for r in rows]


def _preset_users(
    model: Model,
    targets: tuple[Path, ...],
    db: Session | None,
    config_dir: Path | None = None,
) -> list[str]:
    """Aliases of vLLM presets that would break if ``model`` were purged.

    A preset is what ``llmctl run`` and llm-router start, so purging weights
    a preset still names takes that lane down on its next start (and the
    deployed tutor generator is one of them). Matches the loader's own link,
    the preset's ``model_id`` against the row, and the HF cache folder the
    purge would delete.
    """
    try:
        from llmctl.presets import load_all_records
        from llmctl.services.preset_loader import resolve_preset_link

        records = load_all_records(**({"config_dir": config_dir} if config_dir else {}))
    except Exception:
        return []
    models = RegistryService(db).list_models() if db is not None else [model]
    keys = {k for k in (model.source, model.name, model.path) if k}
    doomed = {t.name for t in targets}
    users = []
    for alias, record in sorted(records.items()):
        model_id = record.model.model_id
        linked, _state = resolve_preset_link(record.model, models)
        if (
            (linked is not None and linked.id == model.id)
            or model_id in keys
            or f"models--{model_id.replace('/', '--')}" in doomed
        ):
            users.append(alias)
    return users


def plan_remove(
    model: Model,
    *,
    db: Session | None = None,
    roots: list[Path] | None = None,
    hub_cache: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
    preset_dir: Path | None = None,
) -> RemovePlan:
    """Describe what removing ``model`` does. Reads the disk, writes nothing."""
    engine = model.runtime
    blockers: tuple[str, ...] = ()
    if db is not None and model.id:
        running = _active_sessions(db, model.id)
        if running:
            blockers = (
                f"session {', '.join(running)} is using it; stop it first (Sessions screen, x).",
            )

    host = effective_host(model.host)
    if host != local_host():
        return RemovePlan(
            model=model,
            can_purge=False,
            purge_summary=f"It lives on {host}, not this machine; remove it there. "
            "Only the row is hidden here.",
            purge_label="",
        )

    if engine == RuntimeName.OLLAMA:
        tag = model.source or model.name
        return RemovePlan(
            model=model,
            can_purge=True,
            purge_summary=f"Removes {tag} from the Ollama library.",
            purge_label=f"Also remove it from Ollama (ollama rm {tag})",
            command=f"ollama rm {tag}",
            size_bytes=model.size_bytes,
            blockers=blockers,
        )

    if engine in (RuntimeName.FREETOKEN, RuntimeName.OPENAI):
        return RemovePlan(
            model=model,
            can_purge=False,
            purge_summary=f"{engine.value} models live outside llmctl; only the row is hidden.",
            purge_label="",
        )

    roots = _model_roots() if roots is None else [Path(r).resolve() for r in roots]
    hub = _hf_hub_cache() if hub_cache is None else hub_cache

    if engine == RuntimeName.LMSTUDIO:
        folder, why = _lmstudio_path(model, roots, runner)
        if folder is None:
            return RemovePlan(
                model=model, can_purge=False, purge_summary=why, purge_label=""
            )
        targets: tuple[Path, ...] = (folder,)
    else:
        raw = (model.path or model.source or "").strip()
        candidate = Path(raw).expanduser() if raw else None
        if candidate is not None and not candidate.is_absolute():
            # vLLM rows from the live probe carry the repo id, not a path.
            repo_dir = hub / f"models--{raw.replace('/', '--')}" if "/" in raw else None
            candidate = repo_dir if repo_dir is not None and repo_dir.exists() else None
        if candidate is None or not (candidate.exists() or candidate.is_symlink()):
            return RemovePlan(
                model=model,
                can_purge=False,
                purge_summary="No local files found for it; only the row is hidden.",
                purge_label="",
            )
        repo_dir = _hf_repo_dir(candidate, hub)
        if candidate.suffix.lower() == ".gguf":
            shards = _shard_set(candidate)
            others = []
            if repo_dir is not None:
                others = [
                    p
                    for p in repo_dir.rglob("*.gguf")
                    if _SHARD_RE.sub("", p.name) != _SHARD_RE.sub("", candidate.name)
                    and "blobs" not in p.parts
                ]
            targets = (repo_dir,) if repo_dir is not None and not others else tuple(shards)
        else:
            targets = (repo_dir,) if repo_dir is not None else (candidate,)

    outside = [t for t in targets if not _inside(t, roots)]
    if outside:
        real = outside[0].resolve()
        where = f" (it resolves to {real})" if real != outside[0] else ""
        return RemovePlan(
            model=model,
            can_purge=False,
            purge_summary=f"{outside[0]}{where} is outside llmctl's model roots "
            "(model_dirs.yaml), so llmctl will not delete it. Only the row is hidden.",
            purge_label="",
        )
    presets = _preset_users(model, targets, db, preset_dir)
    # A preset `add` wrote goes with the weights; a hand-written one blocks.
    starter = (model.metadata or {}).get("starter_preset")
    own = tuple(p for p in presets if p == starter)
    presets = [p for p in presets if p not in own]
    if presets:
        blockers += (
            f"preset {', '.join(presets)} uses it; delete the preset first "
            "(Presets screen, del).",
        )
    also = f" and its starter preset {', '.join(own)}" if own else ""
    shown = str(targets[0]) + (f" (+{len(targets) - 1} more)" if len(targets) > 1 else "")
    links = [t for t in targets if t.is_symlink() and "snapshots" not in t.parts]
    if links:
        # A catalog link (~/AI/models/store is one) can share its target with
        # other links; following it could delete a checkpoint something else
        # serves. Remove the link and say that is all it does.
        return RemovePlan(
            model=model,
            can_purge=True,
            purge_summary=f"Deletes the link {shown}{also} only; the files it points "
            f"to ({links[0].resolve()}) stay.",
            purge_label=f"Also delete the link {shown}{also} (its target stays)",
            command=f"rm {targets[0]}",
            targets=targets,
            size_bytes=0,
            blockers=blockers,
            presets_to_delete=own,
        )
    sizes = [_blob_aware_sizes(t) for t in targets]
    size, shared = sum(f for f, _ in sizes), sum(sh for _, sh in sizes)
    freed_text = f"frees {human_size(size)}"
    if shared:
        freed_text += f"; {human_size(shared)} is hard-linked elsewhere and stays"
    return RemovePlan(
        model=model,
        can_purge=True,
        purge_summary=f"Deletes {shown}{also} ({freed_text}).",
        purge_label=f"Also delete the files{also} ({freed_text}): {shown}",
        presets_to_delete=own,
        command=f"rm -r {targets[0]}" if len(targets) == 1 else None,
        targets=targets,
        size_bytes=size,
        blockers=blockers,
    )


def _blob_aware_sizes(target: Path) -> tuple[int, int]:
    """``(freed, shared)`` for a target; an HF snapshot link counts its blob."""
    if target.is_symlink():
        try:
            return _tree_sizes(target.resolve())
        except OSError:
            return 0, 0
    return _tree_sizes(target)


def _unlink_target(target: Path) -> None:
    """Delete one target. An HF snapshot link takes its blob with it."""
    if target.is_symlink():
        blob = target.resolve()
        target.unlink()
        if blob.exists() and "blobs" in blob.parts:
            blob.unlink()
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class AddResult:
    """Outcome of :meth:`ModelOps.add`: the row, and how to serve it."""

    model: Model
    #: The vLLM preset that serves it (written now, or already there).
    preset: str | None = None
    #: The one command that serves it, when there is one.
    serve: str | None = None
    notes: tuple[str, ...] = ()


@dataclass
class RemoveResult:
    """Outcome of :meth:`ModelOps.remove`, stated plainly."""

    removed_row: bool
    purged: bool
    message: str
    freed_bytes: int | None = None


@dataclass
class ModelOps:
    """Run add/remove plans against the registry and the engines.

    Collaborators are injectable so tests never touch the network, Ollama, or
    the real Hugging Face cache.
    """

    db: Session
    router: RuntimeRouter | None = None
    hf_api: Any = None
    download: Callable[..., str] | None = None
    runner: Callable[..., Any] = subprocess.run
    disk_free: Callable[[Path], int] = field(
        default=lambda p: shutil.disk_usage(p).free
    )
    hub_cache: Path | None = None
    #: Where starter presets go; None = the canonical preset directory.
    preset_dir: Path | None = None

    def _router(self) -> RuntimeRouter:
        if self.router is None:
            self.router = RuntimeRouter()
        return self.router

    # -- add ---------------------------------------------------------------

    def add(self, plan: AddPlan, *, on_progress: Progress | None = None) -> AddResult:
        """Fetch (if needed) and register ``plan``'s model, then make it servable.

        For vLLM that means a starter preset, so ``llmctl run NAME`` can
        serve it the moment ``add`` returns.
        """
        say = on_progress or (lambda _msg: None)
        if plan.method == "ollama-pull":
            model = self._add_ollama(plan, say)
        elif plan.method == "hf-download":
            model = self._add_hf(plan, say)
        elif plan.method == "lmstudio-get":
            model = self._add_lmstudio(plan, say)
        elif plan.method == "register":
            model = self.register(
                Model(
                    name=plan.name,
                    runtime=plan.engine,
                    source=plan.source,
                    path=plan.source,
                    format="gguf" if plan.source.lower().endswith(".gguf") else "safetensors",
                    size_bytes=_tree_size(Path(plan.source)),
                    status=ModelStatus.REGISTERED,
                )
            )
        else:  # pragma: no cover
            raise ModelOpsError(f"Unknown add method {plan.method!r}.")
        return self._make_servable(model)

    def _make_servable(self, model: Model) -> AddResult:
        """Say how to serve ``model``; write the vLLM starter preset it needs."""
        if model.runtime == RuntimeName.VLLM:
            from llmctl.services.preset_scaffold import ensure_starter_preset

            try:
                starter = ensure_starter_preset(
                    self.db, model, config_dir=self.preset_dir
                )
            except ValueError as exc:
                return AddResult(
                    model,
                    notes=(f"No preset written: {exc}. Write one on the Presets screen (a).",),
                )
            verb = "Wrote starter preset" if starter.created else "Uses preset"
            refreshed = RegistryService(self.db).get_model(model.id) if model.id else None
            return AddResult(
                refreshed or model,
                preset=starter.alias,
                serve=f"llmctl run {starter.alias}",
                notes=(f"{verb} {starter.alias}" + (f" ({starter.path})" if starter.path else ""),)
                + starter.notes,
            )
        if model.runtime in (RuntimeName.OLLAMA, RuntimeName.LMSTUDIO):
            return AddResult(
                model,
                notes=(f"{model.runtime.value} loads it on the first request; nothing to start.",),
            )
        if model.runtime == RuntimeName.LLAMA_CPP:
            return AddResult(model, serve=f"llmctl run {model.name}")
        if model.runtime in (RuntimeName.SGLANG, RuntimeName.FREETOKEN, RuntimeName.STRATA):
            return AddResult(model, serve=f"llmctl run {model.name}")
        return AddResult(model)

    def register(self, model: Model) -> Model:
        """Insert or refresh the row for ``model``; revive it if it was deleted.

        Identity is the scan key ``(runtime, source, host)``, so a later
        ``scan`` refreshes this row instead of adding a twin. A row removed
        earlier comes back: re-adding a model is how an operator undoes a
        remove, and a hidden row would make the add look like it did nothing.
        """
        source = model.source or model.name
        host = effective_host(model.host)
        record = self.db.exec(
            select(ModelRecord).where(
                ModelRecord.runtime == model.runtime,
                ModelRecord.source == source,
                ModelRecord.host == host,
            )
        ).first()
        if record is None:
            record = ModelRecord(
                name=model.name,
                runtime=model.runtime,
                source=source,
                host=host,
                status=model.status,
                active=True,
                tags=list(model.tags),
                metadata_json=dict(model.metadata),
            )
        elif record.status in (ModelStatus.DELETED, ModelStatus.MISSING):
            record.status = model.status
            record.active = True
        record.name = model.name or record.name
        record.path = model.path or record.path
        record.format = model.format or record.format
        record.quantization = model.quantization or record.quantization
        record.size_bytes = model.size_bytes or record.size_bytes
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_model(record)

    def _add_ollama(self, plan: AddPlan, say: Progress) -> Model:
        adapter = self._router().get_adapter(RuntimeName.OLLAMA)
        pull = getattr(adapter, "pull_model", None)
        if pull is None:
            raise ModelOpsError("The ollama adapter cannot pull.")
        last: dict[str, Any] = {"status": None, "decade": -1}

        def on_progress(status: str, completed: int | None, total: int | None) -> None:
            if status and status != last["status"]:
                last["status"], last["decade"] = status, -1
                if not total:
                    say(status)
            if total and completed is not None:
                decade = min(10, completed * 10 // total)
                if decade > last["decade"]:
                    last["decade"] = decade
                    say(f"{human_size(completed)} / {human_size(total)} ({decade * 10}%)")

        status = run_sync(pull(plan.source, on_progress=on_progress))
        if status.state != HealthState.OK:
            raise ModelOpsError(str(status.message))
        found = None
        try:
            for item in run_sync(adapter.discover_models()):
                if (item.source or item.name) == plan.source:
                    found = item
                    break
        except Exception:
            found = None
        model = found or Model(
            name=plan.source, runtime=RuntimeName.OLLAMA, source=plan.source
        )
        return self.register(
            model.model_copy(update={"name": plan.name, "status": ModelStatus.DISCOVERED})
        )

    def _hf(self) -> Any:
        if self.hf_api is None:
            from huggingface_hub import HfApi

            self.hf_api = HfApi()
        return self.hf_api

    def resolve_hf_files(self, plan: AddPlan) -> tuple[list[str], int]:
        """Files ``plan`` will download and their total size (network)."""
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        try:
            info = self._hf().model_info(plan.source, files_metadata=True)
        except GatedRepoError as exc:
            raise ModelOpsError(
                f"{plan.source} is gated: accept its license on huggingface.co, "
                "then `hf auth login`."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise ModelOpsError(
                f"Hugging Face has no repo {plan.source} (or it is private: `hf auth login`)."
            ) from exc
        siblings = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
        if plan.engine == RuntimeName.LLAMA_CPP:
            ggufs = [
                f
                for f in siblings
                if f.lower().endswith(".gguf") and not Path(f).name.lower().startswith("mmproj")
            ]
            if not ggufs:
                raise ModelOpsError(f"{plan.source} has no .gguf files; try the vllm engine.")
            if plan.filename:
                if plan.filename not in siblings:
                    raise ModelOpsError(f"{plan.source} has no file {plan.filename}.")
                chosen = [plan.filename]
            elif plan.quant:
                pat = re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(plan.quant)}(?![A-Za-z0-9_])", re.IGNORECASE
                )
                chosen = [f for f in ggufs if pat.search(f)]
                if not chosen:
                    raise ModelOpsError(
                        f"No {plan.quant} file in {plan.source}. Available: "
                        + _gguf_menu(ggufs)
                    )
            else:
                chosen = ggufs
            groups = {_SHARD_RE.sub("", f) for f in chosen}
            if len(groups) > 1:
                raise ModelOpsError(
                    f"{plan.source} has several GGUF builds; add a quant, e.g. "
                    f"{plan.source}:Q4_K_M. Available: " + _gguf_menu(chosen)
                )
            files = sorted(chosen)
            return files, sum(siblings[f] for f in files)
        weights = [f for f in siblings if f.endswith((".safetensors", ".bin"))]
        if not weights:
            raise ModelOpsError(
                f"{plan.source} has no safetensors weights. For a GGUF repo use "
                "the llama_cpp engine."
            )
        from fnmatch import fnmatch

        files = [
            f for f in siblings if not any(fnmatch(f, pat) for pat in _HF_SAFETENSORS_IGNORE)
        ]
        return sorted(files), sum(siblings[f] for f in files)

    def _check_disk(self, need: int) -> None:
        cache = self.hub_cache or _hf_hub_cache()
        probe = cache
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = self.disk_free(probe)
        if need + DISK_MARGIN_BYTES > free:
            raise ModelOpsError(
                f"Not enough disk: the download needs {human_size(need)} and "
                f"{cache} has {human_size(free)} free (llmctl keeps "
                f"{human_size(DISK_MARGIN_BYTES)} spare). Free space first."
            )

    def _add_hf(self, plan: AddPlan, say: Progress) -> Model:
        files, need = self.resolve_hf_files(plan)
        self._check_disk(need)
        say(f"Downloading {len(files)} file(s), {human_size(need)}, from {plan.source}")
        download = self.download
        if download is None:
            from huggingface_hub import snapshot_download

            download = snapshot_download
        kwargs: dict[str, Any] = {"repo_id": plan.source, "allow_patterns": files}
        if self.hub_cache is not None:
            kwargs["cache_dir"] = str(self.hub_cache)
        snapshot = Path(download(**kwargs))
        if plan.engine == RuntimeName.LLAMA_CPP:
            first = next((f for f in files if not _SHARD_RE.search(f)), None) or files[0]
            path = snapshot / first
            fmt, quant = "gguf", plan.quant
        else:
            path, fmt, quant = snapshot, "safetensors", None
        say(f"Downloaded to {path}")
        return self.register(
            Model(
                name=plan.name,
                runtime=plan.engine,
                # vLLM's live probe reports the repo id; keying on it lets a
                # later scan land on this same row.
                source=plan.source if plan.engine != RuntimeName.LLAMA_CPP else str(path),
                path=str(path),
                format=fmt,
                quantization=quant,
                size_bytes=need,
                status=ModelStatus.DISCOVERED,
            )
        )

    def _add_lmstudio(self, plan: AddPlan, say: Progress) -> Model:
        say(f"Running: lms get {plan.source} --yes")
        try:
            proc = self.runner(
                [lms_binary() or "lms", "get", plan.source, "--yes"],
                capture_output=True,
                text=True,
                timeout=6 * 3600,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ModelOpsError("The `lms` CLI is not on PATH; install LM Studio.") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise ModelOpsError("lms get failed: " + " | ".join(tail))
        return self.register(
            Model(
                name=plan.name,
                runtime=RuntimeName.LMSTUDIO,
                source=plan.source,
                status=ModelStatus.DISCOVERED,
            )
        )

    # -- remove ------------------------------------------------------------

    def remove(self, model_id: str, *, purge: bool = False) -> RemoveResult:
        """Hide the row; with ``purge`` remove the model from its engine first.

        Purge runs first and a failed purge leaves the row in place, so the
        operator can see the model is still there and retry.
        """
        registry = RegistryService(self.db, self.router)
        model = registry.get_model(model_id)
        if model is None:
            return RemoveResult(False, False, "Nothing changed: the model was not found.")
        freed: int | None = None
        purged = False
        if purge:
            plan = plan_remove(
                model,
                db=self.db,
                hub_cache=self.hub_cache,
                runner=self.runner,
                preset_dir=self.preset_dir,
            )
            if plan.blockers:
                raise ModelOpsError("Not removed: " + " ".join(plan.blockers))
            if not plan.can_purge:
                raise ModelOpsError("Not removed: " + plan.purge_summary)
            if model.runtime == RuntimeName.OLLAMA:
                adapter = self._router().get_adapter(RuntimeName.OLLAMA)
                status = run_sync(adapter.delete_model(model))
                if status.state != HealthState.OK:
                    raise ModelOpsError(str(status.message))
            else:
                for target in plan.targets:
                    _unlink_target(target)
            if plan.presets_to_delete:
                from llmctl.presets import delete_preset

                for alias in plan.presets_to_delete:
                    delete_preset(
                        alias, **({"config_dir": self.preset_dir} if self.preset_dir else {})
                    )
            freed, purged = plan.size_bytes, True
        registry.delete_model(model_id)
        if purged:
            what = "removed from Ollama" if model.runtime == RuntimeName.OLLAMA else "files deleted"
            size = f", {human_size(freed)} freed" if freed else ""
            if plan.presets_to_delete:
                what += f", preset {', '.join(plan.presets_to_delete)} deleted"
            return RemoveResult(True, True, f"Removed {model.name}: {what}{size}.", freed)
        return RemoveResult(
            True,
            False,
            f"Removed {model.name} from llmctl. The model itself is untouched; "
            "`add` it again to bring the row back.",
        )


def _gguf_menu(files: list[str]) -> str:
    groups = sorted({_SHARD_RE.sub("", Path(f).name).removesuffix(".gguf") for f in files})
    # "Qwen3-8B-Q4_K_M" -> "Q4_K_M": the shared model-name prefix is noise
    # when the operator is choosing what to type after the colon.
    prefix = os.path.commonprefix(groups) if len(groups) > 1 else ""
    cut = max(prefix.rfind("-"), prefix.rfind("."))
    if cut > 0:
        groups = [g[cut + 1 :] for g in groups]
    shown = ", ".join(groups[:10])
    return shown + (f" (+{len(groups) - 10} more)" if len(groups) > 10 else "")


__all__ = [
    "ADDABLE_ENGINES",
    "AddPlan",
    "AddResult",
    "ModelOps",
    "ModelOpsError",
    "RemovePlan",
    "RemoveResult",
    "human_size",
    "normalize_ollama_tag",
    "plan_add",
    "plan_remove",
]
