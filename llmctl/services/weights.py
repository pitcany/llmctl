"""Are a target's weights actually on this machine?

A registry row, a vLLM preset and a router lane all outlive their weights:
``ollama rm``, a pruned hub cache or a moved model directory touch none of
them. The run list then keeps offering a model that can only fail -- and on
this host the failure is expensive, because starting a vLLM preset or a GPU
lane first stops whatever is serving. ``llmctl ls`` marks such rows
``weights gone`` and ``llmctl run`` refuses them before stopping anything.

Three sources of truth, one verdict each, and every one is read-only:

* a vLLM preset's ``model_id``: an absolute path must exist; a Hugging Face
  repo id must have a snapshot in a hub cache (``HF_HOME``,
  ``HUGGINGFACE_HUB_CACHE``, ``~/.cache/huggingface``), since a managed
  single unit must never start a multi-gigabyte download mid-swap;
* an Ollama lane's ``served_model``: ``/api/tags`` on the lane's own server,
  or on the host's default Ollama when the lane's server is stopped (they
  share ``OLLAMA_MODELS`` on one host);
* anything else: no opinion (``None``), so the row stays listed.

``None`` always means "could not tell", never "gone".
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Seconds an Ollama server may take to answer ``/api/tags``.
TAGS_TIMEOUT_S = 1.5

HttpGet = Callable[[str, float], Any]

#: What a preset ``model_id`` looks like when it is a filesystem path.
_LOCAL_PATH_PREFIXES = ("/", "~", ".")


def _default_http_get(url: str, timeout: float) -> Any:
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only


@dataclass(frozen=True)
class Verdict:
    """``present`` is ``True`` (found), ``False`` (gone) or ``None`` (unknown)."""

    present: bool | None
    reason: str

    @property
    def gone(self) -> bool:
        return self.present is False


# --------------------------------------------------------------------------- #
# Hugging Face hub cache
# --------------------------------------------------------------------------- #


def hub_cache_roots(extra: Iterable[Path | str] = ()) -> list[Path]:
    """Hub cache directories in lookup order: ``extra`` first, then the environment's."""
    roots: list[Path] = [Path(os.path.expanduser(str(p))) for p in extra]
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.path.expanduser(os.environ["HUGGINGFACE_HUB_CACHE"])))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.path.expanduser(os.environ["HF_HOME"])) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def repo_in_hub_cache(repo_id: str, roots: Sequence[Path]) -> bool:
    """True when some root holds ``models--org--name/snapshots/<x>/config.json``.

    A dangling symlink or an empty ``snapshots`` dir (the weights were moved
    or pruned, the folder left behind) does not count.
    """
    folder = "models--" + repo_id.strip().replace("/", "--")
    for root in roots:
        snapshots = root / folder / "snapshots"
        try:
            if not snapshots.is_dir():
                continue
            for snap in snapshots.iterdir():
                if snap.is_dir() and (snap / "config.json").is_file():
                    return True
        except OSError:
            continue
    return False


def is_local_path(model: str) -> bool:
    return model.strip().startswith(_LOCAL_PATH_PREFIXES)


def vllm_model_verdict(model: str | None, *, hf_home: str | None = None) -> Verdict:
    """Where a vLLM preset's ``model_id`` resolves, or that it does not.

    A local path is checked on disk. A repo id (``org/name``) is looked up
    in the hub caches, the preset's own ``HF_HOME`` first. Anything else
    (empty, or a shape llmctl does not recognise) is ``None``.
    """
    raw = (model or "").strip()
    if not raw:
        return Verdict(None, "no model_id")
    if is_local_path(raw):
        path = Path(raw).expanduser()
        try:
            path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return Verdict(False, f"model path does not exist: {path}")
        except OSError as exc:
            return Verdict(None, f"cannot read {path}: {exc.strerror or exc}")
        return Verdict(True, f"found at {path}")
    if raw.count("/") == 1 and not raw.startswith(("http://", "https://")):
        extra = (Path(os.path.expanduser(hf_home)) / "hub",) if hf_home else ()
        roots = hub_cache_roots(extra)
        if repo_in_hub_cache(raw, roots):
            return Verdict(True, f"{raw} is in the Hugging Face hub cache")
        return Verdict(
            False,
            f"{raw} is not in any Hugging Face hub cache "
            f"({', '.join(str(r) for r in roots)})",
        )
    return Verdict(None, f"cannot judge {raw!r}")


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def ollama_installed_models(
    base_url: str, *, timeout: float = TAGS_TIMEOUT_S, http_get: HttpGet | None = None
) -> list[str] | None:
    """Model names an Ollama server has on disk (``/api/tags``); ``None`` if it does not answer."""
    http_get = http_get or _default_http_get
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        payload = json.loads(http_get(url, timeout).read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    names: list[str] = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict):
            names += [v for v in (item.get("name"), item.get("model")) if isinstance(v, str)]
    return names


def _ollama_names_match(wanted: str, installed: Iterable[str]) -> bool:
    """``qwen3:8b`` and ``qwen3:8b`` match; so do ``m`` and ``m:latest``."""
    want = wanted.strip()
    if not want:
        return False
    candidates = {want, f"{want}:latest"} if ":" not in want else {want}
    return any(name in candidates for name in installed)


def ollama_verdict_from_installed(
    served_model: str | None, installed: Iterable[str], base_url: str
) -> Verdict:
    """Judge ``served_model`` against one server's ``/api/tags`` answer."""
    if not served_model:
        return Verdict(None, "no served model")
    if _ollama_names_match(served_model, installed):
        return Verdict(True, f"{served_model} is installed on {base_url}")
    return Verdict(
        False,
        f"{served_model} is not installed on {base_url} (`ollama list`); "
        f"`llmctl add {served_model}` pulls it",
    )


def ollama_model_verdict(
    served_model: str | None,
    endpoints: Sequence[str],
    *,
    timeout: float = TAGS_TIMEOUT_S,
    http_get: HttpGet | None = None,
) -> Verdict:
    """Whether some Ollama server in ``endpoints`` has ``served_model`` installed.

    The first server that answers decides: a stopped lane server says
    nothing, so the host's default Ollama (same ``OLLAMA_MODELS``) is asked
    next. No answer at all is ``None``.
    """
    if not served_model:
        return Verdict(None, "no served model")
    for base in endpoints:
        installed = ollama_installed_models(base, timeout=timeout, http_get=http_get)
        if installed is not None:
            return ollama_verdict_from_installed(served_model, installed, base)
    return Verdict(None, "no Ollama server answered /api/tags")


# --------------------------------------------------------------------------- #
# The run list
# --------------------------------------------------------------------------- #


def gone_targets(
    catalog: Iterable[Any],
    *,
    presets: dict[str, Any] | None = None,
    ollama_default: str | None = None,
    http_get: HttpGet | None = None,
) -> dict[str, str]:
    """Catalog key -> why its weights are gone, for the rows that can be judged.

    ``presets`` maps a vLLM alias to its launch spec (``model`` attribute);
    ``ollama_default`` is the host's default Ollama endpoint, asked when a
    lane's own server is down. Each Ollama server is asked once per listing.
    Rows that cannot be judged are left out.
    """
    presets = presets or {}
    gone: dict[str, str] = {}
    tags_cache: dict[str, list[str] | None] = {}

    def tags(base: str) -> list[str] | None:
        if base not in tags_cache:
            tags_cache[base] = ollama_installed_models(base, http_get=http_get)
        return tags_cache[base]

    for row in catalog:
        kind = getattr(row, "kind", None)
        if kind == "vllm":
            spec = presets.get(row.key)
            if spec is None:
                continue
            verdict = vllm_model_verdict(getattr(spec, "model", None))
        elif kind == "lane" and getattr(row.lane, "runtime", None) == "ollama":
            verdict = Verdict(None, "no Ollama server answered /api/tags")
            for base in (e for e in (row.upstream, ollama_default) if e):
                installed = tags(base)
                if installed is not None:
                    verdict = ollama_verdict_from_installed(row.lane.served_model, installed, base)
                    break
        else:
            continue
        if verdict.gone:
            gone[row.key] = verdict.reason
    return gone
