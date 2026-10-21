"""Fetch model metadata from Hugging Face without downloading weights."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelMetadata:
    model_id: str
    config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    tags: list[str]
    pipeline_tag: str | None
    gated: bool
    private: bool
    error: str | None = None
    #: Total size of the repo's ``*.safetensors`` files, when the Hub reports it:
    #: the real weight size, whatever the quantization format.
    weight_bytes: int | None = None


def _try_load_json(model_id: str, filename: str) -> dict[str, Any]:
    """Download a small JSON metadata file from HF (no weights)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, repo_type="model")
    except EntryNotFoundError:
        return {}
    except (GatedRepoError, RepositoryNotFoundError):
        raise
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def inspect_model(model_id: str) -> ModelMetadata:
    """Return metadata + small JSON configs. Never downloads weights."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.model_info(model_id, files_metadata=True)
    # GatedRepoError subclasses RepositoryNotFoundError, so it must come first.
    except GatedRepoError:
        return ModelMetadata(
            model_id=model_id,
            config={},
            tokenizer_config={},
            tags=[],
            pipeline_tag=None,
            gated=True,
            private=False,
            error=(
                "Gated repo. Accept the license on Hugging Face and set "
                "HF_TOKEN (or run `huggingface-cli login`)."
            ),
        )
    except RepositoryNotFoundError:
        return ModelMetadata(
            model_id=model_id,
            config={},
            tokenizer_config={},
            tags=[],
            pipeline_tag=None,
            gated=False,
            private=False,
            error=f"Model not found on Hugging Face: {model_id}",
        )
    except Exception as e:  # noqa: BLE001 - 5xx, rate limits, network errors
        return ModelMetadata(
            model_id=model_id,
            config={},
            tokenizer_config={},
            tags=[],
            pipeline_tag=None,
            gated=False,
            private=False,
            error=f"Could not fetch model info: {e}",
        )

    config: dict[str, Any] = {}
    tok: dict[str, Any] = {}
    err: str | None = None
    try:
        config = _try_load_json(model_id, "config.json")
        tok = _try_load_json(model_id, "tokenizer_config.json")
    except GatedRepoError:
        err = "Gated repo. Accept the license and set HF_TOKEN."
    except Exception as e:  # noqa: BLE001
        err = f"Could not read metadata: {e}"

    return ModelMetadata(
        model_id=info.id,
        config=config,
        tokenizer_config=tok,
        tags=list(info.tags or []),
        pipeline_tag=info.pipeline_tag,
        gated=bool(getattr(info, "gated", False)),
        private=bool(getattr(info, "private", False)),
        error=err,
        weight_bytes=_safetensors_bytes(getattr(info, "siblings", None)),
    )


def _safetensors_bytes(siblings: Any) -> int | None:
    """Size of the ``*.safetensors`` checkpoint vLLM loads, or None when unknown.

    A repo can ship one checkpoint twice: a single ``model.safetensors`` beside
    its shards (``model-00001-of-00004.safetensors``), or Mistral's
    ``consolidated.safetensors`` beside the HF-format files. Counting both
    doubles the size, so the shards win over the single file, and the
    ``consolidated`` copy is dropped when other weights exist.
    """
    sizes: dict[str, int] = {}
    for sib in siblings or []:
        name = getattr(sib, "rfilename", "") or ""
        size = getattr(sib, "size", None)
        if name.endswith(".safetensors") and isinstance(size, int):
            sizes[name] = size
    shard_prefixes = {
        m.group(1) for name in sizes if (m := _SHARD_RE.match(name.rsplit("/", 1)[-1]))
    }
    kept = {
        name: size
        for name, size in sizes.items()
        if name.rsplit("/", 1)[-1].removesuffix(".safetensors") not in shard_prefixes
    }
    others = [n for n in kept if not n.rsplit("/", 1)[-1].startswith("consolidated")]
    if others:
        kept = {n: kept[n] for n in others}
    return sum(kept.values()) or None


#: ``model-00001-of-00004.safetensors`` -> ``model``.
_SHARD_RE = re.compile(r"^(.+)-\d+-of-\d+\.safetensors$")
