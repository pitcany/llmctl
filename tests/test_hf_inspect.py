"""inspect_model error mapping, with the Hub stubbed out."""

from __future__ import annotations

from unittest.mock import patch

import httpx
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from llmctl.hf.inspect import inspect_model

_RESPONSE = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))


def _model_info_raises(exc: Exception):
    return patch("huggingface_hub.HfApi.model_info", side_effect=exc)


def test_gated_repo_gets_the_license_hint_not_not_found():
    # GatedRepoError subclasses RepositoryNotFoundError in huggingface_hub.
    assert issubclass(GatedRepoError, RepositoryNotFoundError)
    with _model_info_raises(GatedRepoError("gated", response=_RESPONSE)):
        meta = inspect_model("meta-llama/Llama-3.3-70B-Instruct")
    assert meta.gated is True
    assert "Gated repo" in (meta.error or "")


def test_missing_repo_is_not_found():
    with _model_info_raises(RepositoryNotFoundError("missing", response=_RESPONSE)):
        meta = inspect_model("nobody/nothing")
    assert meta.gated is False
    assert "not found" in (meta.error or "")


def test_transient_hub_error_is_reported_not_raised():
    with _model_info_raises(HfHubHTTPError("503 Service Unavailable", response=_RESPONSE)):
        meta = inspect_model("Qwen/Qwen2.5-7B-Instruct")
    assert "Could not fetch model info" in (meta.error or "")
    assert meta.config == {}
