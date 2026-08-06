"""Binding the control plane off loopback must not be one flag away from RCE.

`POST /sessions/start` with `runtime=python_script` is local command execution by
design: the scheduler builds `[sys.executable, parameters['script'], *args]`. Two
defaults used to stand between that and the network:

* `scheduler.require_auth_token` is False, so `create_app` never installs the
  bearer middleware and every mutating route is open; and
* the only bind refusal was gated on `scheduler.allow_public_bind` -- the *same*
  flag the scheduler tells operators to set to bind a **model** publicly
  ("Binding to public host {host}; set scheduler.allow_public_bind=true to
  allow").

So enabling public model serving silently also permitted binding the
unauthenticated control plane to 0.0.0.0.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmctl import cli
from llmctl.config import Settings


@pytest.fixture
def settings(monkeypatch) -> Settings:
    """Default settings, with uvicorn stubbed so nothing actually binds."""
    s = Settings()
    monkeypatch.setattr(cli, "load_settings", lambda: s)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
    return s


def _serve(*args: str):
    return CliRunner().invoke(cli.app, ["serve", *args])


# --- H2a: the two flags must be independent ----------------------------------


def test_model_public_bind_does_not_open_the_control_plane(settings) -> None:
    """scheduler.allow_public_bind is about models; it must not bind the API."""
    settings.scheduler.allow_public_bind = True

    res = _serve("--host", "0.0.0.0")

    assert res.exit_code == 2, res.output
    assert "0.0.0.0" in res.output


def test_api_public_bind_flag_is_what_permits_it(settings) -> None:
    """The API has its own opt-in -- but see the token requirement below."""
    settings.api.allow_public_bind = True
    settings.api.auth_token = "s3cret"

    res = _serve("--host", "0.0.0.0")

    assert res.exit_code == 0, res.output


def test_loopback_is_unaffected(settings) -> None:
    res = _serve("--host", "127.0.0.1")
    assert res.exit_code == 0, res.output


# --- H2b: a public bind must actually enforce auth ----------------------------


def test_public_bind_without_a_token_is_refused(settings) -> None:
    """Opting in to a public bind is not enough; there must be a real token."""
    settings.api.allow_public_bind = True
    settings.api.auth_token = None

    res = _serve("--host", "0.0.0.0")

    assert res.exit_code == 2, res.output
    assert "token" in res.output.lower()


def test_public_bind_enforces_auth_even_when_the_flag_is_off(settings, monkeypatch) -> None:
    """`require_auth_token` defaults False -- a public bind must override it.

    Otherwise the token exists but the middleware is never installed, and every
    mutating route stays open.
    """
    settings.api.allow_public_bind = True
    settings.api.auth_token = "s3cret"
    settings.scheduler.require_auth_token = False
    captured: dict = {}
    monkeypatch.setattr(
        cli, "create_app", lambda s: captured.setdefault("settings", s)
    )

    res = _serve("--host", "0.0.0.0")

    assert res.exit_code == 0, res.output
    used = captured["settings"]
    assert used.scheduler.require_auth_token is True, (
        "public bind did not force the bearer middleware on"
    )


def test_loopback_does_not_force_auth_on(settings, monkeypatch) -> None:
    """The local default stays frictionless."""
    captured: dict = {}
    monkeypatch.setattr(
        cli, "create_app", lambda s: captured.setdefault("settings", s)
    )

    res = _serve("--host", "127.0.0.1")

    assert res.exit_code == 0, res.output
    assert captured["settings"].scheduler.require_auth_token is False
