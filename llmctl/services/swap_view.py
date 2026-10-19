"""The swap list as plain data, shared by ``llmctl ls --json`` and the API.

The API and the CLI share one builder so the two cannot drift: a row means
the same thing on the command line and over HTTP. Nothing here imports
``typer`` or ``rich``; rendering stays in ``cli_swap``.
"""

from __future__ import annotations

from typing import Any

from llmctl.services.lanes import SwapState, SwapTarget, cards_label, last_load_s


def gateway_aliases(settings: Any) -> list[Any]:
    """The gateway's alias views; best effort, ``[]`` when anything fails."""
    try:
        from sqlmodel import Session

        from llmctl.db import get_engine, init_db
        from llmctl.services.gateway import GatewayService

        init_db(settings.database_url)
        with Session(get_engine(settings.database_url)) as db:
            return list(GatewayService(db, settings).alias_view())
    except Exception:  # noqa: BLE001 - aliases are decoration on `ls`
        return []


def _aliases_for(row: SwapTarget, serving: bool, aliases: list[Any] | None) -> list[str]:
    if not serving:
        return []
    return sorted(
        f"local-{a.name}"
        for a in aliases or []
        if a.healthy and a.resolved_served_name == row.model
    )


def gateway_url(router: Any) -> str:
    """Where the gateway answers; one spelling for the CLI and the API."""
    return f"http://{router.host}:{router.port}/v1"


def endpoint_for(state: SwapState, target: SwapTarget) -> str | None:
    """The ``endpoint`` of a row: its upstream's ``/v1``, ``None`` without one.

    ``state`` is unused today; it is taken so a row's endpoint can later
    depend on the swap state without changing callers.
    """
    return f"{target.upstream}/v1" if target.upstream else None


def rows_json(state: SwapState, aliases: list[Any] | None = None) -> list[dict[str, Any]]:
    """The swap list as plain data, in catalog order (``llmctl ls --json``)."""
    return [
        {
            "number": index,
            "name": row.key,
            "model": row.model,
            "engine": row.engine,
            "kind": row.kind,
            "cards": cards_label(row),
            "state": state.state_of(row),
            "last_load_s": last_load_s(row.key),
            "endpoint": endpoint_for(state, row),
            "aliases": _aliases_for(row, state.state_of(row) == "serving", aliases),
        }
        for index, row in enumerate(state.catalog, start=1)
    ]
