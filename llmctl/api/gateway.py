"""OpenAI-compatible router/gateway FastAPI app.

This is a *separate* ASGI app from the control-plane API in
:mod:`llmctl.api.app`. Mounting it standalone keeps the two surfaces
independent: an outage in the gateway's upstream sessions can't take
down the control plane, and the control plane never needs to load
httpx/proxy code paths it doesn't use.

The gateway exposes a tight OpenAI subset:

* ``GET  /health``               — gateway liveness + router config view
* ``GET  /v1/models``            — active sessions + bound aliases
* ``POST /v1/chat/completions``  — proxied (streaming preserved)
* ``POST /v1/completions``       — proxied (streaming preserved)

Resolution lives in :class:`~llmctl.services.gateway.GatewayService`;
this module is the thin proxy/auth shell around it. Streaming is done
via ``httpx.AsyncClient.stream(...)`` so the upstream's SSE chunks pass
through unchanged — the gateway never buffers a streaming response in
full, which matters for long completions.

Bearer-token auth is enforced for every ``/v1/*`` route when
``router.auth_token`` is set in settings. ``/health`` deliberately
*does* require the token too, when configured, because the token-aware
posture is "if you have to send the header anywhere, send it
everywhere"; the alternative leaks a public liveness probe that lets an
unauthenticated caller enumerate the gateway's existence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.config import Settings, load_settings
from llmctl.db import SQLModel, apply_migrations, get_engine
from llmctl.services.gateway import GatewayService, RouteTarget

# Chat/completions can take many minutes; the gateway is intentionally
# patient with the upstream. Connect timeout stays short so a dead
# upstream fails fast.
_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0)


def create_gateway_app(
    settings: Settings | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Build the gateway FastAPI app.

    Schema migrations run at boot for the same reason the control-plane
    app does it: the gateway needs ``SessionRecord`` rows to resolve
    routes, and a fresh DB file would otherwise 500 on every request
    until someone ran ``llmctl scan``.
    """
    effective_settings = settings or load_settings()
    if not effective_settings.router.enabled:
        # The factory still returns an app — disabling the router only
        # affects launch from the CLI. Returning a real app means tests
        # can still spin one up to assert the disabled-mode behavior.
        pass
    effective_database_url = database_url or effective_settings.database_url
    engine = get_engine(effective_database_url)
    SQLModel.metadata.create_all(engine)
    apply_migrations(engine)

    app = FastAPI(
        title="LLMCTL Gateway",
        version="0.1.0",
        description="Local OpenAI-compatible router for active llmctl sessions.",
    )

    def session_dependency() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db_session] = session_dependency

    def require_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        token = effective_settings.router.auth_token
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health", tags=["health"])
    def health(
        _: None = Depends(require_auth),
        db: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        service = GatewayService(db, effective_settings)
        aliases = service.alias_view()
        return {
            "status": "ok",
            "router": {
                "host": effective_settings.router.host,
                "port": effective_settings.router.port,
                "auth_required": bool(effective_settings.router.auth_token),
                "auto_start": effective_settings.router.auto_start,
                "fallback_policy": effective_settings.router.fallback_policy,
            },
            "aliases": [
                {
                    "name": a.name,
                    "target": a.target,
                    "session_id": a.resolved_session_id,
                    "served_name": a.resolved_served_name,
                    "healthy": a.healthy,
                }
                for a in aliases
            ],
        }

    @app.get("/v1/models", tags=["openai"])
    def list_models(
        _: None = Depends(require_auth),
        db: Session = Depends(get_db_session),
    ) -> dict[str, Any]:
        models = GatewayService(db, effective_settings).list_models()
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", tags=["openai"])
    async def chat_completions(
        request: Request,
        _: None = Depends(require_auth),
        db: Session = Depends(get_db_session),
    ) -> Any:
        return await _proxy(request, db, effective_settings, "/v1/chat/completions")

    @app.post("/v1/completions", tags=["openai"])
    async def completions(
        request: Request,
        _: None = Depends(require_auth),
        db: Session = Depends(get_db_session),
    ) -> Any:
        return await _proxy(request, db, effective_settings, "/v1/completions")

    return app


async def _proxy(
    request: Request,
    db: Session,
    settings: Settings,
    path: str,
) -> Any:
    """Resolve the route, rewrite ``model``, and proxy the request body.

    Streaming responses are passed through line-by-line via
    ``StreamingResponse``; non-streaming responses round-trip as JSON.
    Errors surfaced from the upstream are wrapped in OpenAI-style error
    payloads so callers don't have to disambiguate gateway-vs-upstream
    failures.
    """
    payload = await _read_json(request)
    if payload is None:
        return _openai_error("Invalid JSON body.", code="bad_request", http_status=400)

    requested = str(payload.get("model", "")).strip()
    if not requested:
        return _openai_error(
            "Missing 'model' field. Pass an alias (e.g. 'local-coding'), a "
            "served model name, or a session id.",
            code="missing_model",
            http_status=400,
        )

    target = GatewayService(db, settings).resolve(requested)
    if target is None:
        return _openai_error(
            f"No active local session resolves to model '{requested}'. "
            "Start one with `llmctl start` or bind an alias with `llmctl set-alias`.",
            code="model_unavailable",
            http_status=503,
        )

    upstream_payload = dict(payload)
    upstream_payload["model"] = target.served_name
    streaming = bool(payload.get("stream"))

    url = f"{target.base_url.rstrip('/')}{path}"
    headers = _forwarded_headers(request)
    headers["X-Llmctl-Route"] = target.via
    headers["X-Llmctl-Session"] = target.session_id

    if streaming:
        return await _stream_proxy(url, upstream_payload, headers, target)
    return await _json_proxy(url, upstream_payload, headers, target)


async def _read_json(request: Request) -> dict[str, Any] | None:
    """Return the request body as a dict, or ``None`` on parse failure."""
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - any decode error is "bad JSON"
        return None
    return data if isinstance(data, dict) else None


def _forwarded_headers(request: Request) -> dict[str, str]:
    """Headers to forward upstream — drop hop-by-hop + the gateway token."""
    drop = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        # Caller's gateway token must not leak to the upstream; upstreams
        # have their own auth scheme (or no auth at all on localhost).
        "authorization",
    }
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


async def _json_proxy(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    target: RouteTarget,
) -> JSONResponse:
    """Proxy a non-streaming request and round-trip the JSON response."""
    try:
        async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        return _openai_error(
            f"Upstream session {target.session_id} unreachable: {exc}",
            code="upstream_unreachable",
            http_status=502,
        )
    return JSONResponse(
        content=_safe_json(resp),
        status_code=resp.status_code,
        headers={
            "X-Llmctl-Route": target.via,
            "X-Llmctl-Session": target.session_id,
        },
    )


async def _stream_proxy(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    target: RouteTarget,
) -> StreamingResponse:
    """Proxy a streaming request; SSE chunks pass through unchanged."""

    async def body() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        # Drain into an error event so the SSE consumer
                        # sees something parseable rather than a silent
                        # disconnect.
                        body_bytes = await resp.aread()
                        yield (
                            b"data: "
                            + _openai_error_bytes(
                                body_bytes.decode("utf-8", errors="replace")
                                or f"upstream returned HTTP {resp.status_code}",
                                code="upstream_error",
                            )
                            + b"\n\n"
                        )
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
        except httpx.HTTPError as exc:
            yield (
                b"data: "
                + _openai_error_bytes(
                    f"Upstream session {target.session_id} unreachable: {exc}",
                    code="upstream_unreachable",
                )
                + b"\n\n"
            )

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "X-Llmctl-Route": target.via,
            "X-Llmctl-Session": target.session_id,
            # SSE-friendly: disable buffering on intermediate proxies.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _safe_json(resp: httpx.Response) -> Any:
    """Decode upstream JSON or wrap the raw body in an error envelope."""
    try:
        return resp.json()
    except ValueError:
        return _openai_error_dict(
            resp.text or f"upstream returned HTTP {resp.status_code}",
            code="upstream_bad_response",
        )


def _openai_error(message: str, *, code: str, http_status: int) -> JSONResponse:
    """Return an OpenAI-style error envelope as a JSONResponse."""
    return JSONResponse(
        status_code=http_status,
        content=_openai_error_dict(message, code=code),
    )


def _openai_error_dict(message: str, *, code: str) -> dict[str, Any]:
    """Return the OpenAI-style error envelope as a dict."""
    return {
        "error": {
            "message": message,
            "type": "llmctl_gateway_error",
            "code": code,
        }
    }


def _openai_error_bytes(message: str, *, code: str) -> bytes:
    """JSON-encode an error envelope for streaming SSE wrapping."""
    import json

    return json.dumps(_openai_error_dict(message, code=code)).encode("utf-8")
