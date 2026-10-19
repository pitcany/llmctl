"""Static dashboard served at /ui.

Three explicit routes map a fixed file name to a media type, so no request
input ever reaches the filesystem (no traversal, no directory listing).
Bytes are read through ``importlib.resources`` so this works from the wheel.
"""

from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import RedirectResponse, Response

router = APIRouter(include_in_schema=False)

#: Fixed name -> media type; the only files this module ever opens.
_MEDIA_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}

UI_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; img-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


def _static(name: str, extra: dict[str, str] | None = None) -> Response:
    data = (files("llmctl.api") / "static" / name).read_bytes()
    headers = {"X-Content-Type-Options": "nosniff", **(extra or {})}
    return Response(content=data, media_type=_MEDIA_TYPES[name], headers=headers)


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/ui", status_code=307)


@router.get("/ui")
def ui_index() -> Response:
    return _static("index.html", {"Content-Security-Policy": UI_CSP})


@router.get("/ui/app.js")
def ui_js() -> Response:
    return _static("app.js")


@router.get("/ui/app.css")
def ui_css() -> Response:
    return _static("app.css")
