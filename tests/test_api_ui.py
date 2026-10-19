"""The /ui dashboard shell: routes, headers, auth exemption, static hygiene."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from importlib.resources import files

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llmctl import cli
from llmctl.api.app import create_app
from llmctl.api.routes_ui import UI_CSP
from llmctl.config import Settings

NAMES = ("index.html", "app.js", "app.css")


def _settings(tmp_path, **sched) -> Settings:
    s = Settings()
    for k, v in sched.items():
        setattr(s.scheduler, k, v)
    return s


def _app(tmp_path, settings: Settings):
    return create_app(settings, database_url=f"sqlite:///{tmp_path / 'ui.sqlite3'}")


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(_app(tmp_path, _settings(tmp_path)), follow_redirects=False)


def _read(name: str) -> str:
    return (files("llmctl.api") / "static" / name).read_text(encoding="utf-8")


def test_ui_routes_serve_content_types(client) -> None:
    r = client.get("/ui")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    r = client.get("/ui/app.js")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/javascript")
    r = client.get("/ui/app.css")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/css")


def test_root_redirects_to_ui(client) -> None:
    r = client.get("/")
    assert r.status_code == 307
    assert r.headers["location"] == "/ui"


@pytest.mark.parametrize(
    "path",
    ["/ui/index.html", "/ui/../app.py", "/ui/static", "/ui/foo", "/ui/%2e%2e/app.py"],
)
def test_other_ui_paths_404(client, path) -> None:
    assert client.get(path).status_code == 404


def test_headers(client) -> None:
    for p in ("/ui", "/ui/app.js", "/ui/app.css"):
        assert client.get(p).headers["x-content-type-options"] == "nosniff"
    assert client.get("/ui").headers["content-security-policy"] == UI_CSP


def test_routes_hidden_from_schema(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert not {"/", "/ui", "/ui/app.js", "/ui/app.css"} & set(paths)


def test_auth_exemptions(tmp_path) -> None:
    s = _settings(tmp_path, require_auth_token=True)
    s.api.auth_token = "tok"
    c = TestClient(_app(tmp_path, s), follow_redirects=False)
    for p in ("/", "/ui", "/ui/app.js", "/ui/app.css"):
        assert c.get(p).status_code in (200, 307), p
    assert c.get("/swap").status_code == 401


@pytest.mark.parametrize("name", NAMES)
def test_no_absolute_urls(name) -> None:
    assert not re.search(r"https?://", _read(name))


class _Scan(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.bad: list[str] = []
        self.in_script = False
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs) -> None:
        for k, v in attrs:
            if re.fullmatch(r"on\w+", k) or k == "style":
                self.bad.append(k)
            if k == "id" and v:
                self.ids.add(v)
        if tag == "script":
            self.in_script = True
            if not dict(attrs).get("src"):
                self.bad.append("inline script")

    def handle_endtag(self, tag) -> None:
        self.in_script = False

    def handle_data(self, data) -> None:
        if self.in_script and data.strip():
            self.bad.append("script body")


def test_index_has_no_inline_code() -> None:
    p = _Scan()
    p.feed(_read("index.html"))
    assert p.bad == []
    html = _read("index.html")
    assert '<link rel="stylesheet" href="/ui/app.css">' in html
    assert '<script src="/ui/app.js" defer></script>' in html
    assert {
        "rows-table",
        "serving-box",
        "confirm-panel",
        "job-panel",
        "error-area",
        "token-form",
    } <= p.ids


# --- the `Dashboard:` line printed by `llmctl serve` -------------------------


@pytest.fixture
def serve_settings(monkeypatch) -> Settings:
    s = Settings()
    monkeypatch.setattr(cli, "load_settings", lambda: s)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
    return s


def _serve(*args: str):
    return CliRunner().invoke(cli.app, ["serve", *args])


def test_dashboard_line_explicit(serve_settings) -> None:
    res = _serve("--host", "127.0.0.1", "--port", "9999")
    assert "Dashboard: http://127.0.0.1:9999/ui" in res.output


def test_dashboard_line_default(serve_settings) -> None:
    res = _serve()
    assert f"Dashboard: http://127.0.0.1:{serve_settings.api.port}/ui" in res.output
    assert serve_settings.api.port == 8088


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_dashboard_line_wildcard_hosts(serve_settings, host) -> None:
    serve_settings.api.allow_public_bind = True
    serve_settings.api.auth_token = "s3cret"
    res = _serve("--host", host, "--port", "7000")
    assert "Dashboard: http://127.0.0.1:7000/ui" in res.output


def test_dashboard_line_brackets_ipv6(serve_settings) -> None:
    res = _serve("--host", "::1", "--port", "7001")
    assert "Dashboard: http://[::1]:7001/ui" in res.output


def test_no_dashboard_line_when_refused(serve_settings) -> None:
    res = _serve("--host", "0.0.0.0")
    assert res.exit_code == 2
    assert "Dashboard:" not in res.output
