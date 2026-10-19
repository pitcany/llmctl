from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_changelog_mentions_swap_routes():
    text = (ROOT / "CHANGELOG.md").read_text().split("\n## ")[1]
    for route in ("GET /swap", "POST /swap/run", "GET /swap/jobs/{id}", "POST /swap/stop"):
        assert route in text


def test_guide_swap_section():
    text = (ROOT / "docs/USER-GUIDE.md").read_text()
    section = text.split("### HTTP API: swap")[1].split("\n## ")[0]
    assert section.count("curl ") >= 4
    assert "lost when" in section and "poll" in section
    assert "Bearer" in section
