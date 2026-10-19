from pathlib import Path

APP = Path(__file__).parent.parent / "llmctl" / "api" / "static" / "app.js"


def test_job_id_cleared_when_polling_stops():
    src = APP.read_text()
    assert src.count("jobId = null;") >= 3  # init + completion + 404
    assert "lastDoneId" in src
    assert "refreshQueued" in src and "jobBusy" in src


def test_line_cap():
    assert len(APP.read_text().splitlines()) <= 450
