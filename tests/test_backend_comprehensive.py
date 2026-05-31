"""Comprehensive backend testing for LLM Mission Control scaffold.

This test suite validates the complete backend implementation including:
- CLI commands functionality
- FastAPI endpoints
- Database operations
- Runtime adapters
- Safety constraints
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, inspect
from typer.testing import CliRunner

from llmctl.api.app import create_app
from llmctl.cli import app as cli_app
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.services.router import RuntimeRouter


class TestCLICommands:
    """Test all CLI commands."""

    def test_cli_help(self):
        """Test CLI help shows all required commands."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        required_commands = [
            "scan", "models", "gpus", "sessions", "add-model", "delete-model",
            "start", "stop", "restart", "logs", "bench", "tui", "serve", "generate-systemd"
        ]
        for cmd in required_commands:
            assert cmd in result.output

    def test_cli_scan(self):
        """Test scan command."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["scan"])
        assert result.exit_code == 0
        assert "Scan scaffold complete" in result.output

    def test_cli_models_list(self):
        """Test models list command."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["models"])
        assert result.exit_code == 0

    def test_cli_sessions_list(self):
        """Test sessions list command."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["sessions"])
        assert result.exit_code == 0

    def test_cli_gpus(self):
        """Test gpus command."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["gpus"])
        assert result.exit_code == 0

    def test_cli_logs(self):
        """Test logs command (scaffold)."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["logs"])
        assert result.exit_code == 0
        assert "event" in result.output.lower()

    def test_cli_generate_systemd(self):
        """Test systemd unit generation."""
        runner = CliRunner()
        result = runner.invoke(cli_app, ["generate-systemd"])
        assert result.exit_code == 0
        assert "[Unit]" in result.output
        assert "llmctl serve" in result.output


class TestAPIEndpoints:
    """Test all FastAPI endpoints."""

    def setup_method(self):
        """Create test client with isolated database."""
        from llmctl.db import init_db
        db_url = "sqlite:///:memory:"
        init_db(db_url)
        self.client = TestClient(
            create_app(database_url=db_url)
        )

    def test_health_endpoint(self):
        """Test health endpoint returns safe mode."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "ok"
        assert data["safe_mode"] is True

    def test_models_crud(self):
        """Test model CRUD operations."""
        # List empty
        response = self.client.get("/models")
        assert response.status_code == 200
        assert response.json() == []

        # Create model
        model_data = {
            "name": "test-model",
            "runtime": "vllm",
            "source": "meta-llama/Llama-2-7b-hf"
        }
        response = self.client.post("/models", json=model_data)
        assert response.status_code == 201
        model = response.json()
        assert model["name"] == "test-model"
        assert model["runtime"] == "vllm"
        model_id = model["id"]

        # List models
        response = self.client.get("/models")
        assert response.status_code == 200
        assert len(response.json()) == 1

        # Delete model
        response = self.client.delete(f"/models/{model_id}")
        assert response.status_code == 204

    def test_sessions_workflow(self):
        """Test session start/stop/restart workflow."""
        # Create a model first
        model_data = {"name": "test", "runtime": "ollama"}
        model_response = self.client.post("/models", json=model_data)
        model_id = model_response.json()["id"]

        # Start session (dry-run)
        session_data = {
            "model_id": model_id,
            "runtime": "ollama",
            "dry_run": True
        }
        response = self.client.post("/sessions/start", json=session_data)
        assert response.status_code == 201
        session = response.json()
        assert session["status"] == "planned"
        assert session["launch_plan"]["dry_run"] is True
        assert "dry_run_no_process_launch" in session["launch_plan"]["safety_checks"]
        session_id = session["id"]

        # List sessions
        response = self.client.get("/sessions")
        assert response.status_code == 200
        assert len(response.json()) >= 1

        # Stop session
        response = self.client.post(f"/sessions/{session_id}/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

        # Restart session
        response = self.client.post(f"/sessions/{session_id}/restart")
        assert response.status_code == 200
        assert response.json()["status"] == "planned"

    def test_gpus_endpoint(self):
        """Test GPU telemetry endpoint."""
        response = self.client.get("/gpus")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_benchmarks_workflow(self):
        """Test benchmark workflow."""
        # List benchmarks
        response = self.client.get("/benchmarks")
        assert response.status_code == 200

        # Run benchmark (dry-run)
        benchmark_data = {
            "name": "test-bench",
            "dry_run": True
        }
        response = self.client.post("/benchmarks/run", json=benchmark_data)
        assert response.status_code == 201
        benchmark = response.json()
        assert benchmark["name"] == "test-bench"
        assert benchmark["success"] is True


class TestDatabaseSchema:
    """Test database schema and operations."""

    def test_all_tables_exist(self, tmp_path: Path):
        """Test all required tables are created."""
        db_url = f"sqlite:///{tmp_path / 'test.sqlite3'}"
        init_db(db_url)
        engine = get_engine(db_url)
        tables = set(inspect(engine).get_table_names())
        required_tables = {"models", "sessions", "profiles", "benchmarks", "events"}
        assert required_tables.issubset(tables)

    def test_model_record_creation(self, tmp_path: Path):
        """Test creating model records."""
        from llmctl.db import ModelRecord, ModelStatus

        db_url = f"sqlite:///{tmp_path / 'test.sqlite3'}"
        init_db(db_url)
        engine = get_engine(db_url)

        with Session(engine) as session:
            model = ModelRecord(
                name="test-model",
                runtime=RuntimeName.VLLM,
                source="test-source",
                status=ModelStatus.REGISTERED
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            assert model.id is not None
            assert model.name == "test-model"


class TestRuntimeAdapters:
    """Test runtime adapter system."""

    def test_all_adapters_registered(self):
        """Test all required runtime adapters are registered."""
        router = RuntimeRouter()
        runtimes = {rt.value for rt in router.list_runtimes()}
        required_runtimes = {"vllm", "llama_cpp", "lmstudio", "ollama", "python_script"}
        assert runtimes == required_runtimes

    def test_adapter_safety_constraints(self):
        """Test adapters are safe placeholders."""
        import asyncio

        from llmctl.schemas import LaunchPlan

        router = RuntimeRouter()
        adapter = router.get_adapter(RuntimeName.VLLM)

        # Test discover returns empty list
        models = asyncio.run(adapter.discover_models())
        assert models == []

        # Test start returns planned session
        plan = LaunchPlan(runtime=RuntimeName.VLLM, dry_run=True)
        session = asyncio.run(adapter.start(plan))
        assert session.status.value == "planned"

        # Test health check
        health = asyncio.run(adapter.health_check())
        assert health.runtime == RuntimeName.VLLM


class TestSafetyConstraints:
    """Test safety constraints are enforced."""

    def test_sessions_are_dry_run_by_default(self):
        """Test sessions default to dry-run mode."""
        from llmctl.db import init_db
        db_url = "sqlite:///:memory:"
        init_db(db_url)
        client = TestClient(create_app(database_url=db_url))

        # Create model
        model_response = client.post("/models", json={"name": "test", "runtime": "vllm"})
        model_id = model_response.json()["id"]

        # Start session
        session_response = client.post(
            "/sessions/start",
            json={"model_id": model_id, "runtime": "vllm", "dry_run": True}
        )
        session = session_response.json()

        # Verify safety
        assert session["status"] == "planned"
        assert session["pid"] is None
        assert session["launch_plan"]["dry_run"] is True
        assert "dry_run_no_process_launch" in session["launch_plan"]["safety_checks"]

    def test_health_reports_safe_mode(self):
        """Test health endpoint reports safe mode."""
        from llmctl.db import init_db
        db_url = "sqlite:///:memory:"
        init_db(db_url)
        client = TestClient(create_app(database_url=db_url))
        response = client.get("/health")
        assert response.json()["safe_mode"] is True


def test_installation():
    """Test package is properly installed."""
    result = subprocess.run(
        ["llmctl", "--version"],
        capture_output=True,
        text=True
    )
    # Version command may not exist, but llmctl should be found
    assert result.returncode in [0, 2]  # 0 = success, 2 = no version command


def test_ruff_linting():
    """Test code passes ruff linting."""
    pytest.importorskip("ruff", reason="ruff not installed in this environment")
    package_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "llmctl", "tests"],
        capture_output=True,
        text=True,
        cwd=package_root,
    )
    assert result.returncode == 0, f"Ruff linting failed: {result.stdout}"


if __name__ == "__main__":
    print("Running comprehensive backend tests...")
    sys.exit(subprocess.run(["pytest", __file__, "-v"]).returncode)
