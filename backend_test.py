"""Backend API testing for LLM Mission Control.

Tests all API endpoints to ensure they work correctly with the fixed
sqlite:///:memory: connection pooling.
"""

import sys

from fastapi.testclient import TestClient

from llmctl.api.app import create_app
from llmctl.db import init_db


class BackendAPITester:
    """Test backend API endpoints."""

    def __init__(self):
        """Initialize test client with in-memory database."""
        self.db_url = "sqlite:///:memory:"
        init_db(self.db_url)
        self.client = TestClient(create_app(database_url=self.db_url))
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name: str, test_func):
        """Run a single test and track results."""
        self.tests_run += 1
        print(f"\nTesting {name}...")
        try:
            test_func()
            self.tests_passed += 1
            print("Passed")
            return True
        except AssertionError as e:
            print(f"Failed: {e}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_health_endpoint(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["state"] == "ok", f"Expected state=ok, got {data['state']}"
        assert data["safe_mode"] is True, "Expected safe_mode=True"

    def test_models_crud_flow(self):
        """Test complete model CRUD flow."""
        # List empty models
        response = self.client.get("/models")
        assert response.status_code == 200
        assert response.json() == []

        # Create model
        model_data = {
            "name": "test-llama-2-7b",
            "runtime": "vllm",
            "source": "meta-llama/Llama-2-7b-hf",
            "tags": ["test", "llama"]
        }
        response = self.client.post("/models", json=model_data)
        assert response.status_code == 201
        model = response.json()
        assert model["name"] == "test-llama-2-7b"
        assert model["runtime"] == "vllm"
        assert "id" in model
        model_id = model["id"]

        # List models
        response = self.client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert len(models) == 1
        assert models[0]["id"] == model_id

        # Delete model
        response = self.client.delete(f"/models/{model_id}")
        assert response.status_code == 204

    def test_session_workflow(self):
        """Test session start/stop/restart workflow."""
        # Create a model first
        model_data = {"name": "test-model", "runtime": "ollama"}
        model_response = self.client.post("/models", json=model_data)
        assert model_response.status_code == 201
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
        sessions = response.json()
        assert len(sessions) >= 1

        # Stop session
        response = self.client.post(f"/sessions/{session_id}/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

        # Restart session
        response = self.client.post(f"/sessions/{session_id}/restart")
        assert response.status_code == 200
        assert response.json()["status"] == "planned"

    def test_gpu_endpoint(self):
        """Test GPU telemetry endpoint."""
        response = self.client.get("/gpus")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_benchmark_workflow(self):
        """Test benchmark creation and listing."""
        # List benchmarks
        response = self.client.get("/benchmarks")
        assert response.status_code == 200

        # Run benchmark (dry-run)
        benchmark_data = {
            "name": "test-benchmark",
            "dry_run": True
        }
        response = self.client.post("/benchmarks/run", json=benchmark_data)
        assert response.status_code == 201
        benchmark = response.json()
        assert benchmark["name"] == "test-benchmark"
        assert benchmark["success"] is True

    def test_model_session_benchmark_integration(self):
        """Test complete integration flow: model -> session -> benchmark."""
        # Create model
        model_data = {
            "name": "integration-test-model",
            "runtime": "vllm",
            "source": "test-source"
        }
        model_response = self.client.post("/models", json=model_data)
        assert model_response.status_code == 201
        model_id = model_response.json()["id"]

        # Start session
        session_data = {
            "model_id": model_id,
            "runtime": "vllm",
            "dry_run": True
        }
        session_response = self.client.post("/sessions/start", json=session_data)
        assert session_response.status_code == 201
        session_id = session_response.json()["id"]

        # Run benchmark
        benchmark_data = {
            "name": "integration-benchmark",
            "model_id": model_id,
            "session_id": session_id,
            "dry_run": True
        }
        benchmark_response = self.client.post("/benchmarks/run", json=benchmark_data)
        assert benchmark_response.status_code == 201
        benchmark = benchmark_response.json()
        assert benchmark["model_id"] == model_id
        assert benchmark["session_id"] == session_id

    def test_safety_constraints(self):
        """Test that safety constraints are enforced."""
        # Health should report safe mode
        response = self.client.get("/health")
        assert response.json()["safe_mode"] is True

        # Sessions should be dry-run by default
        model_response = self.client.post("/models", json={"name": "test", "runtime": "vllm"})
        model_id = model_response.json()["id"]

        session_response = self.client.post(
            "/sessions/start",
            json={"model_id": model_id, "runtime": "vllm", "dry_run": True}
        )
        session = session_response.json()
        assert session["status"] == "planned"
        assert session["pid"] is None
        assert session["launch_plan"]["dry_run"] is True

    def run_all_tests(self):
        """Run all backend tests."""
        print("=" * 60)
        print("Backend API Testing - LLM Mission Control")
        print("=" * 60)

        self.run_test("Health Endpoint", self.test_health_endpoint)
        self.run_test("Models CRUD Flow", self.test_models_crud_flow)
        self.run_test("Session Workflow", self.test_session_workflow)
        self.run_test("GPU Endpoint", self.test_gpu_endpoint)
        self.run_test("Benchmark Workflow", self.test_benchmark_workflow)
        self.run_test(
            "Model-Session-Benchmark Integration",
            self.test_model_session_benchmark_integration,
        )
        self.run_test("Safety Constraints", self.test_safety_constraints)

        print("\n" + "=" * 60)
        print(f"Tests passed: {self.tests_passed}/{self.tests_run}")
        print("=" * 60)

        return 0 if self.tests_passed == self.tests_run else 1


def main():
    """Run backend tests."""
    tester = BackendAPITester()
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
