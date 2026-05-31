# LLM Mission Control

LLM Mission Control is a **local-first Linux power-user control plane** for discovering, organizing, launching, monitoring, and eventually routing requests across local LLM runtimes such as **vLLM**, **llama.cpp**, **LM Studio**, **Ollama**, and arbitrary Python launch scripts.

This repository has progressed beyond scaffold: **model discovery, GPU/process telemetry, and real process supervision are implemented**. Destructive actions (process launch/kill) are gated behind an explicit `dry_run=False` flag; the default behavior remains safe (dry-run planning).

## Goals

- Manage local LLM runtimes from one CLI/API/TUI surface.
- Support NVIDIA GPU telemetry and dual-GPU placement planning.
- Keep safety first: no destructive operations or arbitrary process launches without explicit confirmation in future phases.
- Use a modular adapter/service architecture so runtimes can evolve independently.
- Run locally on Linux workstations and optionally be reached over a private network such as Tailscale.

## Architecture

```text
configs/                Example YAML configuration
llmctl/config.py        Config discovery/loading and typed settings
llmctl/db.py            SQLite + SQLModel database schema/session helpers
llmctl/schemas.py       Pydantic API/service contracts
llmctl/adapters/        RuntimeAdapter interface and runtime skeletons
llmctl/telemetry/       GPU and process telemetry helpers
llmctl/services/        Registry, sessions, scheduler, benchmarks, health services
llmctl/api/             FastAPI application and route modules
llmctl/tui/             Textual TUI application and screens
llmctl/cli.py           Typer command-line interface
tests/                  Smoke tests for install/import/API/CLI/DB
```

### Data model

The SQLite schema includes tables for:

- `models`: runtime, identifier, path/source, quantization, tags, metadata
- `sessions`: lifecycle and resource information for active/previous model processes
- `profiles`: launch presets and default runtime parameters
- `benchmarks`: benchmark result history
- `events`: audit/event/log records

### Runtime adapter contract

Every runtime adapter implements:

- `discover_models()`
- `start()`
- `stop()`
- `status()`
- `health_check()`
- `delete_model()`

Concrete adapters provide **real** integrations with graceful fallback when a
runtime/binary/GPU is absent:

- **Ollama** & **LM Studio** — HTTP adapters (`httpx`). Discovery via `GET /api/tags`
  and `GET /v1/models`; health via `GET /api/version` / `GET /v1/models`.
- **vLLM** & **llama.cpp** — process-launch adapters. Filesystem discovery
  (`config.json` / `*.gguf`) plus real subprocess supervision.
- **Python script** — launches an arbitrary Python entry point as a supervised
  child process.

Process-launch adapters delegate to a shared `ProcessSupervisor` that launches
detached process groups, captures logs, and terminates gracefully (SIGTERM →
SIGKILL). All discovery/health calls degrade to empty/`UNAVAILABLE` results when
the backing runtime is not present.

## Installation

```bash
cd llm-mission-control
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run smoke tests:

```bash
pip install -e '.[dev]'
pytest
```

## CLI usage

```bash
llmctl --help
llmctl scan                      # discover models across all runtimes
llmctl add-model --name my-llm --runtime vllm --source org/Model-7B
llmctl models
llmctl profiles                  # list launch profiles (fast, coding, reasoning, ...)
llmctl gpus                      # GPU telemetry table (VRAM, util, temp, power, procs)
llmctl health                    # overall + per-runtime health
llmctl start MODEL_ID --profile fast --gpus auto      # auto-pick most-free GPU(s)
llmctl start MODEL_ID --profile tutoring --cpu        # CPU-only (hide GPUs)
llmctl start MODEL_ID --gpus 0,1                       # explicit GPUs
llmctl start MODEL_ID --dry-run                        # plan only, no launch
llmctl sessions                  # list sessions (auto-reconciles dead ones)
llmctl logs SESSION_ID --lines 100   # tail a session's log file
llmctl logs                          # recent event/audit log
llmctl stop SESSION_ID
llmctl restart SESSION_ID
llmctl serve --host 127.0.0.1 --port 8088
llmctl tui
```

## API skeleton

The FastAPI app is exposed by `llmctl.api.app:create_app` and includes:

- `GET /health`
- `GET /models`
- `POST /models`
- `DELETE /models/{id}`
- `GET /sessions`
- `POST /sessions/start`
- `POST /sessions/{id}/stop`
- `POST /sessions/{id}/restart`
- `GET /gpus`
- `GET /benchmarks`
- `POST /benchmarks/run`

## Configuration

Example files live in `configs/`:

- `model_dirs.yaml`: model discovery roots and runtime hints
- `profiles.yaml`: reusable runtime launch profiles. Ships with defaults:
  `fast`, `coding`, `reasoning`, `long-context` (vLLM), `tutoring`, `quant`
  (llama.cpp), and `adtech` (vLLM). Profiles are synced into the database and
  referenced by name via `llmctl start MODEL_ID --profile NAME`.
- `settings.yaml`: database, telemetry, API, scheduler, per-runtime, and safety settings

At runtime, configuration can be loaded from a chosen path or from the default user config location. Machine-specific paths should be supplied by the operator through config files or environment variables instead of hardcoding them in source.

Relevant environment variables:

- `LLMCTL_CONFIG_DIR`: override configuration directory
- `LLMCTL_DB_URL`: override SQLite database URL
- `LLMCTL_LOG_LEVEL`: override log level

## Safety model

The control plane is conservative by default:

- Session starts default to `dry_run=True`: a `PLANNED` session is recorded and
  **no process is launched**. Real process control happens only with an explicit
  `--no-dry-run` (CLI) / `dry_run=false` (API).
- The `ProcessSupervisor` terminates only the process groups it is told to stop,
  escalating SIGTERM → SIGKILL with a timeout.
- Every lifecycle action (plan/start/stop/restart) writes an `EventRecord` for
  a complete local audit trail (`llmctl logs`).
- All runtime discovery/health calls fail closed (empty / `UNAVAILABLE`) when a
  runtime, binary, or GPU is unavailable.
- Remote access should bind to localhost by default; expose only over a trusted
  private network.

## Local and Tailscale usage notes

Recommended future deployment mode:

1. Keep the API bound to `127.0.0.1` by default.
2. Use Tailscale SSH or a Tailscale Funnel/Serve configuration only when intentionally exposing the control panel.
3. Prefer read-only dashboards remotely; require local confirmation for model deletion or process termination.
4. Store runtime logs and event history locally; avoid sending model paths or prompts to third-party services.

## Planned phases

### Phase 1: Scaffold and contracts — DONE

- Package layout, config loading, database schema, Pydantic contracts,
  adapter/service interfaces, CLI/API/TUI skeletons, smoke tests.

### Phase 2: Runtime discovery — DONE

- Safe model directory scanning (`discovery.py`)
- Ollama/LM Studio HTTP discovery; vLLM/llama.cpp filesystem discovery
- GPU telemetry snapshots (pynvml) with non-NVIDIA fallback

### Phase 3: Controlled launching — DONE

- Scheduler-built launch plans (command/env/port/GPU placement)
- Session state transitions with event auditing
- Real process supervision (SIGTERM→SIGKILL) gated by `dry_run`

### Phase 4: Monitoring and benchmarks — IN PROGRESS

- Live Textual dashboard data binding (pending)
- Benchmark runners against live endpoints (pending)
- Per-runtime health checks (DONE) and event/log viewer (DONE via `llmctl logs`)

### Phase 5: Routing and automation — PLANNED
