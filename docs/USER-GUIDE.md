# llmctl User Guide

Operational reference for everything `llmctl` does. For 5-minute
install + first command, see [QUICKSTART.md](./QUICKSTART.md).

## Table of contents

1. [Concepts](#concepts)
2. [CLI reference](#cli-reference)
3. [TUI reference](#tui-reference)
4. [Configuration](#configuration)
5. [Integrations](#integrations)
6. [Presets](#presets)
7. [Slots](#slots)
8. [Troubleshooting](#troubleshooting)
9. [Internals — how it works under the hood](#internals)

---

## Concepts

llmctl has four primary objects. Understanding them up front makes
the rest of the docs make sense.

### Preset

A YAML file at `~/.config/llmctl/presets/<alias>.yaml` that describes
**what to serve**: the HuggingFace model id, quantization, max
context length, tool/reasoning parser, etc. Schema is owned by
`llmctl.presets`. Existing legacy presets under
`~/.config/llm-models/` are picked up through the one-shot symlink
fallback.

A preset is data — it doesn't say where to run, just what.

### Managed unit

An externally-installed systemd `.service` file that llmctl owns the
**how** for. The default config knows about three:

- `vllm-tp` (port 8003, dual-GPU TP=2, daily-driver TP fleet)
- `vllm-coder` (port 8001, GPU 0, slot)
- `vllm-reasoner` (port 8002, GPU 1, slot)

llmctl never installs these units itself — they must already exist
on your box. The reference unit files used in development live in
the upstream `~/AI/services/` directory; you can adapt them or write
your own. llmctl just writes their `EnvironmentFile`, calls
`sudo systemctl restart <unit>`, and polls `/v1/models` for readiness.

### Slot

A logical "named GPU lane" — `coder` on GPU 0, `reasoner` on GPU 1.
A slot's identity (name, GPU, port) is **stable**; the underlying
model can swap. Downstream clients that talk to `coder` keep
working when you change which model is loaded there.

Slots are TP=1 single-GPU. They share unit names with the
"managed unit" of the same name (`vllm-coder` runs the `coder`
slot), but the conceptual distinction matters: a slot pins served
name to its identity, not to the preset's `served_name`.

### Launch spec / orchestrator

When you run `llmctl vllm <preset>` or `llmctl slot <name> <preset>`,
the orchestrator:

1. Loads the preset via the canonical preset store
2. Applies any CLI overrides (`--tq`, `--no-tq`)
3. Builds a `VLLMLaunchSpec` (Pydantic, validated)
4. Runs the fleet preflight (stop competing units)
5. Runs the Harbor preflight (stop `harbor.ollama` container)
6. Renders the env file, writes it, restarts the unit
7. Polls `/v1/models` until ready (or times out at 5 min)
8. Verifies the Hermes provider URL still matches

Steps 4, 5, and 8 are optional and can be disabled in code; from the
CLI they're always on.

---

## CLI reference

### Daily-driver commands

| Command | What |
|---------|------|
| `llmctl presets` | List presets from `~/.config/llmctl/presets/` |
| `llmctl vllm <preset>` | Start TP-fleet unit on preset |
| `llmctl slot <name> <preset>` | Apply preset to slot (`coder` / `reasoner`) |
| `llmctl status` | Managed units + slots with resolved env paths and ports |
| `llmctl health` | Per-runtime health rollup (vLLM, llama.cpp, LM Studio, Ollama) |

### Common flags

Used by `vllm` and `slot`:

| Flag | Effect |
|------|--------|
| `--dry-run` | Render the env file + print planned actions, change nothing |
| `--no-wait` | Skip polling `/v1/models` after restart (returns immediately) |
| `--tq` | Force `--kv-cache-dtype turboquant_k8v4` regardless of preset |
| `--no-tq` | Force TurboQuant off (omit `VLLM_KV_DTYPE` entirely) |

`--tq` and `--no-tq` are mutually exclusive.

### Observability

| Command | What |
|---------|------|
| `llmctl gpus` | NVML telemetry: VRAM, util, temp, power, processes |
| `llmctl doctor` | Backend binary diagnostics + GPU + scheduler config |
| `llmctl models` | Registered model rows (Ollama tags + filesystem scans + live vLLM probe) |
| `llmctl validate` | Find drift between the records and the disk/network (exits 1 on findings) |
| `llmctl logs [SESSION_ID]` | Tail a session log, or show recent events |

`llmctl validate` does four read-only checks:

| Check | What it finds |
|-------|---------------|
| `preset-model-missing` | A preset `model_id` path that is not on the disk. A symlink that does not resolve gives the same result. |
| `registry-path-missing` | A registry row with a `path` that is not on the disk. |
| `broken-symlink` | A symlink in a model root from `model_dirs.yaml` that does not resolve. Only this check finds orphans. |
| `port-drift` | A managed unit that systemd reports as active, but that serves nothing on its registered port. |

The checks skip values that are Hugging Face repository ids
(`org/model`). Such a value is a valid `model_id`, but it is not a
statement about the disk.

The `port-drift` check gives no result if the host has no `systemctl`,
or if the unit is not active. A unit that is stopped is not drift.

### Sessions and scheduler

These belong to the original scaffold and aren't part of the daily
flow; useful for debugging.

| Command | What |
|---------|------|
| `llmctl sessions` | List launched sessions |
| `llmctl scan` | Refresh discovery cache (vLLM HTTP probe + filesystem + Ollama API) |
| `llmctl start MODEL_ID --profile NAME` | Scheduler-based launch (subprocess, not systemd) |
| `llmctl stop SESSION_ID` | Mark a session stopped |
| `llmctl restart SESSION_ID` | Plan a restart |
| `llmctl plan MODEL_ID` | Print a launch plan without executing |
| `llmctl cleanup [--remove-stale]` | Free ports + purge dead sessions |
| `llmctl add-model`, `delete-model` | Manual model registry CRUD |

### Other surface

| Command | What |
|---------|------|
| `llmctl tui` | Launch the Textual TUI (see [TUI reference](#tui-reference)) |
| `llmctl serve` | Run the FastAPI control plane (`:8088` by default) |
| `llmctl bench MODEL` | Streaming-TTFT benchmark runner |
| `llmctl profiles` | List orchestration profiles (different from presets!) |
| `llmctl generate-systemd`, `install-systemd` | Generate / install systemd units for the API itself |

---

## TUI reference

```bash
llmctl tui
```

### Keybindings (any screen)

| Key | Screen |
|-----|--------|
| `d` | Dashboard |
| `p` | Presets |
| `u` | Units |
| `m` | Models |
| `s` | Sessions |
| `g` | GPUs |
| `l` | Logs |
| `o` | Doctor |
| `b` | Benchmarks |
| `r` | Refresh active screen |
| `q` | Quit |

Auto-refresh fires every 3s on the active screen.

### Presets screen (`p`)

The daily driver. One row per preset. Enter on a row opens a
launch picker:

| Key in picker | Target |
|----------------|--------|
| `t` | TP fleet (vllm-tp, both GPUs) |
| `c` | Coder slot (GPU 0) |
| `r` | Reasoner slot (GPU 1) |
| `esc` | Cancel |

Picking a target runs the orchestrator in a worker thread.
Notifications surface launch start, success (with port), and
failure (with reason).

### Units screen (`u`)

Live status. One row per managed unit + one row per slot. Columns:

- **Role** — `unit vllm-tp`, `slot coder`, etc.
- **Unit** — bare systemd unit name
- **Active** — `active` (green) or `inactive` (muted)
- **Port** — what llmctl thinks the unit listens on
- **Served (live)** — model IDs from `/v1/models` probe; `starting?` (yellow) when active-but-empty
- **Env file** — resolved path to the `EnvironmentFile`

Probes use a 1.5s timeout each; 5 down units add ≤8s to refresh.
`ctrl+r` triggers an immediate refresh.

### Models screen (`m`)

The registry view. Shows everything `llmctl models` shows. Enter
on a row opens a launch plan modal (the original scheduler-based
flow, not the orchestrator).

### Other screens

- **Dashboard (`d`)** — overview cards: model count, sessions, profiles, GPUs, runtime health
- **Sessions (`s`)** — launched sessions with status; `x` to stop, `ctrl+r` to restart
- **GPUs (`g`)** — live NVML
- **Logs (`l`)** — event audit trail with severity color
- **Doctor (`o`)** — backend binary table; `c` copies install command for missing backends
- **Benchmarks (`b`)** — history with delta columns; baseline support; enter to rerun

---

## Configuration

Config files live in `~/.config/llmctl/` (XDG-respected). Override
the location with `$LLMCTL_CONFIG_DIR`.

### `settings.yaml`

The complete schema with defaults that match yannik-desktop's
production posture:

```yaml
app:
  log_level: INFO
  safe_mode: true

database:
  url: null  # auto: sqlite:///<data-dir>/llmctl.sqlite3

api:
  host: 127.0.0.1
  port: 8088

vllm:
  defaults:
    gpus: "0,1"
    tensor_parallel: 2
    port: 8003
    host: "0.0.0.0"
    max_model_len: 32768
    gpu_memory_utilization: 0.85
    prefix_cache: true
    chunked_prefill: true
    nccl_p2p_disable: false

managed_units:
  vllm_tp:
    enabled: false               # opt-in only; orchestrator commands ignore this
    unit_name: vllm-tp
    env_file_path: null          # auto: $AI_HOME/services/vllm-tp.env -> ~/AI/services/vllm-tp.env
    launcher_marker: vllm-launcher.sh   # set to null to disable legacy-unit guard
    default_port: 8003
  vllm_coder:   { unit_name: vllm-coder,    default_port: 8001 }
  vllm_reasoner: { unit_name: vllm-reasoner, default_port: 8002 }

  slots:
    coder:    { gpu: "0", port: 8001, unit_name: vllm-coder }
    reasoner: { gpu: "1", port: 8002, unit_name: vllm-reasoner }

  fleet:
    tp: vllm-tp
    coder: vllm-coder
    reasoner: vllm-reasoner
    ollama: ollama
    fleet_target: agents.target

scheduler:
  default_host: 127.0.0.1
  port_ranges:
    vllm: [8000, 8099]
    llama_cpp: [8100, 8199]
    python_script: [8200, 8299]
  safety_margin_gb: 1.0
```

You only need entries for things you want to override. A completely
empty file works.

### Environment variables

| Var | Effect |
|-----|--------|
| `LLMCTL_CONFIG_DIR` | Override config dir (default: `~/.config/llmctl/`) |
| `LLMCTL_DB_URL` | Override the SQLite URL |
| `LLMCTL_LOG_LEVEL` | Override log level |
| `LLMCTL_PYTHON_ROOT` | Where the vLLM interpreter lives (for `LD_LIBRARY_PATH` / `PATH` in rendered env files) |
| `LLMCTL_CUDA_ROOT` | CUDA toolkit root (default `/usr/local/cuda`) |
| `LLMCTL_VLLM_ENV_FILE` | Direct override for the vllm-tp env file path |
| `LLMCTL_SLOT_CODER_ENV_FILE`, `LLMCTL_SLOT_REASONER_ENV_FILE` | Per-slot env file overrides |
| `AI_HOME` | Used to resolve `$AI_HOME/services/<unit>.env` when no explicit path is set |
| `HF_HOME` | HuggingFace cache (default `~/.cache/huggingface`) |
| `LLMCTL_QUIET_DEPRECATION` | Set to `1` to silence the `gpu-models` shim's deprecation hint |
| `LLMCTL_BIN` | Used by the `gpu-models` shim to locate `llmctl` (defaults to conda env's bin) |

### Env-file path resolution

For each managed unit, the env file path is resolved in this order:

1. Explicit `env_file_path` in `settings.yaml`
2. `$LLMCTL_VLLM_ENV_FILE` (or per-slot `$LLMCTL_SLOT_<UPPER>_ENV_FILE`)
3. `$AI_HOME/services/<unit_name>.env`
4. `~/AI/services/<unit_name>.env` (fallback)

---

## Integrations

llmctl exposes three optional integrations as lifecycle hooks. From
the CLI they're always on; from code you can selectively disable them
via `OrchestratorOptions`.

### Hermes verify (post-start)

After a successful restart, llmctl looks up the corresponding provider
in `~/.hermes/config.yaml`:

| Unit role | Hermes provider name |
|-----------|----------------------|
| `vllm-tp` | `vllm` |
| `vllm-coder` | `vllm-coder` |
| `vllm-reasoner` | `vllm-reasoner` |

The verify is **read-only**. It prints one of:

- `hermes: vllm -> http://127.0.0.1:8003/v1 (verified)` — all good
- `hermes: WARNING — vllm.base_url is X, expected Y` — drift; user fixes via `hermes config edit`
- `hermes: no 'vllm' provider in config.yaml` — user adds it manually
- `hermes: config not found` — Hermes not installed; no-op

Never auto-mutates the user's config (would clobber hand-tuned
fallback providers / model defaults).

### Harbor preflight (pre-start)

Before starting a GPU-claiming unit, stops the `harbor.ollama`
container if running. This frees GPU memory so vLLM's init doesn't
OOM on a partially-occupied GPU. No-op when Docker isn't installed
or the container isn't running.

### Fleet preflight (pre-start)

Stops competing systemd units before starting the target. Order
matters — the fleet target is stopped before the slot services it
gates to avoid systemd `Wants=` restart loops.

| Starting | Stops (in order) |
|----------|------------------|
| `vllm-tp` (TP fleet) | `agents.target`, `vllm-coder`, `vllm-reasoner`, `ollama`, `vllm-tp` |
| `vllm-coder` (slot) | `vllm-tp`, `ollama` |
| `vllm-reasoner` (slot) | `vllm-tp`, `ollama` |

Slot starts intentionally do **not** stop the sibling slot.

---

## Presets

### Where they live

`~/.config/llmctl/presets/<alias>.yaml`. The location is XDG-respected
via `$XDG_CONFIG_HOME`. On first load, llmctl symlinks legacy presets
from `~/.config/llm-models/` into the new directory if the new directory
does not exist yet.

### Schema

```yaml
alias: llama-3.3-70b              # required, used as the CLI argument
served_name: llama-3.3-70b        # required, advertised in /v1/models
model_id: casperhansen/llama-3.3-70b-instruct-awq    # required, HF id or local path
quantization: awq                 # one of: awq, gptq, fp8, bnb, none, gguf, compressed-tensors
vllm_quantization_flag: awq_marlin  # what gets passed to vLLM's --quantization
tensor_parallel_size: 2           # 1..8
max_model_len: 65536              # context window
max_num_seqs: 64                  # concurrency cap
gpu_memory_utilization: 0.85
kv_cache_dtype: fp8               # "auto" to omit; "turboquant_*" for TQ
tool_parser: llama3_json          # optional; nullable
reasoning_parser: deepseek_r1     # optional; folded into VLLM_EXTRA as --reasoning-parser
host: 0.0.0.0
port: 8000                        # ignored by llmctl (managed unit pins port)
trust_remote_code: false
schema_version: 1
```

### Editing presets

There's no `llmctl preset add` / `edit` / `delete` yet. Just edit
the YAML file directly:

```bash
$EDITOR ~/.config/llmctl/presets/<alias>.yaml
llmctl presets   # confirm it picked up
```

To see the full set of fields, look at any existing preset on the
box.

### How presets become env files

```
~/.config/llmctl/presets/<alias>.yaml
    ↓ llmctl.presets.load_all()
Model (canonical schema)
    ↓ model_to_launch_spec(model, defaults, port_override)
VLLMLaunchSpec (Pydantic, validated)
    ↓ apply_to_spec_dict(spec, override=tq_override)   # CLI --tq/--no-tq flags
VLLMLaunchSpec (final)
    ↓ render_vllm_env(spec)   OR   render_slot_env(spec, slot)
services/<unit>.env (systemd EnvironmentFile body)
    ↓ systemd reads on next restart
vllm-launcher.sh    ↓
vllm.entrypoints.openai.api_server with the right args
```

The two render functions (`render_vllm_env` for the TP fleet,
`render_slot_env` for slots) produce env file output that is
**byte-identical** to the output `gpu-models` used to produce. The
parity is locked in by 14 fixture files at
`tests/fixtures/env_renders/`.

---

## Slots

### What they are

A slot is a stable serving identity tied to one GPU. The
production fleet has two:

| Slot | GPU | Port | Unit | Hermes provider |
|------|-----|------|------|-----------------|
| `coder` | 0 | 8001 | `vllm-coder.service` | `vllm-coder` |
| `reasoner` | 1 | 8002 | `vllm-reasoner.service` | `vllm-reasoner` |

### Why they exist

Clients (Hermes, Open WebUI, llama-tools) pin their config to
served names like `coder`. If you swap the underlying model, the
clients break. Slots solve this by **decoupling**: the slot's served
name is its identity, the preset only contributes
model/quant/ctx.

### Slot vs TP fleet

| | TP fleet | Slot |
|--|----------|------|
| Unit | `vllm-tp` | `vllm-coder` / `vllm-reasoner` |
| GPUs | both (TP=2) | one (TP=1) |
| Port | 8003 | 8001 / 8002 |
| Served name | preset's `served_name` | slot identity (`coder` / `reasoner`) |
| Concurrent with the other thing? | No (mutually exclusive at systemd `Conflicts=` level) | Yes (slots coexist) |

### Slot safety

Legacy `gpu-models slot` refused 70B/72B/80B presets with a
`slot_eligible: false` heuristic. llmctl doesn't replicate this
yet — large presets that don't fit a single 32 GiB GPU at TP=1
will fail at vLLM init time, not at llmctl input validation.
Use `--dry-run` to render and inspect before applying.

Add a new slot by editing `settings.yaml`:

```yaml
managed_units:
  slots:
    vision:
      gpu: "2"
      port: 8004
      unit_name: vllm-vision
```

You'll also need to install `vllm-vision.service` separately
(llmctl doesn't generate slot units).

---

## Troubleshooting

### Health says vllm is unavailable but the unit IS running

The HTTP probe targets `http://localhost:<default_port>/v1/models`.
If your unit listens on a different port, override
`managed_units.vllm_tp.default_port` in `settings.yaml`.

Verify with: `curl http://localhost:8003/v1/models`. If that
returns a model list, llmctl should too — file a bug if not.

### Legacy unit refusal

```
LegacyUnitError: vllm-tp.service ExecStart does not contain 'vllm-launcher.sh'.
```

The unit installed in `/etc/systemd/system/vllm-tp.service` predates
the launcher-script ExecStart and speaks the older `VLLM_TP_*`
schema. llmctl writes the new `VLLM_*` schema, which the old
launcher would silently ignore.

Two options:

- Install a launcher-script-based unit (the upstream `~/AI` repo
  ships an example `apply-spec-rollout.sh` and a `vllm-launcher.sh`
  reference; replicate the pattern for your host)
- Disable the guard if you've installed a non-standard launcher:
  ```yaml
  managed_units:
    vllm_tp:
      launcher_marker: null
  ```

### Fleet preflight fails on a unit

Surfaced as:

```
fleet preflight failed on: vllm-coder
```

Means `sudo systemctl stop vllm-coder` returned non-zero. Most
common cause: the unit is in a failed state and needs
`sudo systemctl reset-failed vllm-coder` first.

### Cold start times out

The default readiness timeout is 300s. Qwen3-Next-80B with
CUDA-graph capture reliably needs ~3 min on 2x5090; if you're
loading anything bigger or doing more compilation work, bump the
timeout in code via `OrchestratorOptions(timeout_s=600)`. CLI flag
support for this is pending.

### Open WebUI says "model not found"

WebUI custom models in `Workspace → Models` pin to `base_model_id`.
After swapping vLLM, if no served name matches a pin, the pin
errors. Two fixes:

- Keep using the same `served_name` across swaps (use slots — they
  fix this structurally)
- Rebind the pin in the WebUI UI

---

## Internals

### Package structure

```
llmctl/
├── adapters/        # vLLM, llama.cpp, LM Studio, Ollama, python
│   ├── vllm.py              # HTTP-probe health + discovery (Phase A)
│   └── vllm_systemd.py      # VLLMSystemdAdapter (Phase 1)
├── integrations/    # external-system glue
│   ├── vllm_env.py          # render_vllm_env, render_slot_env (Phase 1+4)
│   ├── systemctl.py         # SystemctlRunner (Phase 1)
│   ├── hermes.py            # provider verify (Phase 3)
│   ├── harbor.py            # ollama-container stop (Phase 3)
│   ├── fleet.py             # preflight_stop (Phase 4)
│   └── turboquant.py        # --tq override (Phase 4)
├── services/
│   ├── preset_loader.py     # Model -> VLLMLaunchSpec (Phase 2)
│   └── vllm_orchestrator.py # start_vllm_tp, start_slot (Phase 5)
├── presets/         # schema + XDG-aware YAML loader (Phase 7c)
├── tui/             # Textual screens
│   ├── screens_presets.py   # Presets screen (Phase B)
│   ├── screens_units.py     # Units screen (Phase C)
│   └── _modals_presets.py   # TP/coder/reasoner picker (Phase B)
├── cli.py           # Typer command surface
├── config.py        # all settings schemas
└── ...

tests/               # see README for current test count
└── fixtures/env_renders/    # 14 byte-parity fixtures
```

### Test coverage

Notable suites:

- `test_vllm_env_render.py` + `test_vllm_env_parity.py` — render
  function unit tests + byte-parity vs frozen `gpu-models` fixtures
- `test_preset_loader.py` — end-to-end YAML → spec → env file
- `test_vllm_systemd_adapter.py` — full lifecycle with injected systemctl/clock/sleep/http_get
- `test_fleet_preflight.py`, `test_harbor_integration.py`, `test_hermes_integration.py`
- `test_vllm_orchestrator.py` — full lifecycle with stubbed deps
- `test_vllm_adapter_http_probe.py` — Phase A probe behavior
- `test_tui_presets.py`, `test_tui_units.py` — Textual pilot tests

Run: `uv run pytest -q` (≈45–55s for the CI-eligible subset; markers
`requires_gpu`, `requires_systemd`, `live_hf`, `bench_live` are skipped
in CI and need the right host to run).

### Where to look when debugging

| Symptom | Look at |
|---------|---------|
| Wrong env file content | `llmctl/integrations/vllm_env.py` |
| Wrong systemd interaction | `llmctl/integrations/systemctl.py`, `llmctl/adapters/vllm_systemd.py` |
| Preset not loading | `llmctl/presets/`, `llmctl/services/preset_loader.py` |
| Wrong port/path resolution | `llmctl/config.py` (`ManagedUnitConfig.resolve_env_file`) |
| Health probe wrong | `llmctl/adapters/vllm.py:_probe_unit` |
| Fleet preflight wrong order | `llmctl/integrations/fleet.py:units_to_stop` |
| TUI behaviour | `llmctl/tui/screens_*.py` + `_base.py` (DataScreen) |

### Migration history

This package replaced `gpu-models` in May 2026 via 8 phases:

- **Phase 0**: adopt `local-llm-orchestrator` scaffold
- **Phase 1**: ManagedSystemdAdapter + byte-diff parity with gpu-models
- **Phase 2**: preset loader wired to the shared preset package
- **Phase 3**: Hermes + Harbor integrations as lifecycle hooks
- **Phase 4**: slot system + TurboQuant + fleet preflight
- **Phase 5**: production CLI verbs + injectable preset shim
- **Phase 7**: deleted the upstream `packages/gpu-models/`; compat shim ships in the upstream `~/AI` repo at `bin/gpu-models`
- **Phase 7c**: forked the preset schema into `llmctl.presets`
- **Phase 8**: migration note added to the upstream `~/AI` runbook

Then a second pass added:

- **Phase A**: vLLM HTTP probe for health + discovery
- **Phase B**: TUI Presets screen
- **Phase C**: TUI Managed Units screen

The byte-parity guarantee (fixtures at
`tests/fixtures/env_renders/`) means cutover from `gpu-models`
produces zero diff on disk.
