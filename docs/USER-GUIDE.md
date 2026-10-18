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
**how** for. The default config (as of 2026-06-14) knows about one:

- `vllm-engine` (port 8003, dual-GPU TP=2, active)

Since 2026-08-05 you can register **additional roles of any runtime**
under `managed_units.units` (see the schema below). Those roles then
appear in `llmctl status`, are adoptable via `llmctl adopt-managed
<role>`, show on the TUI Units screen, and are covered by the
port-drift check in `llmctl validate` / `llmctl doctor`. The names
`vllm-engine`, `vllm_engine`, the old names `vllm-tp`/`vllm_tp`, and `fleet` are reserved and rejected inside
`units`. Set `runtime:` explicitly for anything that is not vLLM, or
`adopt-managed` mislabels the session.

llmctl never installs these units itself — they already exist on
your box (look in `~/AI/services/`). It just writes their
`EnvironmentFile`, calls `sudo systemctl restart <unit>`, and polls
`/v1/models` for readiness.

### Slot

> **Deprecated (2026-06-14):** The per-GPU slot system (`vllm-coder`, `vllm-reasoner`)
> was decommissioned. Only `vllm-engine` (TP=2 across both GPUs) is now active.
> This section is retained for reference.

### Launch spec / orchestrator

When you run `llmctl vllm <preset>`, the orchestrator:

1. Loads the preset via the canonical preset store
2. Applies any CLI overrides (`--tq`, `--no-tq`)
3. Builds a `VLLMLaunchSpec` (Pydantic, validated)
4. **Refuses if the preset's model path is a local path that is missing**
   (see [Common flags](#common-flags); `--force` overrides)
5. Runs the fleet preflight (stop competing units)
6. Stops every plugin *companion* that is running (on this stack: the
   `harbor.ollama` container, from the `llmctl-stack` plugin)
7. Renders the env file, writes it, restarts the unit
8. Polls `/v1/models` until ready (or times out at 15 min)
9. Starts the stopped services again, then runs the plugin *post-start
   checks* (on this stack: the Hermes provider URL)

Steps 5, 6, and 9 are optional and can be disabled in code; from the
CLI they're always on. Because step 4 precedes 5–7, a refused start
leaves ollama running, the companions untouched, and the env file
holding the previous preset.

---

## CLI reference

### Daily-driver commands

| Command | What |
|---------|------|
| `llmctl presets` | List presets from `~/.config/llmctl/presets/` |
| `llmctl vllm <preset>` | Start TP=2 unit on preset (only active managed unit) |
| `llmctl status` | Managed units with resolved env paths, ports, and live served models |
| `llmctl runtimes` | All runtimes: health, version, endpoint, loaded models |
| `llmctl runtimes inspect <rt>` | One runtime in detail, with capability flags |
| `llmctl health` | Per-runtime health rollup (vLLM, llama.cpp, LM Studio, Ollama) |
| `llmctl run NAME` | Serve anything: a preset, a lane, or any registered model (routed by engine). `NAME` can also be the row number (`#`) from `llmctl ls` |
| `llmctl ls [--json]` | List models with row numbers, whole names at 80 columns; `--json` for machine-readable output |
| `llmctl model add SOURCE` | Add a model for any engine: `qwen3:8b` (Ollama), `Qwen/Qwen3-8B` (vLLM), `org/X-GGUF:Q4_K_M` (llama.cpp), `lmstudio:KEY`, or a local path |
| `llmctl model rm NAME [--purge]` | Remove it from llmctl; `--purge` also removes it from its engine (`ollama rm`, HF cache, files) |
| `llmctl pull <tag>` | Ollama-only shortcut for `model add` (streaming progress; registers the model; exits 1 on failure) |
| `llmctl start-unit <unit>` | Start a systemd unit llmctl does not own (scope detected; no sudo for user units) |

Most read commands accept `--json`. The JSON output is plain and
stable. Use it in scripts.

`llmctl vllm`, `llmctl stop`, and `llmctl delete-model` ask for
confirmation when you run them in a terminal and the matching
`scheduler.require_confirmation_for_*` setting is on. They do not ask
in scripts (no TTY). Pass `--yes` to skip the prompt. A declined
prompt exits 0.

### Common flags

Used by `vllm`:

| Flag | Effect |
|------|--------|
| `--dry-run` | Render the env file + print planned actions, change nothing. Still refuses on a missing local model path |
| `--no-wait` | Skip polling `/v1/models` after restart (returns immediately). Reports "readiness not checked" rather than "ready" — nothing probed the endpoint |
| `--force` | Restart even when the preset's local model path is missing |
| `--tq` | Force `--kv-cache-dtype turboquant_k8v4` regardless of preset |
| `--no-tq` | Force TurboQuant off (omit `VLLM_KV_DTYPE` entirely) |

**Missing model paths.** `llmctl vllm` exits 2 and refuses when the
preset's `model_id` is a *local* path (`/`, `~`, `./`, `../`) that is not
on disk — it would otherwise evict the running model and only then fail
to load. The refusal precedes the fleet preflight, so nothing is stopped
and no env file is written. HuggingFace repo ids (`org/name`) fail open
and are never refused. `--force` overrides.

`--tq` and `--no-tq` are mutually exclusive.

Used by `vllm` only (slots deprecated as of 2026-06-14).

### Observability

| Command | What |
|---------|------|
| `llmctl gpus` | NVML telemetry: VRAM, util, temp, power, processes |
| `llmctl doctor` | Environment report: pass/warn/fail checks with remediation (exits 1 on failures) |
| `llmctl config path\|show\|validate` | Show the settings file, the resolved settings (secrets redacted), or validate them |
| `llmctl models` | Registered model rows (Ollama tags + filesystem scans + live vLLM probe) |
| `llmctl validate` | Find drift between the records and the disk/network (exits 1 on findings) |
| `llmctl logs [SESSION_ID]` | Tail a session log, or show recent events |

`llmctl logs` reads the log file that llmctl wrote for the session. An
adopted session has no such file. For an adopted session that records a
systemd unit (for example `vllm-engine`), the command reads the unit's
journal with `journalctl -u <unit>` instead. It reads the system journal
first; if that query succeeds but finds no entries, it then reads the
user journal (for `systemctl --user` units). If the system-journal query
fails — for example, your user is not a member of the `adm` or
`systemd-journal` group — the command shows the journalctl error and
does not try the user journal.

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

`start`, `stop` and `restart` here are real operational commands for
scheduler-launched sessions; `scan`, `plan` and `cleanup` are the
scaffold-and-debugging end of the table. **Adopted** sessions are how you
drive the local llama.cpp servers day to day — see
[Driving an adopted server](#driving-an-adopted-server) below.

Session states: a non-dry-run `start` of a server runtime waits for
the endpoint. The session is `running` only after the endpoint
answers. If the process dies first, the session is `failed` and the
error holds the last log lines. If the wait budget ends first
(`runtimes.<name>.readiness_timeout_s`, default 90 s), the session
stays `starting` and `reconcile` promotes it later. `reconcile` also
sets `degraded` when the process is alive but the endpoint does not
answer; the gateway does not route to a `degraded` session.

| Command | What |
|---------|------|
| `llmctl sessions` | List sessions. Probes liveness first; `--no-fresh` skips the probe |
| `llmctl scan` | Refresh discovery cache (vLLM HTTP probe + filesystem + Ollama API) |
| `llmctl start MODEL_ID --profile NAME` | Scheduler-based launch (subprocess, not systemd) |
| `llmctl stop SESSION` | Mark a session stopped (`--yes` skips the prompt). `--systemd/-s` stops its backing unit instead |
| `llmctl restart SESSION` | Stop the process and relaunch it from the stored plan (`--yes` skips the prompt); exits 1 if the relaunch failed. `--systemd/-s` restarts its backing unit instead |
| `llmctl plan MODEL_ID` | Print a launch plan without executing |
| `llmctl cleanup [--remove-stale]` | Free ports + purge dead sessions and leftover dry-run plans |
| `llmctl add-model`, `delete-model` | Manual model registry CRUD |
| `llmctl adopt -e URL [-r RUNTIME] [--unit UNIT]` | Track an endpoint llmctl did not launch as an `adopted` session |
| `llmctl adopt-managed ROLE` / `--all` | Adopt a role declared in `managed_units` (incl. your own `units` entries) |
| `llmctl detach SESSION` | Remove an adopted session record; the upstream keeps running |
| `llmctl start-unit UNIT` | Start a unit llmctl does not own — the counterpart to `stop --systemd` |

`--remove-stale` purges `STOPPED`/`FAILED` rows **and** `PLANNED` rows
left by `llmctl start --dry-run`. A dry-run records its plan without
launching anything, so nothing ever transitions that row — and while it
exists it reserves its endpoint against `llmctl adopt`. (One such row
from 2026-06-13 blocked adoption of `:8000` for months.) Real in-flight
`PLANNED` rows are never touched.

When `adopt` refuses because another session already claims the
endpoint, the error names the blocking session id, its status, and the
exact command that clears it — `llmctl cleanup --remove-stale` for a
dry-run leftover, `llmctl stop` for an owned session, `llmctl detach`
for an adopted one.

`llmctl adopt` accepts any runtime id. Use `--runtime openai` for a
generic OpenAI-compatible endpoint (a remote box, LiteLLM, or
llama-server). The gateway routes to the adopted session by its served
name, and `reconcile` probes it like any other adopted session. The
`openai` runtime is adopt-only, because there is no local process to
launch: `llmctl plan` shows the refusal reason, and `llmctl start`
refuses to launch. A start forced past the refusal records a `failed`
session instead of crashing.

### Driving an adopted server

The local llama.cpp servers — `gpt-oss-120b` and `glm-5.3-flash` —
run as systemd **user** units that llmctl tracks but did not launch. They
adopt themselves on start and detach on stop, so llmctl always reflects
what is actually up.

You can drive them without leaving llmctl:

```bash
llmctl sessions                            # what is up right now
llmctl stop  gpt-oss-120b --systemd        # SESSION selector -> stops its unit
llmctl start-unit gpt-oss-120b             # UNIT name
llmctl logs  gpt-oss-120b                  # SESSION selector
```

Note the two different subjects. `stop`, `restart` and `logs` take a
**session selector** (id, prefix, or served name). `start-unit` takes a
**systemd unit name**. Here they happen to read the same because the unit
is named after what it serves; on other units they will not match, and
llmctl will not accept one where it wants the other.

`llmctl logs` shows the session's log file. For an adopted session, which
has none, it falls back to the recorded unit's journal.

**Name it, don't UUID it.** Session ids are UUIDs and `llmctl sessions`
prints them elided, so the id in the table cannot be pasted back. Every
session command accepts three forms, most specific first:

| Form | Example |
|------|---------|
| exact id | `llmctl stop 6778e6c3-1c4e-4c2a-9f31-0d2b7a5e8c11` |
| unique id prefix | `llmctl stop 6778e6c3` |
| served name | `llmctl stop gpt-oss-120b` |

A served name prefers *live* sessions, so a stopped historical row cannot
shadow the server answering now. Anything ambiguous is refused with the
matching ids listed — llmctl will not guess which server to stop.

**Why `stop --systemd` but `start-unit`.** `stop` and `restart` take a
*session*. Stopping one of these units removes its session row (they detach
themselves), so by the time you want it back there is nothing left to name
— `llmctl restart gpt-oss-120b --systemd` would report "Session not found".
`start-unit` takes the **unit** as its subject instead. Use `restart
--systemd` to bounce a server that is currently up, and `start-unit` to
bring back one that is down.

**What to expect.** These models take minutes to load, so a restarted
session is recorded `starting`, not `running` — systemd accepting the
command is not evidence the endpoint works. `reconcile` promotes it to
`running` on the first successful probe, and `llmctl sessions` reconciles
by default:

```
$ llmctl restart gpt-oss-120b --systemd
Session … is starting; the endpoint is not ready yet (large models load
for minutes). `llmctl sessions` will show it running once it responds.
```

**Scope is detected, not configured.** These are user units, so llmctl
issues `systemctl --user` with no `sudo`; `vllm-engine` is a system unit and
still goes through `sudo`. You do not have to say which.

**They cannot share the GPUs.** Starting one of these evicts whatever else
holds the cards. Stop `vllm-engine` first, or the guard wrapper refuses the
start rather than OOM-ing partway through loading the weights.

**If `stop --systemd` or `restart --systemd` is refused** with "llmctl does
not manage its lifecycle", the session has no unit recorded (`start-unit`
is unaffected — it takes a unit name, not a session) — it was adopted without `--unit`. Re-adopt
with `llmctl adopt -e URL -r llama_cpp --unit UNIT` (the units' own
`ExecStartPost` hooks do this for you).

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

A screen's own binding **shadows** these, so a screen never rebinds a
navigation key. Screen-local keys are listed with each screen below.

Auto-refresh fires every 3s on the active screen.

### Presets screen (`p`)

The daily driver. One row per preset. Enter on a row opens a
launch picker:

| Key in picker | Target |
|----------------|--------|
| `t` | TP fleet (vllm-engine, both GPUs) |
| `esc` | Cancel |

Picking a target runs the orchestrator in a worker thread.
Notifications surface launch start, success (with port), and
failure (with reason).

### Units screen (`u`)

Live status. One row per managed unit + one row per slot. Columns:

- **Role** — `unit vllm-engine`, etc. (slots deprecated as of 2026-06-14)
- **Unit** — bare systemd unit name
- **Active** — `active` (green) or `inactive` (muted)
- **Port** — what llmctl thinks the unit listens on
- **Served (live)** — model IDs from `/v1/models` probe; `starting?` (yellow) when active-but-empty
- **Env file** — resolved path to the `EnvironmentFile`

Probes use a 1.5s timeout each; 5 down units add ≤8s to refresh.

| Key | What |
|-----|------|
| `a` | Start the unit under the cursor |
| `x` | Stop it — confirms first, showing what it is currently serving |
| `ctrl+r` | Refresh now |

Both act on the **cursor row**, not the first row. Scope is detected, so a
user unit (`gpt-oss-120b`, `glm-5.3-flash`) is driven with
`systemctl --user` and no `sudo`, while `vllm-engine` goes through `sudo`.

`a` will not bounce a unit that is already active — start is not a synonym
for restart. A start that the guard refuses (missing model shards, a
download still in flight, or not enough free VRAM) surfaces the refusal as
a notification; the screen does not crash and the unit is left alone.

Starting is not instant: the model loads for minutes before the endpoint
answers, so **Served (live)** shows `starting?` until it does. These units
cannot share the GPUs — stop `vllm-engine` first, or the guard refuses.

Note `a`, not `s`: `s` is the app-level binding for the Sessions screen,
and a screen binding shadows it.

### Models screen (`m`)

The registry view. Shows everything `llmctl models` shows. Enter
serves the row the way `llmctl run NAME` would: a vLLM checkpoint opens
the Swap confirm for its preset (or offers to write a starter preset
first), a llama.cpp row opens a Swap confirm that stops vLLM and lanes and
starts `llama-server` (refused up front when the model fits on no card
combination), a script row opens the launch plan modal, and Ollama / LM
Studio rows say they load on the first request. See the README section
"Serving a model: what swap does per engine".

- `a` add: type one source (`qwen3:8b`, `Qwen/Qwen3-8B`,
  `unsloth/Qwen3-8B-GGUF:Q4_K_M`, `lmstudio:qwen/qwen3-8b`, or a path). The
  line under it says which engine will get it and the equivalent shell command.
  Add stays refused until that line is green. Progress arrives as toasts.
  *Manual entry* opens the old field-by-field form.
- `del` remove: the dialog says what happens to this engine's copy. The box
  reads "Also remove it from Ollama (ollama rm ...)" or "Also delete the files
  (frees N GB)". When llmctl will not touch the model itself (another machine,
  outside the model roots, a preset still uses it), the dialog says why and
  offers only the row.

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
  auth_token: null   # bearer token; used when scheduler.require_auth_token is true

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
  vllm_engine:
    enabled: false               # opt-in only; orchestrator commands ignore this
    unit_name: vllm-engine
    runtime: vllm                # runtime `adopt-managed` tags the session with
    env_file_path: null          # auto: $AI_HOME/services/<unit_name>.env, else ~/.local/state/llmctl/units/<unit_name>.env
    launcher_marker: vllm-launcher.sh   # set to null to disable legacy-unit guard
    default_port: 8003

  # Slots (vllm-coder, vllm-reasoner) deprecated as of 2026-06-14; only vllm-engine active

  units:                         # optional: additional roles, any runtime
    gpt-oss:                     # the local gpt-oss-120b llama.cpp server
      unit_name: gpt-oss-120b
      default_port: 8005
      runtime: llama_cpp

  fleet:
    tp: vllm-engine
    ollama: ollama

scheduler:
  default_host: 127.0.0.1
  port_ranges:
    vllm: [8000, 8099]
    llama_cpp: [8100, 8199]
    python_script: [8200, 8299]
  safety_margin_gb: 1.0
  require_auth_token: false
```

You only need entries for things you want to override. A completely
empty file works.

With `scheduler.require_auth_token: true`, every control-plane API
route except `/health` and the docs pages demands
`Authorization: Bearer <token>`. The token comes from `api.auth_token`
or, with higher priority, the `LLMCTL_API_TOKEN` environment variable.
If the flag is on but no token is set, `llmctl serve` refuses to start,
and an already-embedded app answers 503 on all protected routes. This
protects the control-plane API (`:8088`), not the gateway (`:9000`) —
the gateway has its own `router.auth_token`.

### Environment variables

| Var | Effect |
|-----|--------|
| `LLMCTL_CONFIG_DIR` | Override config dir (default: `~/.config/llmctl/`) |
| `LLMCTL_DB_URL` | Override the SQLite URL |
| `LLMCTL_LOG_LEVEL` | Override log level |
| `LLMCTL_API_TOKEN` | Control-plane API bearer token; beats `api.auth_token` from settings.yaml |
| `LLMCTL_PYTHON_ROOT` | Where the vLLM interpreter lives (for `LD_LIBRARY_PATH` / `PATH` in rendered env files) |
| `LLMCTL_CUDA_ROOT` | CUDA toolkit root (default `/usr/local/cuda`) |
| `LLMCTL_VLLM_ENV_FILE` | Direct override for the vllm-engine env file path |
| `LLMCTL_SLOT_CODER_ENV_FILE`, `LLMCTL_SLOT_REASONER_ENV_FILE` | Per-slot env file overrides *(deprecated 2026-06-14 with the slot system)* |
| `AI_HOME` | Used to resolve `$AI_HOME/services/<unit>.env` when no explicit path is set |
| `HF_HOME` | HuggingFace cache (default `~/.cache/huggingface`) |
| `LLMCTL_QUIET_DEPRECATION` | Set to `1` to silence the `gpu-models` shim's deprecation hint |
| `LLMCTL_BIN` | Used by the `gpu-models` shim to locate `llmctl` (defaults to conda env's bin) |

### Env-file path resolution

For each managed unit, the env file path is resolved in this order:

1. Explicit `env_file_path` in `settings.yaml`
2. `$LLMCTL_VLLM_ENV_FILE` (or per-slot `$LLMCTL_SLOT_<UPPER>_ENV_FILE`)
3. `$AI_HOME/services/<unit_name>.env`
4. `$XDG_STATE_HOME/llmctl/units/<unit_name>.env` (fallback; what
   `llmctl install-vllm-unit` points the unit at)

---

## Integrations

The fleet preflight is built in. Everything else around a launch comes
from plugins (`llmctl.plugins`; see the README's *Plugins* section):
*companions* are stopped before the restart and started again after,
*post-start checks* run once the model answers. From the CLI they're
always on; from code you can disable them via `OrchestratorOptions`
(`enable_companions`, `enable_post_start_checks`). The two below are
what the `llmctl-stack` plugin (`packages/llmctl-stack/`) registers.

### Hermes verify (post-start check, llmctl-stack)

After a successful restart, the plugin looks up the corresponding provider
in `~/.hermes/config.yaml`:

| Unit role | Hermes provider name |
|-----------|----------------------|
| `vllm-engine` | `vllm` |

The verify is **read-only**. It prints one of:

- `hermes: vllm -> http://127.0.0.1:8003/v1 (verified)` — all good
- `hermes: WARNING — vllm.base_url is X, expected Y` — drift; user fixes via `hermes config edit`
- `hermes: no 'vllm' provider in config.yaml` — user adds it manually
- `hermes: config not found` — Hermes not installed; no-op

Never auto-mutates the user's config (would clobber hand-tuned
fallback providers / model defaults).

### Harbor ollama container (companion, llmctl-stack)

Before starting a GPU-claiming unit, stops the `harbor.ollama`
container if running. This frees GPU memory so vLLM's init doesn't
OOM on a partially-occupied GPU. No-op when Docker isn't installed
or the container isn't running. Started again once vLLM is ready, or
on rollback.

### Fleet preflight (pre-start)

Stops competing systemd units before starting the target. Order
matters — the fleet target is stopped before the slot services it
gates to avoid systemd `Wants=` restart loops.

| Starting | Stops (in order) |
|----------|------------------|
| `vllm-engine` (TP fleet) | `ollama`, `vllm-engine` |

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
# port: 8000                      # VESTIGIAL — read by no launch path; omit it
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
parity is locked in by 12 fixture files at
`packages/llmctl/tests/fixtures/env_renders/`.

---

## Slots

> **Deprecated (2026-06-14):** The per-GPU slot system was decommissioned.
> Only `vllm-engine` (TP=2 across both GPUs) is now active. This section is
> retained for reference.

---

## Troubleshooting

### `Sharded GGUF is incomplete` / `Model is still downloading`

`llmctl start` and `llmctl plan` refuse a **llama.cpp** model when either
is true:

- the path matches `*-NNNNN-of-MMMMM.gguf` and one of the `MMMMM` sibling
  shards is absent — the refusal names the missing files. llama.cpp opens
  the tail shards implicitly, so without this the server loads for minutes
  before dying;
- HuggingFace `*.incomplete` staging files exist under
  `<root>/.cache/huggingface/download/<subdir>/`, meaning the download is
  still in flight.

Fetch the missing shards, or wait for the download, then retry. The check
is llama.cpp-only (vLLM is exempt), and any filename it cannot parse
**fails open** and is never refused on that basis. `--force` overrides it
like any other scheduler refusal.

The same two checks run again at launch time in `scripts/llama-server-guarded` (a symlink to `~/bin/llama-server-guarded`, the canonical copy since 2026-08-08),
which the `gpt-oss-120b` and `glm-5.3-flash` units exec through —
so a unit started directly by systemd gets them too, along with a per-GPU
free-VRAM floor.

### Health says vllm is unavailable but the unit IS running

The HTTP probe targets `http://localhost:<default_port>/v1/models`.
If your unit listens on a different port, override
`managed_units.vllm_engine.default_port` in `settings.yaml`.

Verify with: `curl http://localhost:8003/v1/models`. If that
returns a model list, llmctl should too — file a bug if not.

### Legacy unit refusal

```
LegacyUnitError: vllm-engine.service ExecStart does not contain 'vllm-launcher.sh'.
```

The unit installed in `/etc/systemd/system/vllm-engine.service` predates
the launcher-script ExecStart and speaks the older `VLLM_TP_*`
schema. llmctl writes the new `VLLM_*` schema, which the old
launcher would silently ignore.

Two options:

- Install the launcher-based unit: `bash ~/AI/scripts/apply-spec-rollout.sh --apply`
- Disable the guard if you've installed a non-standard launcher:
  ```yaml
  managed_units:
    vllm_engine:
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
packages/llmctl/
├── llmctl/
│   ├── adapters/        # vLLM, llama.cpp, LM Studio, Ollama, python
│   │   ├── vllm.py              # HTTP-probe health + discovery (Phase A)
│   │   └── vllm_systemd.py      # VLLMSystemdAdapter (Phase 1)
│   ├── plugins.py       # entry-point registry: companions, checks, lanes, commands
│   ├── integrations/    # external-system glue
│   │   ├── vllm_env.py          # render_vllm_env, render_slot_env (Phase 1+4)
│   │   ├── systemctl.py         # SystemctlRunner (Phase 1)
│   │   ├── fleet.py             # preflight_stop (Phase 4)
│   │   └── turboquant.py        # --tq override (Phase 4)
│   ├── services/
│   │   ├── preset_loader.py     # Model -> VLLMLaunchSpec (Phase 2)
│   │   └── vllm_orchestrator.py # start_vllm_tp, start_slot (Phase 5)
│   ├── presets/         # schema + XDG-aware YAML loader (Phase 7c)
│   ├── tui/             # Textual screens
│   │   ├── screens_presets.py   # Presets screen (Phase B)
│   │   ├── screens_units.py     # Units screen (Phase C)
│   │   └── _modals_presets.py   # TP/coder/reasoner picker (Phase B)
│   ├── cli.py           # Typer command surface
│   ├── config.py        # all settings schemas
│   └── ...
└── tests/               # ~470 tests
    └── fixtures/env_renders/    # 12 byte-parity fixtures
```

### Test coverage

~470 tests. Notable:

- `test_vllm_env_render.py` + `test_vllm_env_parity.py` — render
  function unit tests + byte-parity vs frozen `gpu-models` fixtures
- `test_preset_loader.py` — end-to-end YAML → spec → env file
- `test_vllm_systemd_adapter.py` — full lifecycle with injected systemctl/clock/sleep/http_get
- `test_fleet_preflight.py`, `test_harbor_integration.py`, `test_hermes_integration.py`
- `test_vllm_orchestrator.py` — full lifecycle with stubbed deps
- `test_vllm_adapter_http_probe.py` — Phase A probe behavior
- `test_tui_presets.py`, `test_tui_units.py` — Textual pilot tests

Run: `uv run pytest packages/llmctl/tests -q` (≈45–55s)

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
- **Phase 3**: Hermes + Harbor integrations as lifecycle hooks (moved to the `llmctl-stack` plugin 2026-10-14)
- **Phase 4**: slot system + TurboQuant + fleet preflight
- **Phase 5**: production CLI verbs + injectable preset shim
- **Phase 7**: deleted `packages/gpu-models/`; compat shim at `bin/gpu-models`
- **Phase 7c**: forked the preset schema into `llmctl.presets`
- **Phase 8**: RUNBOOK migration note

Then a second pass added:

- **Phase A**: vLLM HTTP probe for health + discovery
- **Phase B**: TUI Presets screen
- **Phase C**: TUI Managed Units screen

The byte-parity guarantee (fixtures at
`tests/fixtures/env_renders/`) means cutover from `gpu-models`
produces zero diff on disk.
