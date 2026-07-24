# llmctl

Local-first Linux control plane for vLLM (and friends).

`llmctl` writes the `EnvironmentFile` for an externally-installed
systemd unit (e.g. `vllm-tp.service`), restarts the unit, polls
`/v1/models` for readiness, and verifies that downstream consumers
(Hermes, Open WebUI) still see the right served name. It owns the
*how* of running a preset; presets themselves live at
`~/.config/llmctl/presets/<alias>.yaml`. Legacy presets in
`~/.config/llm-models/` are symlinked into that directory on first
load so existing installs keep working.

This package replaced `gpu-models` in May 2026. The migration was
non-disruptive — every legacy `gpu-models <verb>` still works via
`bin/gpu-models` shim, the env file output is byte-identical
(verified by 12 fixture files at
`tests/fixtures/env_renders/`), and no production restart was
required.

## What it does

In plain terms: `llmctl` is a control panel for the model servers on one
machine. It does not serve models itself — it manages the programs that do
(Ollama, vLLM, LM Studio, llama.cpp) and gives you one consistent way to drive
all of them, from the CLI or a full-screen dashboard (`llmctl tui`).

One fact explains most of the design: **the GPUs run one big model at a time.**
vLLM is a single systemd unit spanning every card, so "switching models" means
reconfiguring and restarting that unit — a 1–3 minute operation that also stops
Ollama first to free VRAM. Ollama is the opposite: many smaller models, loaded
on demand. Much of `llmctl` exists to manage that split safely.

**Knowing what you have.** `llmctl` keeps a catalog of every model it knows
about — where the files are, which runtime serves it, how big it is.
`llmctl scan` asks each runtime what it has and records the answers;
`llmctl models` lists them; `llmctl model add|edit|clone|delete` maintains them
by hand. `llmctl validate` checks that everything the catalog claims still
exists where it says.

**Getting models.** `llmctl pull <name>` downloads a model into the local
Ollama library with streaming progress. This is Ollama-only — the other
runtimes expect files already on disk, which `scan` then discovers.

**Running models.** Two paths, for two different situations:

| Command | For | What happens |
| --- | --- | --- |
| `llmctl vllm <preset>` | The big shared model on the GPUs | Rewrites the unit's env file and restarts it; stops Ollama first; waits for readiness |
| `llmctl start <model>` | A one-off session for one model | Launches a process, tracks it, and reports `running` only once the endpoint answers |

A **preset** is a recipe at `~/.config/llmctl/presets/<alias>.yaml` — which
checkpoint, how many GPUs, which quantization, how much context — so
`llmctl vllm ornith-35b` means "make the shared unit be Ornith". A **profile**
is the lighter-weight equivalent for the `start` path. Both have a dry run
(`llmctl plan`, `llmctl preview`, or `--dry-run`) that shows the exact command
and VRAM estimate without touching anything.

**Watching it.** `llmctl status` for an overview, `llmctl health` per runtime,
`llmctl gpus` for VRAM and temperature, `llmctl sessions` for what is running,
`llmctl logs` to tail output, and `llmctl doctor` for a pass/warn/fail
environment report.

**Cleaning up after crashes.** When records and reality disagree,
`llmctl reconcile` re-syncs them, `llmctl cleanup` finds dead sessions and
frees the ports they were holding, and `llmctl model prune` clears catalog rows
for models that are genuinely gone.

**Serving it to clients.** `llmctl gateway` runs an OpenAI-compatible endpoint
that downstream apps point at, with aliases (`llmctl aliases`,
`llmctl set-alias`) mapping friendly names onto whatever is actually serving —
so clients need no rewiring when the checkpoint underneath changes.

**Measuring.** `llmctl bench` runs a timed test and records tokens/second,
time-to-first-token and peak VRAM; `llmctl benchmarks` shows the history and
compares runs against a baseline.

**Adopting what it did not start.** `llmctl adopt` brings an
already-running endpoint under management without restarting it, and
`llmctl detach` lets go again — `llmctl` is a layer over an existing setup, not
a replacement for it.

## Docs

- **[Quickstart](docs/QUICKSTART.md)** — install + first command in 5
  minutes
- **[User guide](docs/USER-GUIDE.md)** — full CLI/TUI reference,
  configuration schema, integrations, troubleshooting

For the workstation context this tool was extracted from (Hermes
routing, Harbor services, Tailscale exposure, etc.), see the
`RUNBOOK.md` in the upstream `~/AI` repo.

## Quick taste

```bash
llmctl runtimes                      # what runtimes exist, versions, loaded models
llmctl runtimes inspect ollama       # one runtime's capabilities in detail
llmctl presets                       # list available presets
llmctl vllm ornith-35b               # restart vllm-tp with this preset
llmctl status                        # managed units + what they serve right now
llmctl doctor                        # pass/warn/fail environment report
llmctl tui                           # interactive TUI
```

Add `--dry-run` to render and inspect without changing anything. Most
read commands accept `--json` for stable, script-friendly output.

## Control-plane semantics

- **Runtime inventory.** `llmctl runtimes` shows every configured
  runtime with health, version, endpoint, currently *loaded* models
  (distinct from installed), and honest capability flags — an operation
  a runtime cannot do (e.g. deleting an LM Studio model remotely) is
  reported as unsupported instead of silently no-opping.
- **Readiness-gated starts.** A non-dry-run `llmctl start` of a server
  runtime is only `running` once its endpoint actually answers. If the
  process dies during startup the session is `failed` (with the last
  log lines in the error); if the model is still loading when the wait
  budget (`runtimes.<name>.readiness_timeout_s`, default 90s) runs out
  the session stays `starting` and `reconcile` promotes it later.
- **Degraded detection.** `reconcile` (and `llmctl sessions`, which
  reconciles by default; `--no-fresh` skips it) probes each owned
  session's endpoint: process alive but endpoint dead → `degraded`,
  excluded from gateway routing until it recovers.
- **Confirmations.** `scheduler.require_confirmation_for_start/stop/
  delete` are honored by `vllm`, `stop`, and `delete-model`: llmctl
  prompts when run interactively (never when scripted — prompts are
  TTY-gated) and `--yes/-y` skips the prompt. Declining exits 0.
- **JSON output.** `models`, `sessions`, `gpus`, `status`, `health`,
  `presets`, `validate`, `doctor`, `runtimes`, and `config show` accept
  `--json`: plain JSON on stdout, no ANSI, stable keys.
- **Exit codes.** 0 = success (or a declined confirmation), 1 =
  operational failure / findings (`validate`, `doctor`), 2 = usage
  errors and unknown ids.
- **Config.** `llmctl config path|show|validate` — `show` prints the
  fully-resolved settings with secret-looking fields redacted.

## Model registry

llmctl keeps a SQLite-backed registry of every local model so the CLI,
TUI, and API share the same view.

```bash
llmctl models                    # list active models
llmctl model show MODEL_ID       # full record (by id OR unique name)
llmctl model add                 # interactive add (prompts for missing fields)
llmctl model add --non-interactive --name X --backend vllm --path /srv/...
llmctl model edit MODEL_ID --notes "primary 70B" --max-context 32768
llmctl model clone MODEL_ID new-name
llmctl model disable MODEL_ID    # hide from default listings (status unchanged)
llmctl model enable MODEL_ID
llmctl model delete MODEL_ID                 # soft-delete; files preserved
llmctl model delete MODEL_ID --delete-files  # also remove the artifact (asks for y/n)
```

Scan filesystem & adapter sources, optionally registering the results:

```bash
llmctl scan            # dry-run preview
llmctl scan --import   # persist discovered models into the registry
```

`scan` covers `*.gguf` (llama.cpp + LM Studio), Hugging Face caches
(directories with `config.json`), and Ollama manifests.

For vLLM, `scan` records the checkpoint that each served name resolves
to. It reads the `root` field of `/v1/models` into the model `path`. A
served name that is an alias thus stays traceable to its weights.

## Validation

`llmctl validate` compares the records against the disk and the
network. It is read-only. It exits 1 if it finds anything.

```bash
llmctl validate
```

It does four checks:

| Check | What it finds |
|-------|---------------|
| `preset-model-missing` | A preset `model_id` path that is not on the disk |
| `registry-path-missing` | A registry row with a `path` that is not on the disk |
| `broken-symlink` | A symlink in a configured model root that does not resolve |
| `port-drift` | A managed unit that is active, but that serves nothing on its registered port |

A value such as `org/model` is a Hugging Face repository id, not a
path. The path checks ignore it. The `port-drift` check uses systemd
as its gate: it ignores units that are not active, and it gives no
result on a host that has no `systemctl`.

## Profile management

Profiles are reusable launch configurations (TP layout, context length,
GPU utilisation, extra args, env vars). Seven defaults — `fast`,
`coding`, `reasoning`, `long-context`, `quant`, `adtech`, `tutoring` —
are seeded from `configs/profiles.yaml` on first read.

```bash
llmctl profiles                                # list
llmctl profile show NAME
llmctl profile create                          # interactive
llmctl profile edit NAME --max-model-len 32768
llmctl profile clone NAME new-name
llmctl profile delete NAME
llmctl profile export NAME profile.yaml        # YAML round-trip
llmctl profile import profile.yaml
```

Edits run non-fatal validation — errors block the save, warnings are
reported and the change goes through (so you can save profiles meant
for a different host without llmctl second-guessing you).

## Preview a launch

`preview` renders a launch plan without starting anything: backend,
command, env, selected GPUs, expected VRAM, context length, port, and
health endpoint.

```bash
llmctl preview MODEL_ID --profile coding
```

## Registry migration (backup/restore)

```bash
llmctl export-registry registry.json  # models + profiles + settings
llmctl import-registry registry.json  # merge into the local DB (skips dups)
llmctl import-registry registry.json --replace-profiles
```

Models are deduplicated by `(backend, source)`. Profiles are matched by
name; pass `--replace-profiles` to update in place instead of skipping.

## API

The FastAPI service exposes the same surface so external dashboards can
manage the registry without shelling out to the CLI:

```
GET    /models                # ?include_inactive=true to see disabled rows
POST   /models
GET    /models/{id}
PUT    /models/{id}
DELETE /models/{id}           # ?delete_files=true to also remove files

GET    /profiles
POST   /profiles              # 422 with ValidationIssue[] on errors
GET    /profiles/{id_or_name}
PUT    /profiles/{id_or_name}
DELETE /profiles/{id_or_name}
POST   /profiles/{id}/validate  # preview warnings before PUT
```

Also served (same app): `GET /health`, `GET /doctor` (structured
pass/warn/fail report), `GET /sessions`, `POST /sessions/plan|start|
cleanup`, `POST /sessions/{id}/stop|restart`, `GET /gpus`, and the
`/benchmarks` CRUD. The API binds loopback by default; `llmctl serve`
refuses a public bind unless `scheduler.allow_public_bind` is set
(its mutating routes are unauthenticated).

## TUI keybindings

Global: `d` dashboard, `p` presets, `u` units, `m` models, `f`
profiles, `s` sessions, `g` GPUs, `l` logs, `o` doctor, `b`
benchmarks, `r` refresh, `q` quit, `ctrl+\` command palette. The TUI
auto-refreshes every 3s and never blocks the UI on probes.

| Screen | Keys |
|--------|------|
| Models (`m`) | `enter` preview → *Plan only* or *Launch now* (real start; hidden when the plan is refused), `ctrl+s` scan, `a` add, `e` edit, `c` clone, `d` delete, `x` prune missing |
| Profiles (`f`) | `a` create, `e` edit, `c` clone, `d` delete |
| Presets (`p`) | `enter`/`t` launch preset (restarts the managed unit after confirm), `a` add, `e` edit in `$EDITOR`, `c` clone, `d` delete |
| Sessions (`s`) | `x` stop, `ctrl+r` restart, `c` cleanup; row-highlight tails the log. Failures (e.g. stopping an adopted session) surface as notifications |
| Benchmarks (`b`) | `n` new, `enter` re-run, `c` set baseline, `x` clear baseline, `d` delete |
| Doctor (`o`) | `c` copy install command |

## Best practices

- Prefer **profile updates** over hard-coded `--extra-args` in scripts.
  Updates flow into every consumer at once.
- Use **profile clone** to spin up a variant rather than editing a
  shared baseline you'll later want to roll back.
- `model add` is fine without a registered profile; you can attach a
  `default_profile_id` later via `model edit`.
- For automation pipelines, set `LLMCTL_DB_URL` to point at a
  dedicated SQLite file rather than sharing the user-config DB.

## Troubleshooting

- **"Model name is ambiguous across runtimes"** — two models share a
  name but live on different backends. Pass the model id instead, or
  use `llmctl models` to find the right one.
- **`profile import` fails on missing fields** — every YAML profile
  needs at least `name` and `runtime`. Check with `llmctl profile
  export <existing> -` to see the expected shape.
- **Profiles I created via the API don't show in CLI** — they share
  the same DB, but the CLI and API need the same `LLMCTL_DB_URL`. The
  CLI defaults to `~/.local/share/llmctl/llmctl.sqlite3`; if the API
  is run from a different shell, export the same URL there.
- **`scan --import` skipped my model** — `scan` deduplicates by
  `(backend, source-or-name)`. If you've registered the same file
  manually under a different name, `--import` is a no-op for that
  path.

## Managed units (defaults)

Out-of-the-box, llmctl knows about:

| Role | systemd unit | Port | GPUs |
|------|--------------|------|------|
| TP serving | `vllm-tp` | 8003 | 0,1 (TP=2) |

(The per-GPU coder/reasoner slots were decommissioned 2026-06-14; only the
single `vllm-tp` unit remains, model swapped per preset.)

Every default is overridable in `~/.config/llmctl/settings.yaml` so
llmctl runs on hosts that don't share yannik-desktop's layout. See
the user guide for the full schema.

## Test

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

~655 tests, ~90–115s wall time. Tests marked `requires_gpu`,
`requires_systemd`, `live_hf`, or `bench_live` are skipped in CI; run
them locally on the appropriate host.
