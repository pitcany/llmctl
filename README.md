# llmctl

Local-first Linux control plane for vLLM (and friends).

In plain terms: `llmctl` is a control panel for the model servers on one
machine. It does not serve models itself — it manages the programs that do
(Ollama, vLLM, LM Studio, llama.cpp) and gives you one consistent way to drive
all of them, from the CLI or a full-screen dashboard (`llmctl tui`).

One fact explains most of the design: **the GPUs run one big model at a time.**
vLLM is a single systemd unit spanning every card, so "switching models" means
reconfiguring and restarting that unit — a 1–3 minute operation that also stops
Ollama first to free VRAM. Ollama is the opposite: many smaller models, loaded
on demand. Much of `llmctl` exists to manage that split safely.

## What it does

Eleven verbs cover a normal day. Everything else is one `llmctl help --all` away
and keeps working under its old name.

| Verb | What it does |
| --- | --- |
| `llmctl setup ENGINE` | Install an engine: vLLM, SGLang and FreeToken each get their own Python env; llama.cpp a prebuilt `llama-server` matched to your CUDA driver; Strata its own checkout and installer; Ollama its official installer. Alone, it lists what is installed. |
| `llmctl add SOURCE` | Fetch a model with its engine's own tool and remember it. The shape of SOURCE picks the engine: `Qwen/Qwen3-8B` is vLLM, `qwen3:8b` is Ollama, `unsloth/Qwen3-8B-GGUF:Q4_K_M` is llama.cpp; `--engine sglang\|freetoken` overrides. |
| `llmctl run [NAME]` | Serve a model, whatever its engine. Stops what holds the GPUs, starts NAME, waits until it answers, puts the previous model back if it fails. No NAME: pick from a list. `--dry-run` shows the plan. |
| `llmctl ls` | Everything you can `run`, and which one is serving now. |
| `llmctl stop [NAME]` | Stop what is serving: the vLLM unit, a lane, or a server llmctl started. NAME is a session ID or a name from `ls`; several serving need NAME or `--all`. |
| `llmctl rm NAME` | Forget a model (`--purge` deletes it from the engine and disk too). |
| `llmctl status` | The managed units and what each serves. |
| `llmctl logs [ID]` | Tail a server's log. |
| `llmctl doctor` | Pass/warn/fail report: engines found, GPUs, drift, stale state. |
| `llmctl tui` | The same, as a full-screen dashboard. |
| `llmctl help [CMD]` | Help; `--all` lists the advanced verbs. |

Behind those: a catalog of every model (where the files are, which engine, how
big), presets for vLLM units, sessions for servers llmctl starts itself,
adoption of servers it did not start, an OpenAI-compatible gateway with
aliases, benchmarks, and systemd unit generation. Each is an advanced verb
(`scan`, `model`, `profile`, `start`, `adopt`, `gateway`, `bench`, ...) and is
documented in the [user guide](docs/USER-GUIDE.md).

## How it works

`llmctl` writes the `EnvironmentFile` for an externally-installed
systemd unit (e.g. `vllm-tp.service`), restarts the unit, polls
`/v1/models` for readiness, and runs whatever post-start checks the
installed plugins register (see [Plugins](#plugins)). It owns the
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

## Docs

- **[Quickstart](docs/QUICKSTART.md)** — install + first
  command in 5 minutes
- **[User guide](docs/USER-GUIDE.md)** — full CLI/TUI
  reference, configuration schema, integrations, troubleshooting
- **Workstation runbook** — the author's stack around llmctl (Hermes
  routing, Harbor services, Tailscale) lives in `pitcany/AI`,
  `docs/RUNBOOK.md`

## Quick taste

```bash
llmctl setup vllm                    # its own env with vLLM in it (also: sglang, freetoken, strata, llama.cpp, ollama)
llmctl add Qwen/Qwen3-8B             # download from Hugging Face, serve with vLLM
llmctl add qwen3:8b                  # ollama pull
llmctl ls                            # what can run, what is serving
llmctl run Qwen/Qwen3-8B             # stops what holds the GPUs, starts this, waits
llmctl run                           # pick from the list
llmctl status                        # managed units + what they serve right now
llmctl doctor                        # pass/warn/fail environment report
llmctl tui                           # interactive TUI
```

Add `--dry-run` to `run` to see the plan without changing anything. Most
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
- **A stop that did not take is tracked apart from status.** When a
  process outlives SIGTERM *and* SIGKILL, the row keeps its pid (so
  `reconcile` can still find it) and records `terminate_failed_at`.
  `status` goes on describing reality — once that survivor answers its
  endpoint again, `reconcile` correctly reports it `running` — while
  `llmctl sessions` appends **(stop failed)** so the operator can see
  their stop never landed. The marker clears on a stop that succeeds or
  on a fresh launch. Status is what *is*; `terminate_failed_at` is what
  *happened*, and reconcile owns only the first.
- **Confirmations.** `scheduler.require_confirmation_for_start/stop/
  delete` are honored by `vllm`, `stop`, `restart`, and `delete-model`:
  llmctl prompts when run interactively (never when scripted — prompts
  are TTY-gated) and `--yes/-y` skips the prompt. Declining exits 0.
  `restart` is gated on `require_confirmation_for_stop` — it terminates
  the running process before relaunching, and that half is the risky one.
- **`restart` relaunches.** It stops the session's process and starts it
  again from the stored launch plan, then reports the state it ended in:
  pid and endpoint when running, the error and exit 1 when the relaunch
  failed. A session reports "no process launched" when it has no stored
  plan, and also when its stored plan is itself a dry run — `restart`
  replays the plan as recorded, so a `--dry-run` start stays planned.
  A process that survives SIGTERM *and* SIGKILL is recorded
  `degraded` with its pid kept (so `reconcile` can still see it), and
  both `stop` and `restart` exit 1 rather than claiming success.
- **`start-unit`.** The counterpart to `stop <id> --systemd`. The session
  verbs take a *session*, but stopping one of these units usually removes its
  session row (they detach themselves on stop), leaving nothing to name — so
  `llmctl start-unit gpt-oss-120b` takes the unit as its subject instead.
  Scope is detected, so a user unit starts with `systemctl --user` and no
  sudo. An already-active unit is reported, not restarted, and a name
  neither manager knows is refused after a read-only `systemctl show`
  probe, before any `start` is issued. No session row is created: these units
  adopt themselves from their own `ExecStartPost`, and inventing one here
  would race that hook.
- **Session selectors.** `stop`, `restart`, `logs`, `detach` and the session
  lookup accept an exact id, a **unique id prefix**, or a **served name** —
  `llmctl stop gpt-oss-120b --systemd` rather than a UUID. Session ids are
  UUIDs and `llmctl sessions` prints them elided, so exact-match-only meant
  the id shown in the table could not be pasted back. Resolution is most
  specific first (exact id, then prefix, then name), and a name prefers live
  sessions so a stopped historical row cannot shadow the server answering
  now. Anything ambiguous is refused with the matching ids listed, never
  guessed — guessing would stop the wrong server.
- **Adopted sessions, `--systemd`.** `stop` and `restart` refuse for an
  adopted session by default — systemd owns its lifecycle. `--systemd/-s`
  delegates to the backing unit instead: `stop` issues `systemctl stop`,
  `restart` issues `systemctl restart`, which also brings back a unit that
  is currently down. Scope is detected from systemd's `LoadState`, so user
  units (the llama.cpp servers) get `systemctl --user` and no `sudo`, while
  system units are unaffected. A restarted unit is recorded `starting`, not
  `running` — the endpoint answers only once the model has loaded, and
  `reconcile` promotes the row then.
- **`vllm --no-wait`.** Skipping the readiness poll reports "restart
  issued; readiness not checked", not "ready" — nothing probed the
  endpoint. Exit stays 0: opting out is a choice, not a failure.
- **JSON output.** `models`, `sessions`, `gpus`, `status`, `health`,
  `presets`, `validate`, `doctor`, `runtimes`, and `config show` accept
  `--json`: plain JSON on stdout, no ANSI, stable keys.
- **Exit codes.** 0 = success (or a declined confirmation), 1 =
  operational failure / findings (`validate`, `doctor`), 2 = usage
  errors, unknown ids, and safety refusals (e.g. `llmctl vllm` on a
  preset whose local model path is missing).
- **Config.** `llmctl config path|show|validate` — `show` prints the
  fully-resolved settings with secret-looking fields redacted.

## Model registry

llmctl keeps a SQLite-backed registry of every local model so the CLI,
TUI, and API share the same view.

### Adding and removing models

Same two commands whatever the engine. The source's shape picks the engine;
`--engine` overrides the guess.

| Engine | Add | What runs |
| --- | --- | --- |
| Ollama | `llmctl model add qwen3:8b` | `ollama pull qwen3:8b` |
| vLLM | `llmctl model add Qwen/Qwen3-8B` | `hf download Qwen/Qwen3-8B` (safetensors only) |
| SGLang | `llmctl model add Qwen/Qwen3-8B --engine sglang` | same download |
| llama.cpp | `llmctl model add unsloth/Qwen3-8B-GGUF:Q4_K_M` | `hf download` of that quant's file(s) only |
| LM Studio | `llmctl model add lmstudio:qwen/qwen3-8b` | `lms get qwen/qwen3-8b --yes` |
| any local | `llmctl model add ~/models/foo.gguf` (or a folder) | registers in place, downloads nothing |

Serving is `llmctl run NAME` for every engine (see the table above). A vLLM
`add` also writes a **starter preset** (`~/.config/llmctl/presets/<name>.yaml`)
so the model is swappable at once. It holds only what `config.json` proves:
repo id, native context capped at `vllm.defaults.max_model_len`, TP from
`vllm.defaults` (1 when the heads do not split), an empty quantization flag so
vLLM reads `quantization_config` itself, and **no tool or reasoning parser**.
Set those before tool use. A taken alias gets `-2`; an existing preset for the
same weights is reused. `--purge` deletes a starter preset along with the
weights; a hand-written preset still blocks the purge. SGLang adds are
download-only: SGLang serves through a plugin lane, and llmctl cannot
write one.

### Serving a model: what `run` does per engine

The GPUs are served in three ways, and `llmctl run NAME` covers all three:

| NAME is… | What `run` does |
| --- | --- |
| a vLLM preset, or a vLLM model | Stops what vLLM serves now and restarts `vllm-tp` on this preset (about 2 min). No preset yet: offers to write a starter one first. |
| a lane (FreeToken, GLM, the GPU1 coder) | Stops what shares its cards, runs the lane's start script, waits until it answers. |
| a llama.cpp, SGLang, FreeToken or Strata model | Checks the model fits (one card, or split across both) **before stopping anything**. Then stops vLLM, active lanes and other servers llmctl started, starts the engine's server (`llama-server`, `python -m sglang.launch_server`, `ft serve`, Strata's `serve/server.py`), and waits until it answers. FreeToken and Strata serve from one GPU. |
| an Ollama or LM Studio model | Nothing. Those servers are always running and load a model the first time a request names it (`ollama run qwen3:8b "hi"`, `lms load KEY`). The GPUs are left alone. |
| a model on another machine, an adopted endpoint, an engine that is not installed | Nothing, with the reason (and the `llmctl setup ENGINE` to run). |

Every run that stops something shows the list first and asks. If the new
model does not come up, everything it stopped is started again. Swapping to vLLM
or a lane also stops llama-servers llmctl started, since vLLM will not load next
to them.

**Servers llmctl starts are "sessions".** llmctl runs `llama-server`,
`sglang.launch_server` or `ft serve` itself (no systemd unit) and keeps a
record of it: its process id, port, and status. That record is a session.
`llmctl ls` shows them, and `llmctl stop NAME` ends one. To go back to vLLM, run
`llmctl run PRESET`.

Removing is `llmctl model rm NAME` (`model delete` is the same command). Alone
it only hides the row. The model stays in its engine and on disk, and adding it
again brings the row back. `--purge` removes the model itself: `ollama rm` for
Ollama, the Hugging Face cache entry or files for vLLM/SGLang/llama.cpp, and the
model folder for LM Studio. `--delete-files` is the old name for `--purge`.

Rules to remember:

- A slash means Hugging Face (`ORG/REPO`). No slash means an Ollama tag. Use
  `ollama:user/model` for a namespaced Ollama model, and `hf:ORG/REPO` with
  `--engine ollama` to pull a Hugging Face repo through Ollama.
- `:QUANT` (or `@QUANT`) picks one GGUF build. A GGUF repo with no quant lists
  the builds it has.
- Downloads go to the Hugging Face cache (`$HF_HOME/hub`). A download that
  would leave less than 10 GB free is refused before it starts.
- `--purge` refuses when a preset names the weights (the deployed generator is
  one), while a session runs the model, for rows from another machine, and for
  paths outside `model_dirs.yaml`'s roots. Hard-linked files count as "stays",
  not "freed". Catalog symlinks are unlinked, never followed.

The other registry commands:

```bash
llmctl models                    # list active models
llmctl model show MODEL_ID       # full record (by id OR unique name)
llmctl model add                 # no SOURCE: register a record from prompts
llmctl model add --non-interactive --name X --engine vllm --path /srv/...
llmctl model edit MODEL_ID --notes "primary 70B" --max-context 32768
llmctl model clone MODEL_ID new-name
llmctl model disable MODEL_ID    # hide from default listings (status unchanged)
llmctl model enable MODEL_ID
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

**Pinning the port.** By default the scheduler allocates from
`scheduler.port_ranges`, so a profile launch lands on whatever is free.
Set `parameters.port` to pin it:

```yaml
parameters:
  port: 8005
```

This exists for models reachable two ways — a systemd unit that
hardcodes a port, and `llmctl start --profile`. Without a pin the same
model answers on a different port depending on who launched it, and
anything downstream aimed at the documented endpoint breaks.

An explicit pin is honoured verbatim, including outside the configured
range. If the port is already taken the launch is **refused, never
relocated**: silently moving it is the thing a pin exists to prevent,
and a session whose recorded port is not where the server listens is its
own kind of lie.

## Plugins

llmctl's core knows the engines, the GPUs, presets, sessions and systemd
units. Anything tied to one machine's stack lives in a *plugin*: a package
with an entry point in the `llmctl.plugins` group whose target is a
`register(registry)` function. Four extension points:

| Point | Used for |
| --- | --- |
| companion | a GPU-holding service paused around a vLLM launch (a Docker container, a user service); started again once the model answers, or on rollback |
| post-start check | a read-only verification once the launched model answers (an agent framework's provider URL, a WebUI pin) |
| lane source | extra `llmctl run` rows: servers with their own start script and unit |
| command | extra `llmctl` verbs |

`llmctl-stack` (in `packages/llmctl-stack/`) is the plugin for the `~/AI`
stack: Harbor, Hermes, llm-router lanes and profiles, the model lab.
With no plugin installed `llmctl run` lists presets and registered models
only. `LLMCTL_PLUGINS=none` turns discovery off for one run; a plugin that
fails to import is reported on stderr and skipped.

## Adding a lane (new engine on the GPUs)

A **lane** is a self-contained server with its own systemd unit and start
script (FreeToken, the llama.cpp servers, ...). Four steps, in this order.
FreeToken and GLM-5.3 were both invisible to `llmctl status` while serving,
because they had no `managed_units` row; step 2 now carries that information
so there is nothing left to forget.

1. **Unit + start script.** A systemd *user* unit, and a `<name>-start`
   script that frees the GPUs it needs (`--take-gpus`) and blocks until the
   server answers. The script owns the GPU hand-over — llmctl never
   reimplements it. Have the unit adopt itself with `ds4-llmctl-sync` in
   `ExecStartPost` and detach in `ExecStopPost`.
2. **A lane entry in the lane source a plugin provides.** On this stack that
   is `llm-router/config/router.yaml` (the `llmctl-stack` plugin reads it):
   a backend with `start_cmd`, `upstream`, `served_model`, `gpus:` (the cards
   it occupies — omitted means every card), `health_timeout_s`, and `unit:` +
   `runtime:` (llm-router ignores those two; llmctl reads them). This is the
   lane list, so the lane appears in `llmctl run`, `llmctl status` and the
   TUI as soon as it exists — no `settings.yaml` row needed. Every lane does
   this as of 2026-09-21, so `settings.yaml` holds only units that are not
   lanes (`vllm-tp`, `gpt-oss`, `coder1`). The row's name in `llmctl status`
   is the lane name, and its `source` column reads `lane` rather than
   `settings`.
3. **Only if you need to override it:** a `managed_units.units` row in
   `~/.config/llmctl/settings.yaml` (`unit_name`, `default_port`, `runtime`)
   wins over `unit:`. Units that are not lanes live there too.
4. **Adapter, only if the engine is new** (`llmctl/adapters/<engine>.py`,
   registered in `db.RuntimeName`, `config.default_runtime_configs` and
   `services/router.py`). Without one the lane still works, adopted as the
   generic `openai` runtime, but reports no version and cannot be told apart
   from any other OpenAI-compatible endpoint. Copy `freetoken.py` or
   `sglang.py`: discovery, health and version, never process control.

Then check it: `llmctl status` shows the unit, `llmctl run --list` shows the
lane, `llmctl runtimes` names the engine, and `llmctl run <name> --dry-run`
previews the right stops.

## Preview a launch

`start-unit --dry-run` does the same for a unit llmctl does not own: it
resolves the unit and its scope and says whether it is already active,
starting nothing (an unknown unit is refused here, not on the run after).

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
DELETE /models/{id}           # ?purge=true also removes it from its engine (= model rm --purge); 409 + reason if refused

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
`/benchmarks` CRUD. The API binds loopback by default.

**Exposing it off loopback takes two deliberate steps**, because
`POST /sessions/start` runs local commands by design:

- `api.allow_public_bind: true` — its own flag, *not*
  `scheduler.allow_public_bind` (which governs serving a **model**
  publicly; one shared flag meant enabling that silently exposed the
  control plane too), and
- `api.auth_token` (or `LLMCTL_API_TOKEN`) — a public bind refuses to
  start without one, and forces the bearer middleware on regardless of
  `scheduler.require_auth_token`, which defaults to false.

Related hardening: `api.cors_origins` rejects `"*"` at load (the
middleware allows credentials, so a wildcard would let any page the
operator visits call this API); `runtime=python_script` refuses a
`script` beginning with `-` (the interpreter would read it as a flag,
making the arguments executable code); and `--purge` (`--delete-files`) only
removes paths inside a configured model root.

## TUI keybindings

Global: `d` dashboard, `w` swap, `p` presets, `u` units, `m` models, `f`
profiles, `s` sessions, `g` GPUs, `l` logs, `o` doctor, `b`
benchmarks, `r` refresh, `q` quit, `ctrl+\` command palette. The TUI
auto-refreshes every 3s and never blocks the UI on probes.

| Screen | Keys |
|--------|------|
| Models (`m`) | `a` add (one source box, any engine, live preview; *Manual entry* for the field form), `del` remove (engine-specific "also remove it" box), `enter` serve (same routing as `llmctl run NAME`: vLLM → swap confirm for its preset, or write one first; llama.cpp → swap confirm that frees the GPUs; scripts → launch plan; Ollama/LM Studio → says it loads on demand), `ctrl+s` scan, `e` edit, `c` clone, `x` prune missing |
| Profiles (`f`) | `a` create, `e` edit, `c` clone, `d` delete |
| Swap (`w`) | `enter` preview (what stops/starts, last load time) → confirm swaps; same list as `llmctl run` |
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
  `(backend, source-or-name, host)`. If you've registered the same file
  manually under a different name, `--import` is a no-op for that
  path.
- **The same model shows up twice in `llmctl models`** — `host` is part
  of a model's identity, so a model served from two machines (e.g. an
  LM Studio model on two LM Link devices) is two rows, one per host. The
  `Host` column shows where each lives: the LM Link device name for LM
  Studio fleet models, the local hostname otherwise.

## Serving vLLM on a new machine

`llmctl run` drives one systemd unit per managed role (default role
`vllm-tp`, port 8003). On a fresh host nothing provides that unit yet, so:

```bash
# 1. a Python env that has vLLM (any venv or conda env)
uv venv ~/vllm && VIRTUAL_ENV=~/vllm uv pip install vllm
# 2. the unit: ExecStart = llmctl's shipped launcher, EnvironmentFile =
#    ~/.local/state/llmctl/units/vllm-tp.env (dry-run by default; needs sudo)
llmctl install-vllm-unit --no-dry-run
# 3. a preset (quantization: none for unquantized weights)
cat > ~/.config/llmctl/presets/qwen3-8b.yaml <<'YAML'
alias: qwen3-8b
served_name: qwen3-8b
model_id: Qwen/Qwen3-8B
quantization: none
vllm_quantization_flag: ""
tensor_parallel_size: 1
max_model_len: 32768
YAML
# 4. serve it, from inside the env of step 1 (that is the interpreter the
#    unit runs; $LLMCTL_PYTHON_ROOT pins it without activating)
source ~/vllm/bin/activate && llmctl run qwen3-8b
```

`CUDA_VISIBLE_DEVICES` is the first `tensor_parallel_size` cards unless
`vllm.defaults.gpus` says otherwise; `HF_HOME` is `~/.cache/huggingface`
unless set. `sudo systemctl start|stop|restart vllm-tp` is the only
privileged call, so a NOPASSWD rule for those three keeps `run`
prompt-free.

## Managed units (defaults)

Out-of-the-box, llmctl knows about:

| Role | systemd unit | Port | GPUs |
|------|--------------|------|------|
| TP serving | `vllm-tp` | 8003 | the preset's first `tensor_parallel_size` cards |

Every default is overridable in `~/.config/llmctl/settings.yaml`
(`managed_units.vllm_tp.env_file_path`, `vllm.defaults.gpus`, ...).

You can also **register your own roles**, of any runtime — they then
appear in `llmctl status`, are adoptable via `llmctl adopt-managed`,
show on the TUI Units screen, and are covered by the port-drift check:

```yaml
managed_units:
  units:
    my-llama:
      unit_name: llama-server
      default_port: 8080
      runtime: llama_cpp
```

`vllm-tp`, `vllm_tp` and `fleet` are reserved names inside `units`.
See the user guide for the full schema.

## Test

```bash
cd ~/AI
uv run pytest packages/llmctl/tests -q
uv run ruff check packages/llmctl
```

~770 tests, ~130–160s wall time. Tests use temp SQLite databases, fake
supervisors/probes, and `LLMCTL_CONFIG_DIR` isolation — no GPUs, no
installed runtimes, and no running services are required.
