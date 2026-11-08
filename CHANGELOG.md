# Changelog

## Unreleased

- Named the state machines (`docs/STATE-MACHINES.md`): `SwapJobStatus`
  (running/succeeded/failed, projected from `OperationPhase`), `LaneState`
  (serving/loading/weights gone/idle), `OperationPhase.is_terminal`,
  `SessionStatus.is_active`/`is_terminal`, and a session transition table
  (`SESSION_TRANSITIONS`) that `set_status` checks and logs against. The CLI,
  TUI and API compare against enum members instead of string literals; the
  swap job API models type `status` with the enum. Stored values, JSON and
  rendered text are unchanged.
- `llmctl start`, `plan`, POST `/sessions` and the TUI start refuse a llama.cpp
  model that a lane already serves (gpt-oss → `gptoss`) and name the lane to
  use with `llmctl run`; before, they planned a second llama-server on the
  lane's weights with llmctl defaults. `start`/`plan` without `--profile` now
  apply the model's default profile, as `run` already did. Found by the P3
  pilot trace (`docs/P3-PILOT-TRACE-2026-11-05.md`).
- `llmctl run` hands the host GPU operation lock to llm-router for the
  duration of its `POST /admin/switch` (`op_lock.handed_off()`) and takes it
  back for restoring paused lanes; if another owner kept it, the command stops
  with the holder named instead of continuing unowned. Required before
  llm-router's optional `switch.op_lock_file` is enabled: otherwise the CLI
  would hold the lock the router waits for.
- Configured second vLLM engines now appear in shared managed-unit inventory,
  status/adoption, and vLLM health/discovery. Port validation and generated units
  preserve user scope. API session planning and related session routes use the
  application's settings consistently with session start. Includes a dated
  workstation capability audit and operational workflow; no live configuration
  migration is required.
- Research-workflow follow-up: TUI swaps hold the operation lock through
  the whole swap and finalize interrupted records. Failed lane/session
  restoration retains the unrestored model names and `needs_attention`.
  Browser job IDs and child exit codes survive API restarts, allowing
  polling to resume with phase, output, and recovery details. New benchmark
  manifests and exports include default generation settings; equivalent
  implicit and explicit defaults produce the same workload hash.
- Benchmarks are trustworthy evidence now (2026-11-03 research-workflow
  plan): ordinary runs are strict-live (a missing endpoint records a
  failure, never a synthetic number; `--synthetic` stays the explicit
  labeled mode), workloads carry a recorded size (`--repetitions`,
  `--warmup`) separate from concurrency, and every result notes requested
  vs achieved concurrency, token-count provenance, per-request failures
  and p50/p90/p99 with sample counts. Each run stores an immutable
  manifest (model/engine identity, launch config, workload hash, runner
  vs serving hardware; unknowns recorded as unknown) exportable via
  `llmctl benchmarks --export` and `GET /benchmarks/{id}/export`; re-runs
  link their source and record drift. Result comparisons are gated by a
  shared eligibility check — live-vs-synthetic, failed, or incompatible
  runs render `n/c`, never a green delta.
- GPU lifecycle changes are coordinated across processes: a host-wide
  operation lock (flock, holder identity in refusals) covers `llmctl
  run/stop/restart`, the swap/session cores, and TUI/API mutations, with
  the API handing lock ownership to the CLI child it spawns. Every
  operation is persisted with explicit phases and honest outcomes
  (`needs_attention` when a rollback could not restore), idempotency keys
  on `POST /swap/run`, and conservative recovery after restarts: a
  verified-surviving child blocks conflicting work; dead owners become
  `interrupted` with guidance; nothing is replayed.
- The TUI opens on the research view — Serving now (endpoint, readiness,
  observation time), Current operation (phase, elapsed, recovery), Recent
  experiments — with inventory and config flags demoted. Models gains
  text/engine filters and a detail panel; selection is anchored to model
  ids so refreshes never retarget an action. Benchmarks `enter` inspects
  instead of re-running (`r` re-runs). The web dashboard gains a model
  filter and shows the operation phase.
- `llmctl ls` marks a vLLM preset or Ollama lane whose weights are not on
  this machine as `weights gone` (a preset `model_id` that is an absent path
  or a Hugging Face repo id with no snapshot in any hub cache; an Ollama
  lane whose model `ollama list` does not have), and `llmctl run` refuses it
  before stopping anything, on every surface that shows the preview (CLI,
  TUI, API). Starting such a row first evicted what was serving and then
  failed to load. The orchestrator's own guard now fails closed on repo ids
  too (it used to let them through as "fetched on demand"), the rule
  llm-router already applied; `--force` still overrides it for vLLM presets.
- `llmctl metrics` (`--watch N`, `--all`, `--json`): what each serving engine
  is doing right now -- requests running and waiting, KV-cache fill,
  prefix-cache hit rate, speculative-decoding acceptance, preemptions -- read
  from vLLM, SGLang and llama.cpp `/metrics`, Strata's JSON `/metrics` and
  Ollama's `/api/ps`. One `pressure` column says whether one more request is
  safe (`ok`), will evict the prefix cache (`high`) or will queue and be
  preempted (`saturated`). `llmctl health` prints the same line under each
  serving session (and `--json` carries it as `load` plus a per-runtime
  `pressure`), and `llmctl doctor` warns (`engine-load:<unit>`) when an engine
  is under pressure or has preempted requests since it started, with the
  remedy. GPU telemetry alone could not tell "99% busy" from "KV cache full,
  generation stalled at 3 tok/s"; now it can.
- Plugins can add screens to `llmctl tui` (`PluginRegistry.tui_screens`, a
  `TuiScreen` with a name, title, help line, key and factory): each gets its
  key, a line in the help (`?`) and "Go to ..." in the command palette, and
  `start=True` makes it the screen the TUI opens on. A plugin screen on a key
  or name the TUI already uses, or whose factory raises, is skipped with a
  notification.

- `llmctl run` preview for lanes whose router decides what a start unloads:
  a lane source can set `Lane.plan`, and its lines are the preview's
  "stops:" (llama-swap evicts by group, which llmctl cannot work out from
  units). A lane without a unit previews as `starts: <its command>`. The
  TUI runs a lane's start command with no stdin, so a prompt cannot wait
  unseen behind captured output.
- Hosts without systemd (macOS, most containers): a missing `systemctl` or
  `sudo` now reads as "unit not active" instead of crashing `llmctl run`,
  `ls`, `stop` and the TUI with `FileNotFoundError`.
- Lanes without a systemd unit: a lane source can set `Lane.stop_cmd`, which
  `llmctl stop` runs to stop it. Lanes with `runtime="llama_swap"` read as
  serving only while llama-swap's `/running` reports them ready (its
  `/v1/models` lists every configured model), like Ollama lanes and `/api/ps`.
- Plugins can add `llmctl doctor` findings (`PluginRegistry.doctor_checks`)
  and sections printed after the `llmctl status` table
  (`PluginRegistry.status_sections`; `--json` is unchanged). A hook that
  crashes is reported, not fatal.

- `llmctl generate-systemd` / `install-systemd` and the session units' `llmctl
  start` fallback write an absolute `ExecStart` (the venv's `bin/llmctl`, or
  `<python> -m llmctl.main`). A bare `llmctl` failed under systemd, which does
  not search `~/.local/bin` or a venv.
- Web dashboard: `llmctl serve` prints `Dashboard: http://127.0.0.1:8088/ui`.
  The page shows the `llmctl ls` table, what is serving (endpoints, gateway
  names, Copy buttons), and Run/Stop buttons that preview first (the dry run)
  and ask before acting; a running swap shows its output live and survives a
  reload. Plain HTML/CSS/JS shipped in the package, nothing loaded from the
  internet, strict CSP. When the API needs a token, the page asks once and
  remembers it in that browser until **Forget token**. See "Web dashboard" in
  the user guide.
- HTTP swap API on `llmctl serve`: `GET /swap` (list, gateway URL, running
  job), `POST /swap/run` (dry run, or a background job answered with 202; 409
  while another job runs), `GET /swap/jobs/{id}` (poll status and output) and
  `POST /swap/stop` (select, dry run, real stop). Jobs are kept in memory. See
  "HTTP API: swap" in the user guide.
- `llmctl hf` finds a Hugging Face model and checks that it fits the GPUs
  before you download it: `search` (ranked by family, size, quantization and
  popularity), `inspect` (`config.json` and tokenizer facts), `plan` (tensor
  parallel size, context length and memory use for the detected GPUs, with the
  `vllm serve` command and the `llmctl add` step) and `fit` (a "Fits?" table
  for the top candidates). Read-only. Ported from the `vserve` tool, which this
  replaces; GPUs come from NVML instead of `nvidia-smi`.
- `llmctl health` now agrees with `doctor`: an engine served by a running
  router lane reports as serving (lanes), an engine that is missing and that
  no model uses shows the new `not_used` state, and a missing engine is never
  reported as "installed".
- Status, health and `ls` tell the truth. For Ollama, `status` and `health`
  report a model as serving only when it is loaded (not merely pulled), and
  label models owned by a router lane. `llmctl run` accepts the row numbers
  (`#`) from `llmctl ls` as well as a name, and unknown names get "did you mean"
  suggestions. `llmctl ls` keeps whole names at 80 columns and has `ls --json`.
- `llmctl settings` without NAME shows the `ls` list and asks which row (on a
  terminal), like `llmctl run`; it also takes the `#` from `llmctl ls`. A
  router lane says its settings live in its own script.
- `llmctl ls` shows where each model answers: endpoint and gateway aliases
  appear below the table and in `ls --json`. `llmctl run NAME` on a lane or
  session that is already serving says where it answers instead of starting
  it again.
- `llmctl aliases` and `llmctl hf search` / `hf fit` print whole identifiers
  (model ids, 8-character session prefixes), one line per row, and both take
  `--json` (`aliases --json`, `hf search --json`).
- `llmctl stop --dry-run` shows what would be stopped and stops nothing; a
  real stop prints the same preview first.

## 0.3.0 — 2026-10-14

- `llmctl --version` prints the version, plus the commit (`llmctl
  0.3.0+1a2b3c4d`) when installed from a git URL.
- The distribution is renamed `pitcany-llmctl`, because `llmctl` on PyPI
  belongs to another project and `pip install -U llmctl` would replace this
  tool. The command and the Python import stay `llmctl`. Installs are from
  GitHub: `uv tool install git+https://github.com/pitcany/llmctl`. An
  environment that has the old `llmctl` distribution must uninstall it
  before it installs `pitcany-llmctl`.
- `llmctl setup` and `doctor` report engines correctly. An engine that a
  router lane serves is present (`via lane NAME`). A missing engine warns
  only when a registered model uses it, and the warning names the models;
  an engine nothing uses passes as `not used`. Engines found on PATH show a
  version (a 5 s probe; vLLM reads its package metadata).
- `llmctl --help` lists Strata.

## 0.2.0 — 2026-10-14

First release from github.com/pitcany/llmctl as its own repository.

- `llmctl setup llama.cpp --from-source` clones the latest release and builds
  `llama-server` with cmake (CUDA for the local cards' compute capability
  when nvcc is present, CPU otherwise). It is chosen automatically when glibc
  is older than the 2.38 the prebuilt builds need; a linker from another
  toolchain earlier on PATH is kept out of the build.
- `llmctl stop` with no ID stops what is serving: the vLLM unit, a router
  lane, or a server llmctl started. A name from `llmctl ls` picks one; when
  several serve it lists them and asks for a name or `--all`. `stop ID` for
  a live session works as before.
- `llmctl run NAME --tp N --max-len N --gpu-mem F` changes GPUs, context
  and memory share for one run on any engine llmctl starts; `--save` keeps
  them, in the vLLM preset (line edits, comments kept) or the model's
  profile (created and linked when it has none; a profile shared by several
  models is refused). `llmctl settings NAME [--edit]` shows those values and
  where they live. Direct SGLang/FreeToken/Strata starts from `run` now use
  the model's default profile.
- `llmctl setup ENGINE` installs an engine so `run` can use it, recorded in
  `engines.yaml` (no settings.yaml edit): vLLM/SGLang/FreeToken in their own
  venvs (uv, or the stdlib venv), llama.cpp from the GitHub release matched
  to the CUDA driver, Strata via `git clone` + its own `setup.sh`, Ollama via
  its installer. `llmctl setup` alone lists engines; `doctor` reports them.
- Strata (github.com/Niko1221/Strata) is a sixth engine: `llmctl add
  <its config>.json` registers a model its setup produced; `run` serves it.
- SGLang and FreeToken are engines llmctl starts itself: `llmctl add
  ORG/REPO --engine sglang|freetoken`, then `llmctl run NAME` launches
  `python -m sglang.launch_server` / `ft serve` as a tracked session, frees
  the GPUs first and rolls back on failure. Engine health reads serving /
  installed-but-idle / not installed (with the `llmctl setup` to run).
- One verb per intent: `llmctl add`, `run`, `stop`, `ls`, `rm`, `status`,
  `logs`, `doctor`, `tui`, `help`. `swap`/`switch`/`vllm` are hidden aliases
  of `run`; the thirty advanced and legacy verbs still work and are listed
  by `llmctl help --all`.
- Plugin seam (`llmctl.plugins`, entry-point group `llmctl.plugins`):
  companions paused around a vLLM launch, post-start checks, lane sources
  and command hooks. The Harbor, Hermes, llm-router and model-lab glue
  moved out of the core into the `llmctl-stack` plugin.
- `llmctl install-vllm-unit`: writes the systemd unit `llmctl swap`
  drives, pointing at the launcher llmctl now ships
  (`llmctl/data/vllm-launcher.sh`).
- Host-neutral defaults: the env file falls back to
  `$XDG_STATE_HOME/llmctl/units/<unit>.env`, `HF_HOME` to
  `~/.cache/huggingface`, and `CUDA_VISIBLE_DEVICES` to the preset's first
  `tensor_parallel_size` cards.
- `llmctl swap` without a Python env prints one line instead of a traceback.
- Declared `huggingface-hub` as a dependency; `pytest-asyncio` and `respx`
  as test dependencies.
