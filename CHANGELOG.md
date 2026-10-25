# Changelog

## Unreleased

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
