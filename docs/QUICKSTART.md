# llmctl Quickstart

5 minutes from "what is this" to a running vLLM swap.

`llmctl` is the local-first control plane for vLLM (and friends) on a
Linux workstation. It writes the `EnvironmentFile` for an
externally-installed systemd unit, restarts the unit, waits for
`/v1/models` to answer, and verifies that downstream consumers
(Hermes, Open WebUI) still see the right served name.

If you used `gpu-models` before — `llmctl` replaced it. The legacy
`gpu-models <verb>` invocations still work via a compat shim.

> For the full reference, see [LLMCTL-USER-GUIDE.md](./LLMCTL-USER-GUIDE.md).
> For the broader workstation context, see [RUNBOOK.md](./RUNBOOK.md).

---

## Install

### Option A — conda env (recommended for ad-hoc use)

```bash
conda activate <your-env>            # any env on Python 3.12+
bash ~/AI/scripts/install-llmctl.sh
```

The script installs `llmctl` editable into the active environment.
Without the wrapper, run the one-line manual install below.

Confirm:

```bash
llmctl --help
```

### Option B — uv workspace (recommended for the `~/AI` repo)

If you're inside `~/AI`, llmctl is already a workspace member:

```bash
cd ~/AI
uv sync --extra dev
```

The `llmctl` console script lands in `~/AI/.venv/bin/`. The thin
shim at `~/AI/bin/llmctl` (in `$PATH` by default) execs it so bare
`llmctl` works from anywhere — no `uv run`, no venv activation.

### Manual install (without the script)

If you want to know what the script does, or you're scripting your
own setup:

```bash
pip install -e ~/AI/packages/llmctl
```

### Verify the install picks up your presets

```bash
llmctl presets
```

You should see one row per `~/.config/llmctl/presets/*.yaml` file.
If the table is empty, that's expected on a fresh box — see [User Guide
§ Presets](./LLMCTL-USER-GUIDE.md#presets) for how to write one.

---

## Three commands you'll actually use

### 1. List your presets

```bash
llmctl presets
```

Shows alias, served name, model id, family, size, TP, quant for every
preset on disk. Aliases are what you pass to `vllm` and `slot` below.

### 2. Start the TP fleet on a preset

```bash
llmctl vllm llama-3.3-70b              # restart vllm-tp.service with this preset
llmctl vllm llama-3.3-70b --dry-run    # render the env file, print the plan, change nothing
llmctl vllm llama-3.3-70b --tq         # force TurboQuant KV cache on
llmctl vllm llama-3.3-70b --no-wait    # don't poll /v1/models after restart
```

What this does, in order:

1. Stops competing units (`agents.target`, `vllm-coder`, `vllm-reasoner`, `ollama`)
2. Stops the Harbor `ollama` Docker container if running (frees GPU memory)
3. Writes `~/AI/services/vllm-tp.env` from your preset
4. `sudo systemctl restart vllm-tp`
5. Polls `http://localhost:8003/v1/models` until it answers (≤5 min)
6. Verifies the Hermes `vllm` provider URL matches the served port

Each step prints a one-liner. Failures abort early; the env file is
written before the restart so you can inspect what would have run.

### 3. Apply a preset to a per-GPU slot

```bash
llmctl slot coder qwen2.5-coder-32b       # GPU 0, port 8001
llmctl slot reasoner qwq-32b-awq          # GPU 1, port 8002
llmctl slot coder qwen2.5-coder-32b --dry-run
```

Slots are TP=1 single-GPU units with a **stable served name**
(`coder` / `reasoner`). The preset only contributes
model/quant/ctx — downstream client configs that talk to
`coder` keep working when you swap the underlying model.

Short-form wrappers also work:

```bash
set-coder qwen2.5-coder-32b
set-reasoner qwq-32b-awq
```

---

## Quick TUI tour

```bash
llmctl tui
```

| Key | Screen |
|-----|--------|
| `d` | Dashboard — overview of everything |
| `p` | **Presets** — daily-driver, enter to launch |
| `u` | **Units** — live `systemctl is-active` + `/v1/models` per unit |
| `m` | Models — registry (presets + Ollama tags + filesystem scans) |
| `s` | Sessions — recorded launch history |
| `g` | GPUs — NVML telemetry |
| `l` | Logs — recent event audit trail |
| `o` | Doctor — backend binary diagnostics |
| `b` | Benchmarks — streaming-TTFT runner history |
| `r` | Refresh active screen |
| `q` | Quit |

The two screens you'll use most:

- **Presets (`p`)** — table of every preset; enter on a row opens a
  picker (TP fleet / coder slot / reasoner slot). Confirming runs
  the same orchestrator as `llmctl vllm <preset>` / `llmctl slot
  <name> <preset>` in a background thread, so the TUI stays
  responsive during the 1–3 min vLLM cold start.
- **Units (`u`)** — live status of every managed unit: which ones
  are `active`, what port they're on, which model IDs they're
  currently serving (probed via `/v1/models`).

---

## Troubleshooting

### "vllm: unavailable — vLLM binary 'vllm' not found on PATH"

Means the HTTP probe found nothing serving and fell back to the
binary check. Either:
- `vllm-tp.service` isn't running (`sudo systemctl status vllm-tp`)
- The unit is bound to a different port than llmctl expects
  (override via `managed_units.vllm_tp.default_port` in
  `~/.config/llmctl/settings.yaml`)

### "No presets found"

Write at least one preset file to `~/.config/llmctl/presets/<alias>.yaml`.
Minimal shape:

```yaml
alias: my-model
served_name: my-model
model_id: org/repo-id
quantization: awq
vllm_quantization_flag: awq_marlin
tensor_parallel_size: 2
max_model_len: 32768
```

See `llmctl presets` against an existing box for canonical examples.

### "sudo: a password is required"

`llmctl vllm <preset>` calls `sudo systemctl restart <unit>`. On
yannik-desktop this works because `NOPASSWD` is configured for the
specific unit names; on other hosts you'll need to either
configure passwordless sudo for those units, run llmctl from a
session that's already authenticated, or set
`managed_units.<role>.launcher_marker: null` in settings to bypass
the guard if you've installed a non-standard launcher.

### Legacy `gpu-models` invocations

Still work via `~/AI/bin/gpu-models` (translates to `llmctl`).
Prints a deprecation hint — silence with:

```bash
export LLMCTL_QUIET_DEPRECATION=1
```

### Nothing visible — is the install correct?

```bash
which llmctl              # should be ~/AI/bin/llmctl or your conda env's bin/
llmctl --help             # if this works, install is fine
llmctl health             # vllm should be "ok" if vllm-tp is running
llmctl status             # shows the managed units + slots and their env paths
```

If `llmctl --help` cannot find the command, confirm that `pip` installed
into the environment currently on your `PATH`.

---

## Next steps

- **Operational reference**: [LLMCTL-USER-GUIDE.md](./LLMCTL-USER-GUIDE.md)
- **Broader workstation context**: [RUNBOOK.md](./RUNBOOK.md)
- **Architecture overview**: [ARCHITECTURE.md](./ARCHITECTURE.md)
