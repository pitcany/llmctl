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
(verified by 14 fixture files at
`tests/fixtures/env_renders/`), and no production restart was
required.

## Docs

- **[Quickstart](../../docs/LLMCTL-QUICKSTART.md)** — install + first
  command in 5 minutes
- **[User guide](../../docs/LLMCTL-USER-GUIDE.md)** — full CLI/TUI
  reference, configuration schema, integrations, troubleshooting
- **[Workstation runbook](../../docs/RUNBOOK.md)** — broader
  ~/AI context (Hermes routing, Harbor services, Tailscale, etc.)

## Quick taste

```bash
llmctl presets                       # list available presets
llmctl vllm llama-3.3-70b            # restart vllm-tp with this preset
llmctl slot coder qwen2.5-coder-32b  # restart vllm-coder slot (GPU 0)
llmctl status                        # managed units + slots overview
llmctl tui                           # interactive TUI
```

Add `--dry-run` to render and inspect without changing anything.

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

## TUI keybindings

| Key | Action |
|-----|--------|
| `m` | Models screen — `a` add, `e` edit, `c` clone, `d` delete |
| `f` | Profiles screen — `a` create, `e` edit, `c` clone, `d` delete |
| `p` | Preset aliases (orchestrator-level, distinct from profiles) |
| `enter` | On a model row: preview launch plan and start session |
| `ctrl+s` | Scan model directories |

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

## What it does

Out-of-the-box, llmctl knows about:

| Role | systemd unit | Port | GPUs |
|------|--------------|------|------|
| TP fleet | `vllm-tp` | 8003 | 0,1 (TP=2) |
| Coder slot | `vllm-coder` | 8001 | 0 (TP=1) |
| Reasoner slot | `vllm-reasoner` | 8002 | 1 (TP=1) |

Every default is overridable in `~/.config/llmctl/settings.yaml` so
llmctl runs on hosts that don't share yannik-desktop's layout. See
the user guide for the full schema.

## Test

```bash
cd ~/AI
uv run pytest packages/llmctl/tests -q
uv run ruff check packages/llmctl
```

383 tests, ~70–80s wall time.
