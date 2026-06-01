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

296 tests, ~45–55s wall time.
