# Review fixes deployed — November 20, 2026

The owner authorized commits, coordinated deployment and real-host acceptance.
The fixes are deployed on the workstation, mbp2 and m5. No application repository
was pushed.

## Commits and deployed artifacts

- llmctl: `bd38024` — enforce operation ownership and journal correctness.
- llm-router: `38c03fe` — hand off the host lock during agent delegation.
- Workstation Rust executable: `~/.local/state/llmctl-next-m4a/llmctl-next.bin`.
  SHA-256: `4dfad27e1d3fbe254c7be2ac995de7e6583093cbfde318d515013313bfeb0035`.
  This matches the release artifact and the running agent's `/proc/<pid>/exe`.
- Both Mac executables: `~/.local/bin/llmctl-next`.
  SHA-256: `b6751e7fb60ec1b6c99f66187fe223123a1e65ec5164023cb597849280c394c3`.
  Built from the committed Rust source on mbp2 and installed through
  `host upgrade <host> --binary`. Both agents passed the protocol health check.
- Workstation Python runtime: `~/AI/.venv`, installed from the committed 0.3.0
  wheel with `uv pip install --no-deps --reinstall`. No dependencies changed.
  Installed `llmctl/integrations/llmctl_next.py` SHA-256:
  `c5267242c55298016b5597a1f89ed78654b5c87032093a7fed44733b7bd7e715`.
- The router runs its committed checkout through `~/AI/llm-router/.venv`.

## Deployment sequence and correction

All three agents had no active operation. The router reported no active request,
switch or lease. Each host received a local backup of its old agent executable
and a consistent SQLite journal snapshot under:

`<agent-data-dir>/deploy-backups/2026-11-20-bd38024/`

The workstation deployment held the shared host flock, stopped the router/API,
replaced the Rust binary atomically, and restarted the three control-plane
services. Both Mac agents were then upgraded. Inference services were not
restarted for deployment.

The first installed-runtime handoff drill caught a packaging mistake: a Python
import probe run from the checkout had masked the old package in `~/AI/.venv`.
That package did not contain the handoff helper. The drill refused safely with
`host_lock_busy`; it changed only a disposable service. The installed package
was backed up, the committed wheel was installed, and llmctl-api was restarted.
A neutral-directory import and byte comparison verified the installed module.
The complete drill then passed using the installed Python runtime.

## Acceptance results

| Check | Result |
| --- | --- |
| Real systemd suite, disposable user services | 22 passed |
| Real launchd suite on mbp2, disposable labels | 7 passed |
| Targeted contract drill using the deployed binary and installed callers | All 7 checks passed |
| Existing journals on workstation, mbp2, m5 | Migrated; 56 / 19 / 21 operations remained readable |
| Old-key replay with unavailable request identity | All three agents returned conflict; no operation was inserted |
| Serving workload signatures before/after | Unchanged on all three hosts |
| Router `/health` and Python API `/health` | Healthy |
| Existing vLLM process | PID 1679188 unchanged |
| Small inference request to the already serving qwen3.8-27b | Returned `OK`, two completion tokens |
| Production audit since 2026-11-21 03:33 UTC (November 20 19:33 PST) | Zero new findings on all three agents |
| Disposable service/label cleanup | No acceptance/spike units or plists remained |

The seven targeted checks used a real systemd service and an isolated journal:

1. A true caller PID claim cannot bypass an already-held flock.
2. Killing the admitting CLI leaves the executor's lock held until the real
   service start completes.
3. Exact replay returns the original operation; changed force conflicts.
4. Installed Python delegation releases and reacquires ownership around restart.
5. The router's actual delegation method releases and reacquires ownership around
   an agent-driven start. This invokes the method with a controlled runner; it is
   not a production HTTP router switch.
6. A raw systemctl stop of the disposable managed unit produces one foreign
   transition after an agent operation.
7. Restarting the isolated agent preserves request identity and the finding
   without duplication.

The previous source verification remains: 314 Rust tests passed, 32 ignored;
51 targeted Python tests and 41 router tests passed. The live suites above
exercise the previously ignored service-manager paths.

## Scope and remaining limits

Destructive drills used disposable CPU HTTP servers with a notional GPU index.
No large-model cold load, production GPU eviction, or production rollback was
performed. A plain `run vllm-engine --dry-run` explicitly planned stop/start;
it was not executed as a supposed no-op. Polling audit remains incomplete actor
attribution, and zero findings during this window is not a seven-day acceptance
claim. Request draining, TUI preset/catalog completion and repository format/lint
follow-ups remain separate work.

Keep the journal backup for investigation. Do not restore it automatically after
new operations: that would discard their history. A binary rollback can retain
the additive schema column. Reverting the old caller lock contract requires
coordinating the callers and agent again.

Raw evidence for this session is in `/tmp/llmctl-deploy-*`,
`/tmp/llmctl-desktop-deployment.json`, `/tmp/llmctl-python-deployment.log`, and
`/tmp/llmctl-real-host-acceptance.log`. These files are ephemeral; this report
records the durable results. The targeted script was
`/tmp/llmctl-real-host-acceptance.py`.

Published artifact follow-up: the Driving llmctl guide still needs its Rust
handoff and old-key replay guidance reconciled. Its index was checked. No
published page was edited.
