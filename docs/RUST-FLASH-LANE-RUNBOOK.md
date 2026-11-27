# Rollback fix and agent-owned flash lane

Status on 2026-11-27: merged to `main` as `ccabf75` (with `e15c58c`, the
router-stopped write refusal). The workstation agent runs it since 21:14 UTC
2026-11-27 (binary SHA-256 `208bdf1c4cb29829f4cdf0e65d2c6425bd81c1b5906cedccfb5d8a289276502c`;
backups in `<agent-data-dir>/deploy-backups/2026-11-27-ccabf75/`). The Mac
agents run it since 21:22 UTC 2026-11-27 (built on mbp2, SHA-256
`f559fbc83b43196b69cda738e21e13cbf4f88456571c574d3da60b6628fa6a80`, through
`host upgrade <host> --binary`; real-launchd suite 7 of 7 on mbp2; the old
binary and a journal snapshot in each Mac's `deploy-backups/2026-11-27-ccabf75/`).
After both upgrades: journals 100 / 19 / 49 operations, zero foreign writes.
Flash adoption, 2026-11-27 22:10 UTC: steps 1-8 of "Deploy and adopt" done
(`strata-fast` readiness 1800 s; workload `flash` adopted, inventory revision 55,
readiness 1800 s; llm-router `50e5b39` adds `llmctl_next: flash`; unit map line
added; router restarted). The round-trip test failed at the flash start: the
lane config `strata-unsloth-ud-q4_k_xl.2gpu.res.json` reads
`~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-ple-nvme/`, which the NVMe cleanup of
2026-11-24 deleted; Strata exits in 1 s (`cannot open the text model`). The
rollback fix worked on real hardware: the agent stopped the failed target,
restarted `vllm-engine` and verified it; the tutor served again after 3 min
19 s, and the audit shows zero foreign writes. Flash stays broken until its
weights path is restored or changed; then repeat the Test section.

Checks before the swap: Rust workspace 356 passed, 1 failed (the known
`bridge_subcommand_relays_http_to_the_socket` flake; 3 of 3 passed on rerun).
Real-systemd suite 21 of 22: `logs_show_output_and_run_boundaries_and_follow_a_restart`
fails on systemd 255 (Zorin 18) only by wording. systemd 255 writes
`Started <unit> - <description>`; the test expects the systemd 249 form
`Started <description>`. After the swap: journal 100 operations unchanged,
zero foreign writes, `vllm-engine` pid unchanged and serving.

Do not deploy llm-router `fix/lane-script-lock-handoff` in place of this
procedure. With that branch the tutor cannot return.

## Faults and fixes

| Fault in build `55c83de` | Fix |
| --- | --- |
| A failed run started the evicted engine before it stopped the failed target | The rollback stops the target first |
| A rollback did not check the evicted engine | It waits for readiness and verifies the model, if the engine served before |
| The agent waited 100 s for a stop. The lanes allow 120 s | 150 s |
| `adopt` recorded a 120 s readiness timeout | `adopt --readiness-timeout`, with a preview check |

Until the deploy, a failed start of `gpt-oss-120b` or `strata-fast` can leave
vLLM failed and the agent blocked. Do not switch to these lanes.

## Evidence

Throwaway Debian 12 containers with real systemd, no network and no GPU. The
engines and `nvidia-smi` are fakes. The fake engine takes the cards immediately.
vLLM takes them later, thus the old build can fail less often on the real host.

| Case | `main` `62b94a7` | This branch |
| --- | --- | --- |
| Tutor to lane to tutor | serves | serves |
| Lane readiness timeout, 8 s stop | reported restored while down | serves when the operation ends |
| Lane readiness timeout, 112 s stop | tutor failed (3 OOM starts), agent blocked | serves when the operation ends |
| Lane exits during load | reported restored while loading | serves when the operation ends |

- Rust workspace in a sandbox: 341 passed.
- Real-systemd suite in a container: 20 of 22 passed on both builds. The same 2
  fail on `main` in that container (journal wording, no `uv build`).
- Not tested: real GPUs, real Strata, real vLLM, macOS launchd.

## Before the window

1. Merge this branch to `main`. When `m7a-distribution` merges later, use that
   branch's `rust/Cargo.lock`.
2. Run the real-systemd suite. All 22 tests must pass.
3. Make sure no operation, lane lease, longjob or router request is active.
4. Check RAM: `free -g`. Flash kept 35.25 GiB of experts in RAM. A vLLM stop
   frees about 11 GiB. Continue only if available RAM plus 11 GiB is at least
   45 GiB. On 2026-11-22 it was 43 GiB, with swap full.
5. Tell the owner the downtime: 13-20 minutes, 30 minutes or more on a failure.

## Deploy and adopt

1. Back up the agent binary, `inventory.json` and a journal snapshot, as in
   `RUST-REVIEW-DEPLOYMENT-2026-11-20.md`.
2. Stop the agent under the host lock. Replace the binary.
3. In `inventory.json`, set `readiness_timeout_secs` of `strata-fast` to 1800.
   It timed out after 1200 s on 2026-11-17. Start the agent.
4. Preview the adoption. The `readiness_timeout` check must pass with 1800 s.

   ```bash
   llmctl-next adopt qwen3.8-flash-next-strata.service --name flash --model qwen3.8-flash-next-strata --port 8007 --gpus 0,1 --readiness-timeout 1800
   ```

5. Run the printed command. It keeps `--readiness-timeout 1800`.
6. In llm-router `config/router.yaml`, add `llmctl_next: flash` to backend
   `flash`.
7. In `~/.config/llmctl/settings.yaml`, add
   `qwen3.8-flash-next-strata.service: flash` to `llmctl_next.units`. Do not
   print that file. It holds a token.
8. Restart llm-router when it has no request in flight.

## Test

1. Start the test with `longjob start flash-roundtrip --router`. Do not use
   `--gpu` or `queue`: their lease blocks the router switch.
2. Switch to flash through llm-router. Wait for the model check.
3. Switch to the tutor through llm-router. Send one request.
4. Run `llmctl-next audit --all`. It must show no new finding.

## Limits after the deploy

- Python llmctl stops its wait for the agent after 1200 s
  (`llmctl_next.py`, `_TIMEOUT_S`). Start flash through llm-router until that
  value is above the 1800 s timeout plus the rollback.
- `strata-start` calls systemd directly. A hand start of flash is then a
  foreign write.
- A failed `llamacpp load --evict` starts the evicted engines in the correct
  order but does not verify them. Fix this before gpt-oss moves into the router.
- `flash-uncensored`, `dsv41` and `glm53` are still not agent-owned.

## Reverse

1. Remove the two configuration lines. Restart llm-router.
2. Run `llmctl-next forget flash --yes`. This stops nothing.
3. To reverse the agent too, stop it, restore the binary and the inventory
   backup, and start it.
