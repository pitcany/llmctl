# Rust successor — M5g pilot log

Rule: [M5 plan, M5g](RUST-M5-PLAN.md) (amended 2026-11-22). A day runs
from 20:52 UTC to 20:52 UTC; day 1 starts when M5b was turned back on
(2026-11-20 12:52 PST). A day counts when `llmctl-next audit --all`
shows no unexplained foreign write for owned workloads on any host. The
earliest M5 exit is the end of day 7: 2026-11-27 20:52 UTC (12:52 PST).

Days 1 and 2 were written on 2026-11-22 from the audit and the
[progress log](RUST-M5-PROGRESS.md), after the rule change.

| Day | Window (UTC) | Counted | Agent operations | Foreign writes (workstation / mbp2 / m5) | Findings |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-20 20:52 → 2026-11-21 20:52 | yes | none | 0 / 0 / 3 | m5: 3 llama-swap request and TTL swaps (explained). The workstation agent binary was replaced at 07:36 UTC (service restart for a deploy, not a lifecycle write). No agent operation ran, so the day proves only that nothing went around the guards |
| 2 | 2026-11-21 20:52 → 2026-11-22 20:52 | yes | 63, all Succeeded. Workstation 35: drills 2, 3, 5 (21:12–21:16 UTC), then 13 `vllm-engine` Stop/Run pairs (21:17–07:09 UTC), and Stop `vllm-engine` ×3, `ollama-gpu1` ×1, Run 19:48–19:54 UTC. m5 28: 24 LaneSwap, 4 LaneWrite on `llama-swap`. mbp2 0 | 0 / 0 / 17 | m5: 13 llama-swap swaps (explained); 4 pid changes of `ollama` and `llama-swap` after a clean reboot (20:06 PST 11-21) and after the console login that followed a kernel panic (~01:58 PST 11-22, `watchdog timeout`, macOS 27.0.1); both restarts identified in `last` (explained). Panic cause not investigated |
| 3 | 2026-11-22 20:52 → 2026-11-23 20:52 | yes | 40, 39 Succeeded, 1 NeedsAttention. Workstation 40 (00:04–19:25 UTC): 9 `vllm-engine` Stop/Run pairs; `inkling2-mini` Stop ×2, Run ×1; `llamacpp-router` Stop ×2, Run ×2, Restart ×1, RouterWrite ×3, RouterModel load/unload ×6; `gpt-oss-120b` Stop ×3, Run ×2. mbp2 0, m5 0 | 0 / 0 / 22 | m5: 22 llama-swap request and TTL swaps on one pid (17091), no restarts (explained). Workstation: the RouterWrite at 16:09 UTC (add `inkling-2-mini-uncensored`) ran while `llamacpp-router` was stopped (Stop at 16:00 UTC), so the reload check failed (`the router did not answer GET /models`), the rollback failed the same way, and the op ended NeedsAttention (`restoration_failed`). Run at 16:11 and the same write at 16:12 succeeded; the router serves now. Not a foreign write. Finding: a preset write does not refuse when the router is stopped. The router preset is not in a git repo (`no commit`). The m5 kernel panic at 13:49 PST 11-23 and its restart (21:50 UTC) fall in day 4 |
