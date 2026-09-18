# llmctl TUI UX — Fix Plan

Source: 5-agent headless audit (2026-09-18). 46 findings; repro scripts in `/tmp/llmctl-ux-{nav,dash,models,svc}/`. Test suite currently 147/147 green.

Order = leverage. Each item: file:line, change, done-when check.

## P0 — Crashes & broken behavior

1. **Preset form crashes app on schema-invalid input.**
   - `_modals_presets.py:318` — escape/pydantic error text before rendering (`textual.markup.escape`), and translate field errors into per-field human sentences ("TP must be 1–8").
   - Done when: entering `tensor_parallel_size=99` shows a readable error in the modal and the app stays alive. Regression test exists (out-of-range value, no exception).

2. **Command palette advertised (`^\`) but empty.**
   - `app.py` — register screen-nav actions as palette commands via a command provider, or remove the footer hint.
   - Done when: opening palette and typing "dashboard" lists "Go to Dashboard".

3. **Adopted-session stop is a lie trap.**
   - `screens_sessions.py:126` + `services/sessions.py:637` — mark adopted rows (kind column), disable/redirect `x` for them with an explanatory toast ("managed by systemd — stop from Units screen").
   - Done when: `x` on an adopted row never shows a "will terminate" modal.

4. **`d` shadowed by Delete on 4 screens** (Models/Presets/Profiles/Benchmarks).
   - `screens_models.py:33, screens_presets.py:70, screens_profiles.py:35, screens_benchmarks.py:25` — rebind Delete to `x` (or Dashboard to `home`/`D`); keep footer consistent.
   - Done when: `d` goes Dashboard from every screen and appears in every footer.

5. **systemd-unavailable shown as "inactive".**
   - `screens_units.py:241` — tri-state the Active cell: `unknown (systemd unavailable)`; rename `starting?` → `active, not responding yet`.
   - Done when: unit row renders "unknown" state when systemctl raises.

6. **Doctor `c` copies brew commands on Linux.**
   - `_data.py:102` — pick install commands by `platform.system()`; drop brew entries off-macOS.
   - Done when: `c` on llama_cpp/LM Studio on Linux copies a working command or says "manual install required".

7. **Numeric garbage silently saved as `None`** in Models + Profiles forms.
   - `_modals_registry.py:46-63` — `_optional_int`/`_optional_float` must surface an inline error instead of swallowing `ValueError`.
   - Done when: typing `abc` in Max context → red inline error, save blocked.

## P1 — End-user comprehension

8. **Doctor summary raw booleans.** `screens_doctor.py:88-97` — render "not detected / OK (CPU-only)" instead of `False/True`; suppress GPU-absence amber on CPU-only hosts.
9. **"Logs" screen has no logs.** `screens_logs.py` — either wire `journalctl --user -u <unit>` (read-only) per selected row, or retitle the screen "Events" and put a logs hint on Units.
10. **Dashboard first-run guidance.** `screens_dashboard.py` — when registry empty, show "No models yet — press m to register one"; drop backend-missing warnings that affect zero models; explain or hide "Safe mode".
11. **Sessions semantics.** `screens_sessions.py:52` — add kind (owned/adopted) column + status legend (planned/degraded); toast when reconcile flips a visible row.
12. **Presets clarity.** `screens_presets.py:43-58` — legend for ●/○/⚠; include the missing model_ref in ⚠ rows; warn before `e` suspends into `$EDITOR`; generate launch-modal blast radius from actual settings (not hard-coded Harbor/Hermes) in `_modals_presets.py:83-96`.
13. **GPU polish.** `screens_gpu.py:23-53` — either Enter opens per-GPU detail or drop row cursor; normalize nulls to `—`; render `55 °C`, `12.0 / 32.0 GiB`.
14. **Units polish.** `screens_units.py` — add restart action + logs hint; don't render absolute dev paths for nonexistent env files; make the empty-state branch reachable or delete it.
15. **Profiles purpose + form labels.** `screens_profiles.py:45-49`, `_modals_registry.py:213` — one-line "profiles = reusable launch parameters" header; fix `(0..1]` typo; friendly labels.

## P2 — Chrome & discovery

16. Help surface: `?`/F1 modal listing global + screen keys (none exists today).
17. Footer: make `enter` actions visible on table screens (currently hidden because DataTable consumes enter).
18. `ctrl+c` no-quit: either bind to quit-with-confirm or toast "press q to quit".
19. `llmctl --help`: "tui — Launch the Textual TUI" (drop "skeleton"), note it's the primary UI.
20. Models: toast on add/edit/clone/delete success (`screens_models.py:352` vs presets which does).

## P3 — Test coverage

21. GPU/logs/sessions screens: mount-only today — add 1-2 render assertions each (null-GPU row, empty logs, adopted session row).
22. Profiles UI: add pilot CRUD-through-UI test.
23. Lock in the fix for #1 with a regression test (out-of-range preset schema value → no crash).

## Suggested sequencing

- **Day 1:** P0 #1, #7 (crashes + data-loss-by-silence) — smallest diffs, biggest user protection.
- **Day 2:** P0 #2–#5 (nav trap + lie traps).
- **Day 3+:** P1 in listed order; P2/P3 as fillers. Each P0/P1 item gets its own commit per repo convention.

## Status — implemented 2026-09-18

All 23 items done; regression tests in `tests/test_tui_ux_fixplan.py`.

- **P0:** #1 8c8267ca · #2 2d470609 · #3 db80abc6 · #4 b03d213c (+13b77129 title hints; Delete is now the Delete key, not `x` — `x` is taken on Models/Benchmarks) · #5 96fbf16d · #6 d90b6152 · #7 00d654d0
- **P1:** #8 d57cd141 · #9 c88d64ed (retitled Events; Units `j` shows the unit journal) · #10 26bf412e · #11 f4778495 · #12 8a273b9c · #13 60f9cf59 (row cursor dropped) · #14 4b186ef0 (shift+R restart) · #15 1fcc0cce (+ labels in 00d654d0)
- **P2:** #16 + #18 73d02262 · #17 02373148 · #19 + #20 abf4977a
- **P3:** #21 + #22 846babaa · #23 8c8267ca
