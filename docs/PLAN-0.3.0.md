# Plan: llmctl 0.3.0

Written 2026-10-14. Start point: `main` at 511e3747 (v0.2.0 plus the help-text fix).

> **Status: shipped 2026-10-14 as v0.3.0** (commit 2fa588bb, PR #14). Kept as a record.
> Differences from the plan:
>
> - Registry rows with status `deleted` do not count as engine users. Two such rows caused the SGLang and FreeToken warnings on the workstation.
> - vLLM on PATH reads its version from package metadata. `vllm --version` imports torch and takes about 9 s, which is more than the 5 s timeout.
> - A recorded engine binary must be executable to count as installed. A row with `active` NULL counts as active. Both came from the CodeRabbit review.
> - Strata showed as present only after the Strata lanes in `llm-router/config/router.yaml` changed from `runtime: openai` to `runtime: strata` (llm-router commit 846a5c3).

This release has two changes. Do each change as one commit on one branch, then open one PR.

1. llmctl tells you which build it is, and it has its own distribution name.
2. `setup` and `doctor` report engines correctly.

The help-text fix on `main` (Strata in `llmctl --help`) is part of this release.

---

## Before you start

1. Read this file and `CHANGELOG.md`.
2. Make a branch from `main`:

   ```bash
   cd ~/Projects/llmctl
   git fetch origin && git checkout main && git merge --ff-only origin/main
   git checkout -b feat/0.3.0
   ```

3. Run the test suite once, so that you know the start state:

   ```bash
   LLMCTL_PLUGINS=none .venv/bin/python -m pytest -q    # about 3.5 minutes; expect 1182 passed, 1 skipped
   ```

Do not run `git push --tags` in this clone. The local tags `v0.1.0` and `v0.3.0` are old `~/AI` monorepo tags from the subtree split. They are not llmctl releases. **The local `v0.3.0` tag blocks the new tag:** delete it with `git tag -d v0.3.0` before you tag this release (step 4 of "Release"). `~/AI` and its GitHub remote keep their copy.

---

## Change 1: `llmctl --version` and the distribution name `pitcany-llmctl`

### Problem

- Two different builds report the same version. The `v0.2.0` tag is 9967762a. `main` is 511e3747. Both report `0.2.0`.
- `llmctl --version` does not exist. Typer prints `No such option '--version'`.
- The distribution name is `llmctl`. On PyPI, a different person owns the name `llmctl`. If an operator types `pip install -U llmctl`, pip replaces this tool with that package. If a project lists `llmctl` as a plain dependency, the same thing occurs.

### Decision

We do not upload to PyPI (operator decision, 2026-10-14). Users install from GitHub. We rename the distribution only to prevent the name clash. The command stays `llmctl`. The Python import stays `llmctl`.

### Steps

1. In `pyproject.toml`, change `name = "llmctl"` to `name = "pitcany-llmctl"`. Do not change `[project.scripts]`. The script stays `llmctl = "llmctl.cli:app"`.
2. Add a `--version` option to the Typer app in `llmctl/cli.py` (the `app = typer.Typer(` block is near line 90). Use an `@app.callback()` with an eager option.
3. Make the option print the version and the commit:
   - Read the version from `llmctl.__version__`.
   - For a git install, read the commit from the installed distribution: `importlib.metadata.distribution("pitcany-llmctl").read_text("direct_url.json")`, key `vcs_info.commit_id`. Print the first 8 characters.
   - For an editable install or a source checkout, `direct_url.json` has no `vcs_info`. Print the version only, or the version plus `+dev`.
   - Output examples: `llmctl 0.3.0+511e3747` (git install), `llmctl 0.3.0` (other).
   - If the metadata lookup fails, print the version only. Do not crash.
4. Search for code that reads distribution metadata by the old name, and change it:

   ```bash
   grep -rn 'distribution("llmctl")\|version("llmctl")\|metadata("llmctl")' llmctl tests
   ```

5. Add tests in `tests/test_cli.py`:
   - `llmctl --version` exits 0 and prints `llmctl <version>`.
   - With a fake `direct_url.json` that has a `commit_id`, the output includes `+<8 chars>`.
   - With no metadata, the output is the version only.
6. Update the install commands in `README.md` and `docs/QUICKSTART.md`. The URL does not change. Add one sentence: "The package is named `pitcany-llmctl`; the command is `llmctl`."
7. Run `uv lock` and commit the new `uv.lock`.

### Changes in `~/AI` (after the merge)

The two pins use the form `llmctl @ git+https://...`. After the rename, change both to `pitcany-llmctl @ git+https://github.com/pitcany/llmctl@<sha>`:

- `~/AI/packages/llmctl-stack/pyproject.toml`
- `~/AI/packages/llm-models-config/pyproject.toml`

Then reinstall in both environments. The old distribution `llmctl` must go first, or two distributions own the same `llmctl/` folder:

```bash
SPEC="pitcany-llmctl @ git+https://github.com/pitcany/llmctl@<sha>"
cd ~/AI && uv lock
VIRTUAL_ENV=~/AI/.venv uv pip uninstall llmctl
VIRTUAL_ENV=~/AI/.venv uv pip install --reinstall --no-deps "$SPEC"
~/miniconda3/envs/vllm-serve/bin/python -m pip uninstall -y llmctl
~/miniconda3/envs/vllm-serve/bin/python -m pip install --force-reinstall --no-deps "$SPEC"
```

Make sure the `llmctl-stack` plugin still loads. `llmctl --help` must show `recommend` and `router-profiles`.

### Done when

- `llmctl --version` prints `llmctl 0.3.0+<sha>` in `~/AI/.venv`.
- `pip show pitcany-llmctl` works in both environments. `pip show llmctl` finds nothing.
- The full suite passes.

Estimate: about 1 hour, plus CI.

---

## Change 2: `setup` and `doctor` report engines correctly

### Problem

On a healthy workstation, `llmctl doctor` shows 5 warnings. Four of them are noise:

```text
WARN backend:freetoken: binary 'ft' not found on PATH
WARN engine:sglang: not installed
WARN engine:freetoken: not installed
WARN engine:strata: not installed
WARN orphan-discoveries: 1 discovered model(s) unseen for over 30d ...   <- real, keep it
```

The operator does not use SGLang or FreeToken. `strata` is wrong: the Strata lane `strata` is serving now, but `setup` and `doctor` report Strata as missing. The version column of `llmctl setup` shows `—` for every engine found on PATH.

A health check that always warns teaches the operator to ignore it.

### Where the code is

- `llmctl/services/engines.py`: `EngineStatus` (about line 755), `engine_statuses()` (about line 763), `_really_installed()` (about line 790), `_version_of()` (about line 724).
- `llmctl/services/doctor.py`: `_check_backends()` (about line 137, gives the `ft` warning) and `_check_engines()` (about line 160).
- `llmctl/services/lanes.py`: `load_lanes()` (about line 140). Each `Lane` has a `runtime` field (`"strata"`, `"ollama"`, `"llama_cpp"`, ...).
- `llmctl/cli.py`: the `setup` command with no argument prints the engine table.

### Steps

1. **Engines that a lane serves count as present.**
   - In `engine_statuses()`, load the lanes (`load_lanes()`, inside a `try`, because a lane source can fail).
   - For an engine with no llmctl install and no binary on PATH, look for a lane with `lane.runtime == engine`. If one exists, return a new source value `"lane"`, with the lane name in place of the binary.
   - In the `setup` table, show `via lane <name>` for that row.
   - In `_check_engines()`, a `lane` row is a `pass`.
2. **Warn only about engines that something needs.**
   - An engine is "needed" when a registered model uses it (`ModelRecord.runtime`) or a lane uses it.
   - A missing engine that nothing needs is not a warning. `DoctorCheck.verdict` has only `pass`, `warn` and `fail` (`llmctl/services/doctor.py`, about line 50). Use `pass` with the text `not used`. Do not add a new level unless the doctor output code handles it.
   - A missing engine that a model needs stays a `warn`. Name the models in the message.
   - Apply the same rule in `_check_backends()`, so that `ft` does not warn when no model or lane uses FreeToken. Do not report the same engine two times: if `_check_engines()` already covers an engine, think about whether `_check_backends()` must skip it.
3. **Show versions for engines found on PATH.**
   - `_version_of()` already runs the version command after `setup`. Call it for `path` rows too.
   - Run it with a short timeout (about 5 s). `engine_statuses()` is used by `setup`, `doctor` and the TUI; it must stay fast. If it is slow, cache the result, or run it only for `setup` and `doctor` and not for the TUI.
   - A failed version command gives `—`, not an error.
4. **Tests** in `tests/test_setup_engines.py` and the doctor tests (`grep -rln _check_engines tests`):
   - A lane with `runtime="strata"` and no install gives `source == "lane"` and a `pass`.
   - A missing engine that no model or lane uses gives no `warn`.
   - A missing engine that a registered model uses gives a `warn` that names the model.
   - A `path` engine gets a version from a fake `run`.
   - A version command that times out gives `—`.

### Check on the workstation

Run these read-only checks with the plugin loaded. The plugin gives the lanes. `LLMCTL_PLUGINS=none` shows no lanes.

```bash
cd /tmp
PYTHONPATH=~/Projects/llmctl ~/AI/.venv/bin/python -m llmctl.main setup
PYTHONPATH=~/Projects/llmctl ~/AI/.venv/bin/python -m llmctl.main doctor | grep -E '^(WARN|FAIL)'
```

Use `llmctl.main`, not `llmctl.cli`: only `llmctl/main.py` has a `__main__` block.

Expected result:

- `strata` shows `via lane strata`, or the name of the Strata lane that serves.
- `vllm`, `llama_cpp` and `ollama` show a version.
- `doctor` shows only the real `orphan-discoveries` warning.

Do not run `llmctl stop` or `llmctl run` during these checks. The Strata lane is serving.

### Done when

- `doctor` on the workstation shows no warning about unused engines.
- `setup` shows Strata as present through its lane, and shows versions for engines on PATH.
- The full suite passes.

Estimate: 2 to 3 hours, plus CI.

---

## Release

Do these steps after the two changes are on the branch.

1. Add both changes to `CHANGELOG.md` under `## Unreleased`. Also add the help-text fix: "`llmctl --help` lists Strata."
2. Make the release commit. Set the version to `0.3.0` in `pyproject.toml` and `llmctl/__init__.py`. Change `## Unreleased` to `## 0.3.0 — <date>` and add a new empty `## Unreleased` above it. Run `uv lock`.
3. Push the branch over SSH and open the PR:

   ```bash
   git push git@github.com:pitcany/llmctl.git feat/0.3.0
   gh pr create -R pitcany/llmctl --base main --head feat/0.3.0
   ```

   CodeRabbit reviews in about 5 minutes. It can post a second comment after you push a fix. Read `gh api repos/pitcany/llmctl/pulls/<N>/comments` before you merge. Merge with `gh pr merge <N> -R pitcany/llmctl --rebase --delete-branch` only after CI passes.
4. Tag the merged `main` and publish the Release. Delete the stale local `v0.3.0` tag first (see "Before you start"):

   ```bash
   git fetch origin && git checkout main && git merge --ff-only origin/main
   git tag -d v0.3.0
   git tag -a v0.3.0 -m "llmctl 0.3.0"
   git push git@github.com:pitcany/llmctl.git refs/tags/v0.3.0
   gh release create v0.3.0 -R pitcany/llmctl --verify-tag --title "llmctl 0.3.0" --notes-file <notes>
   ```

   Make the notes from the `## 0.3.0` section of `CHANGELOG.md`. Add the install line `uv tool install "git+https://github.com/pitcany/llmctl@v0.3.0"`.
5. Change the `~/AI` pins to the tagged commit (see "Changes in `~/AI`" in Change 1). Reinstall in both environments. Run the plugin tests:

   ```bash
   cd ~/AI && ~/miniconda3/envs/vllm-serve/bin/python -m pytest -q packages/llmctl-stack packages/llm-models-config
   ```

   Commit in `~/AI`. Push only when the operator asks.

---

## Not in this release

- **PyPI upload.** The operator decided against it.
- **`stop` unloads Ollama models.** It is a small gain, because Ollama unloads on its own timer.
- **Wheels attached to the GitHub Release.** Do this only if someone needs to install without git.
