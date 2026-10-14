# Changelog

## Unreleased

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
