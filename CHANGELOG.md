# Changelog

## Unreleased

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
