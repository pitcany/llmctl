# Changelog

## Unreleased

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
