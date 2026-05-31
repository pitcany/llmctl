"""Integrations with the surrounding ~/AI stack.

Modules here translate llmctl's abstract concepts into the concrete
artifacts consumed by other components on the host:

* :mod:`launcher_env` — resolves host-specific values (Python prefix,
  CUDA root, HF cache) used in systemd ``EnvironmentFile`` bodies.
* :mod:`vllm_env` — renders the EnvironmentFile body consumed by
  ``scripts/vllm-launcher.sh`` (the ExecStart for ``vllm-tp.service``).

These modules are pure-Python and do no I/O of their own, which keeps
the rendering byte-identical to the gpu-models output it replaces.
"""
