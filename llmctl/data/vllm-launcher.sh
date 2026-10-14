#!/usr/bin/env bash
# vllm-launcher.sh — shipped with llmctl (llmctl/data/vllm-launcher.sh).
#
# Generic vLLM OpenAI-API launcher. The systemd unit `llmctl install-vllm-unit`
# writes points ExecStart at this script and passes configuration via the
# EnvironmentFile that `llmctl swap` renders.
#
# Required env vars:
#   VLLM_MODEL          HuggingFace repo or local path
#   VLLM_PORT           TCP port to bind
#   VLLM_SERVED_NAME    Name advertised in /v1/models
#
# Optional env vars (sensible defaults applied when unset):
#   VLLM_TP                   tensor-parallel size (default 1)
#   VLLM_PP                   pipeline-parallel size (default 1)
#   VLLM_HOST                 bind address (default 0.0.0.0 — matches
#                             existing vllm-tp posture so Tailscale clients
#                             keep working. Set to 172.17.0.1 to expose only
#                             on the Docker bridge.)
#   VLLM_MAX_LEN              max model context (default 32768)
#   VLLM_GPU_MEM              gpu-memory-utilization (default 0.85)
#   VLLM_QUANT                --quantization (e.g. awq_marlin, fp8, modelopt_fp4)
#   VLLM_KV_DTYPE             --kv-cache-dtype (e.g. fp8, nvfp4)
#   VLLM_MAX_SEQS             --max-num-seqs
#   VLLM_TOOL_PARSER          --tool-call-parser (also enables --enable-auto-tool-choice)
#   VLLM_SPEC_CONFIG          --speculative-config JSON (passed as one argv element)
#   VLLM_PREFIX_CACHE         "1" to enable --enable-prefix-caching (default 1)
#   VLLM_CHUNKED_PREFILL      "1" to enable --enable-chunked-prefill (default 1)
#   VLLM_MAX_BATCHED_TOKENS   --max-num-batched-tokens
#   VLLM_TRUST_REMOTE         "1" to add --trust-remote-code (default 1)
#   VLLM_DTYPE                --dtype (default auto for model)
#   VLLM_EXTRA                additional args, whitespace-split
#   VLLM_PYTHON               python interpreter (default: the first python3 on
#                             PATH — the env file puts the serving env's bin/
#                             first, so that is the interpreter vLLM is installed in)

set -euo pipefail
set +B   # disable brace expansion so JSON in args survives the shell

: "${VLLM_MODEL:?VLLM_MODEL is required}"
: "${VLLM_PORT:?VLLM_PORT is required}"
: "${VLLM_SERVED_NAME:?VLLM_SERVED_NAME is required}"

# VLLM_PORT is OUR name for the OpenAI API port, but it is ALSO vLLM's own
# environment variable for the BASE of its internal ports: vLLM scans upward
# from it (see vllm/utils/network_utils.py, get_open_ports_list). systemd puts
# the whole env file in the child environment, so leaving it there makes the TP
# worker RPC bind API_PORT+1, +2 ... — ports other services may hold.
# Fix: keep our API port in a separate name, and give vLLM a high internal base
# that no service uses (VLLM_INTERNAL_PORT_BASE overrides it).
VLLM_API_PORT="${VLLM_PORT}"
export VLLM_PORT="${VLLM_INTERNAL_PORT_BASE:-51000}"

# NOTE: the TP worker sockets bind the primary interface address, not
# loopback. Setting VLLM_HOST_IP=127.0.0.1 here was tested on 2026-07-21 and
# did NOT move them. The worker ports are ephemeral and high, so a firewall
# rule on the LAN interface is the remaining control.

PYTHON="${VLLM_PYTHON:-$(command -v python3 || true)}"

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "vllm-launcher: python interpreter not found: '${PYTHON:-python3 on PATH}'." \
       "Set VLLM_PYTHON in the env file, or activate the env that has vLLM" \
       "before running llmctl swap." >&2
  exit 1
fi

args=(
  -m vllm.entrypoints.openai.api_server
  --model "${VLLM_MODEL}"
  --tensor-parallel-size "${VLLM_TP:-1}"
  --pipeline-parallel-size "${VLLM_PP:-1}"
  --max-model-len "${VLLM_MAX_LEN:-32768}"
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.85}"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${VLLM_API_PORT}"
  --served-model-name "${VLLM_SERVED_NAME}"
)

[[ "${VLLM_TRUST_REMOTE:-1}"     == "1"  ]] && args+=(--trust-remote-code)
[[ "${VLLM_PREFIX_CACHE:-1}"     == "1"  ]] && args+=(--enable-prefix-caching)
[[ "${VLLM_CHUNKED_PREFILL:-1}"  == "1"  ]] && args+=(--enable-chunked-prefill)

[[ -n "${VLLM_DTYPE:-}"               ]] && args+=(--dtype "${VLLM_DTYPE}")
[[ -n "${VLLM_QUANT:-}"               ]] && args+=(--quantization "${VLLM_QUANT}")
[[ -n "${VLLM_KV_DTYPE:-}"            ]] && args+=(--kv-cache-dtype "${VLLM_KV_DTYPE}")
[[ -n "${VLLM_MAX_SEQS:-}"            ]] && args+=(--max-num-seqs "${VLLM_MAX_SEQS}")
[[ -n "${VLLM_MAX_BATCHED_TOKENS:-}"  ]] && args+=(--max-num-batched-tokens "${VLLM_MAX_BATCHED_TOKENS}")

if [[ -n "${VLLM_TOOL_PARSER:-}" ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_PARSER}")
fi

# Speculative config: passed as one argv element, JSON survives because of set +B and quoting.
if [[ -n "${VLLM_SPEC_CONFIG:-}" ]]; then
  args+=(--speculative-config "${VLLM_SPEC_CONFIG}")
fi

# Free-form extra: whitespace-split into the argv array.
if [[ -n "${VLLM_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a _extra <<<"${VLLM_EXTRA}"
  args+=("${_extra[@]}")
fi

echo "vllm-launcher: exec ${PYTHON} ${args[*]}"
exec "${PYTHON}" "${args[@]}"
