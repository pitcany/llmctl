# llmctl quickstart

One tool to install an inference engine, fetch a model, and run it — on vLLM,
SGLang, FreeToken, Strata, llama.cpp or Ollama — from a Linux box with an
NVIDIA GPU. Four commands cover the whole path:

```bash
llmctl setup vllm              # 1. install the engine (its own Python env)
llmctl add Qwen/Qwen3-8B       # 2. download the model, remember it
llmctl run Qwen/Qwen3-8B       # 3. serve it (frees the GPUs first, waits until it answers)
llmctl ls                      # 4. what can run, what is serving
```

> Full reference: [USER-GUIDE.md](./USER-GUIDE.md). Advanced verbs:
> `llmctl help --all`.

---

## Install llmctl

```bash
uv tool install git+https://github.com/pitcany/llmctl      # or: pipx install git+...
llmctl --help
```

`pip install git+https://github.com/pitcany/llmctl` into any Python 3.12+
env works too. Then:

```bash
llmctl doctor      # GPUs, which engines are installed, what to fix
llmctl setup       # the engines, installed or not
```

---

## Pick an engine

`llmctl setup ENGINE` installs one and records where it is; `run` finds it
from then on (no config file to edit). Each engine lands in its own place
under `~/.local/share/llmctl/engines/`:

| Engine | Good for | `setup` does |
| --- | --- | --- |
| `vllm` | dense models, batching, tensor parallel over several GPUs | a venv with `vllm` |
| `sglang` | the same models, SGLang's scheduler | a venv with `sglang[all]` |
| `freetoken` | big MoE models on one consumer GPU plus RAM | a venv with `freetoken[accel]` |
| `strata` | Qwen3.8-Flash-Next on one GPU plus RAM | `git clone` + Strata's own `setup.sh` (downloads the model it picks for your RAM: 70–120 GB, an hour or more) |
| `llama_cpp` | GGUF files, CPU or GPU, split across cards | the prebuilt `llama-server` from GitHub, CUDA build matched to your driver (needs Ubuntu 24.04's glibc; older distributions build from source, `setup` says how) |
| `ollama` | many small models loaded on demand | Ollama's installer (asks for sudo) |

```bash
llmctl setup vllm                        # asks once, then installs
llmctl setup llama.cpp --dry-run         # show the steps, do nothing
llmctl setup strata -- --family coder    # flags after -- go to the engine's installer
```

---

## Add a model

The shape of SOURCE picks the engine; `--engine` overrides.

```bash
llmctl add Qwen/Qwen3-8B                         # Hugging Face repo -> vLLM
llmctl add Qwen/Qwen3-8B --engine sglang         # same download, SGLang serves it
llmctl add Qwen/Qwen3.6-35B-A3B --engine freetoken
llmctl add unsloth/Qwen3-8B-GGUF:Q4_K_M          # one GGUF quant -> llama.cpp
llmctl add qwen3:8b                              # Ollama tag -> ollama pull
llmctl add ~/models/foo.gguf                     # a local file, nothing downloaded
```

Strata's own setup already registers the model it installed; `llmctl ls`
shows it. Another Strata config: `llmctl add ~/.local/share/llmctl/engines/strata/strata-iq2_xs.json`.

---

## Run it

```bash
llmctl run Qwen/Qwen3-8B        # stops what holds the GPUs, starts this, waits
llmctl run                      # pick from the list
llmctl run NAME --dry-run       # the plan, nothing changed
```

What happens, in order: llmctl says what it will stop and start and asks;
stops servers it started, active lanes and the vLLM unit; starts the engine's
server for NAME (`vllm` unit restart, `llama-server`, `python -m
sglang.launch_server`, `ft serve`, Strata's `serve/server.py`); waits until
`/v1/models` answers; on failure puts the previous model back. Ollama and
LM Studio models need nothing started — their servers load a model on the
first request.

### Change GPUs, context or memory

```bash
llmctl settings NAME                          # what NAME runs with, and where that is stored
llmctl run NAME --tp 1 --max-len 32768        # this run only
llmctl run NAME --max-len 32768 --save        # keep it for every later run
llmctl settings NAME --edit                   # open a vLLM preset in $EDITOR
```

`--tp` is the number of GPUs, `--max-len` the context length in tokens and
`--gpu-mem` the share of GPU memory (0.05 to 1). `--save` writes only after
the model started, into the vLLM preset file (only the changed lines; comments
stay) or, for the engines llmctl starts as sessions, into the model's profile.
llama.cpp has no memory share and picks its GPUs from the model size, so it
takes `--max-len` and a one-run `--tp`. Ollama, LM Studio and router lanes
keep their own settings; llmctl refuses the flags for them.

Then:

```bash
llmctl ls             # every model you can run; "serving" marks the live one
llmctl status         # the managed units and what they serve
llmctl logs ID        # tail a server llmctl started
llmctl stop ID        # stop it
```

### vLLM as a persistent unit

A vLLM model `run` as above is a session (a process llmctl supervises).
For a vLLM server that survives reboots, write the systemd unit once —
`llmctl install-vllm-unit --no-dry-run` — and `run` restarts that unit onto
a *preset* (`~/.config/llmctl/presets/<alias>.yaml`, written for you by
`add`). The README's "Serving vLLM on a new machine" has the four lines.

---

## Per-engine, in four lines

```bash
# vLLM
llmctl setup vllm && llmctl add Qwen/Qwen3-8B && llmctl run Qwen/Qwen3-8B

# SGLang
llmctl setup sglang && llmctl add Qwen/Qwen3-8B --engine sglang && llmctl run Qwen/Qwen3-8B

# FreeToken (one GPU + RAM, MoE models)
llmctl setup freetoken && llmctl add Qwen/Qwen3.6-35B-A3B --engine freetoken && llmctl run Qwen3.6-35B-A3B

# Strata (one GPU + RAM, Qwen3.8-Flash-Next; setup picks the size for your RAM)
llmctl setup strata && llmctl run

# llama.cpp (GGUF)
llmctl setup llama.cpp && llmctl add unsloth/Qwen3-8B-GGUF:Q4_K_M && llmctl run Qwen3-8B-GGUF:Q4_K_M

# Ollama
llmctl setup ollama && llmctl add qwen3:8b && ollama run qwen3:8b "hi"
```

---

## Troubleshooting

### `X is not installed: binary '...' not found. Run llmctl setup X`

Exactly that. Or, if the engine is already installed somewhere, point llmctl
at it in `~/.config/llmctl/settings.yaml`:

```yaml
runtimes:
  vllm:
    binary: /opt/vllm/.venv/bin/vllm
  sglang:
    binary: /opt/sglang/.venv/bin/python   # the python that has sglang
```

### `llama_cpp installed, but its prebuilt binary needs a newer glibc`

The GitHub builds target Ubuntu 24.04 (glibc 2.38). On Ubuntu 22.04 and
similar, build from source and point `runtimes.llama_cpp.binary` at
`build/bin/llama-server`; the message prints the exact commands.

### `sudo: a password is required`

`run` on the vLLM *unit* calls `sudo systemctl start|stop|restart vllm-tp`.
Allow those three without a password:

```
yourname ALL=(root) NOPASSWD: /usr/bin/systemctl start vllm-tp, /usr/bin/systemctl stop vllm-tp, /usr/bin/systemctl restart vllm-tp
```

Sessions (`llama-server`, SGLang, FreeToken, Strata) need no sudo.

### `No NVIDIA GPUs detected`

vLLM, SGLang, FreeToken and Strata need one. llama.cpp runs on CPU with
`llmctl start NAME --cpu` (advanced verb).

### Old verbs

`llmctl swap`, `switch` and `vllm` are `run`; `models` is `model list`;
everything else still works and is listed by `llmctl help --all`.

---

## Next steps

- **[User guide](./USER-GUIDE.md)** — every verb, the configuration schema,
  presets, the API and the TUI.
- `llmctl tui` — the dashboard: `w` run, `m` models, `u` units, `g` GPUs.
