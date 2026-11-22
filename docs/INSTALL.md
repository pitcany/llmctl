# Install llmctl-next

llmctl-next is an agent that runs as your user. It starts, stops and
watches the model engines that you enroll. The agent needs no Python, no
Rust and no Homebrew. An engine that is a Python program keeps its own
Python.

This file is for the Rust `llmctl-next`. It is not for the Python `llmctl`.

The clean-host test reads this file from the release archive and runs each
command block as written (`scripts/clean-host-test.sh`).

## Supported hosts

| Host | Requirement | Clean-host test |
|---|---|---|
| Linux x86_64 | systemd with a user manager | Debian 12 |
| macOS arm64 | macOS 13.0 or later, user logged in to the desktop | Not run yet |

On macOS the agent is a LaunchAgent in the `gui/<uid>` domain. That
domain exists only while the user is logged in to the desktop:

- Log the user in after each restart of the Mac.
- Do not log the user out. To use a different account, use fast user
  switching. It keeps the first user logged in.
- An install over ssh works only during such a desktop session.

Other hosts are not supported.

## Files

| Item | Linux | macOS |
|---|---|---|
| Binary | `~/.local/bin/llmctl-next` | same |
| Data | `~/.local/state/llmctl-next/` | same |
| Service | `~/.config/systemd/user/llmctl-next.service` | `~/Library/LaunchAgents/com.pitcany.llmctl-next.plist` |
| Log | user journal | `~/Library/Logs/llmctl-next/` |

`agent install` records what it writes in `install-manifest.json` in the
data directory. `agent uninstall` removes only what that file records.

## 1. Check the archive

Put the archive and `SHA256SUMS` in an empty directory. Open a shell in
that directory. No public download exists yet.

Linux:

<!-- run: check linux -->
```bash
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf llmctl-next-*-x86_64-unknown-linux-musl.tar.gz
cd llmctl-next-*-x86_64-unknown-linux-musl
```

macOS:

<!-- run: check macos -->
```bash
grep ' llmctl-next-.*-aarch64-apple-darwin.tar.gz$' SHA256SUMS | shasum -a 256 -c -
tar -xzf llmctl-next-*-aarch64-apple-darwin.tar.gz
cd llmctl-next-*-aarch64-apple-darwin
```

Continue only if the check prints `OK`.

The macOS binary has an ad-hoc signature. It is not notarized.

## 2. Install

Run this in the unpacked directory:

<!-- run: install both -->
```bash
./llmctl-next agent install
```

The host id is the short host name. To select a different id, add
`--host-id` and the id.

A new install has an empty inventory. The install does not start, stop or
change an engine.

## 3. Set up the shell

Run this in each new shell, or put it in the startup file of your shell:

<!-- run: shell both -->
```bash
export PATH="$HOME/.local/bin:$PATH"
export LLMCTL_NEXT_DATA_DIR="$HOME/.local/state/llmctl-next"
```

The second line is necessary. Without it, commands such as `status` use
the development directory `~/.local/state/llmctl-next-dev`. They then
report `no daemon`.

On Linux, keep the agent on after logout:

<!-- run: linger linux -->
```bash
loginctl enable-linger "$USER"
```

## 4. Check the agent

<!-- run: first-status both -->
```bash
llmctl-next --version
llmctl-next build-info
llmctl-next agent status
llmctl-next status
```

`agent status` shows that the service runs and that the socket answers.
`status` shows your host id and `0 workloads`.

## 5. Enroll an engine

Enroll a service that exists. The examples use a service named `sleeper`.
Replace the service, name, model, port and GPU index with your values.

- `--model` is the model id that the service must serve.
- `--port` is the port of its OpenAI-compatible endpoint.
- `--gpus` is the GPU index. Use `0` on a host with no NVIDIA GPU.

Linux (a user unit):

<!-- run: preview linux -->
```bash
llmctl-next adopt sleeper.service --name sleeper --model none --port 9 --gpus 0
```

macOS (the label of a job in `~/Library/LaunchAgents/<label>.plist`):

<!-- run: preview macos -->
```bash
llmctl-next adopt com.example.sleeper --name sleeper --model none --port 9 --gpus 0
```

This command is a preview. It changes nothing. Read each check. To enroll
the service, run the command that the preview prints below
`to transfer ownership:`. The service continues to run and its file does
not change.

The printed command has `--allow-unverified` if the agent could not check
the served model. Make sure that the model and port are correct before you
run it.

<!-- run: after-enroll both -->
```bash
llmctl-next status
llmctl-next ops
```

## Upgrade

Check and unpack the new archive as in step 1. Then run this in the new
directory:

<!-- run: upgrade both -->
```bash
./llmctl-next agent install
```

An upgrade keeps the host id, the inventory and the journal. It refuses to
replace a file that you edited.

## Record an older install

An install from before the manifest has no `install-manifest.json`. If it
uses the paths in "Files", record it:

<!-- run: adopt-install both -->
```bash
llmctl-next agent install --adopt
```

This writes the manifest. It does not change or restart the agent.

## Uninstall

To keep the data:

<!-- run: uninstall both -->
```bash
llmctl-next agent uninstall
```

To remove the data and the macOS logs too, run this instead:

<!-- run: purge both -->
```bash
llmctl-next agent uninstall --purge
```

Uninstall stops only the agent. Engines continue to run.

Empty parent directories can stay if you uninstalled with the data kept
and installed again.

## Logs

Linux:

<!-- run: logs linux -->
```bash
journalctl --user -u llmctl-next.service -n 50 --no-pager
```

macOS:

<!-- run: logs macos -->
```bash
tail -n 50 "$HOME/Library/Logs/llmctl-next/agent.err.log"
tail -n 50 "$HOME/Library/Logs/llmctl-next/agent.out.log"
```

## Optional integrations

Both integrations are off by default. To turn one on, set its key in
`agent.toml` in the data directory and restart the agent.

| Key | Value |
|---|---|
| `router_config` | path of the llm-router configuration file |
| `lane_lease` | path of a lane lease file |

The environment variables `LLMCTL_ROUTER_CONFIG` and
`LLMCTL_NEXT_LANE_LEASE` have priority over the keys. The agent does not
start if `agent.toml` is not valid.

## Programs that llmctl-next uses

| Use | Programs |
|---|---|
| Linux services | `systemctl`, `journalctl`, `busctl`, `loginctl` |
| macOS services | `launchctl`, `plutil`, `id`, `env` |
| Disk and archives | `df`, `du`, `tar`, `unzip`, `sh` |
| NVIDIA hosts | `nvidia-smi` |
| Engine installers, only when used | `git`, `curl`, `uv`, `hf` |
| Vendor engines, only when enrolled | `ollama`, `lms` |
| Engine version checks | the Python of that engine |
| Your hooks | the programs that your hook files name |
| Remote hosts, CLI only | `ssh`, `rsync` |

## Report a problem

Open an issue at <https://github.com/pitcany/llmctl/issues>. Include:

- the output of `llmctl-next build-info`;
- the output of `llmctl-next agent status`;
- the operating system and its version;
- the failed command and its message.

Read logs before you send them. Remove tokens, keys and private paths.
