"""Adapter that manages an externally-installed ``vllm-tp.service`` unit.

Unlike :class:`llmctl.adapters.vllm.VLLMAdapter`, this adapter does not
launch processes directly. The vLLM process is owned by systemd via a
unit (``vllm-tp.service``) installed in ``/etc/systemd/system``. The
adapter's responsibilities are:

1. Render a :class:`~llmctl.integrations.vllm_env.VLLMLaunchSpec` to
   the ``EnvironmentFile`` body consumed by the unit's ExecStart
   (``scripts/vllm-launcher.sh``).
2. Write that body to ``services/vllm-tp.env``.
3. Ask systemd to restart the unit so it picks up the new env.
4. Poll ``/v1/models`` on the configured port until the server is ready.

The ``EnvironmentFile`` is re-read by systemd on every restart, so a
``daemon-reload`` is not needed — keeping the NOPASSWD contract narrow
(``start``/``stop``/``restart`` only, no privileged config reloads).

This adapter intentionally refuses to operate against the legacy
inline-ExecStart unit (the older ``VLLM_TP_*`` schema that can't pass
JSON args). The check mirrors gpu-models's ``_ensure_launcher_unit``.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llmctl.config import ManagedUnitConfig
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env

LifecycleHook = Callable[[VLLMLaunchSpec], None]


class LegacyUnitError(RuntimeError):
    """Raised when the installed unit predates the launcher-script ExecStart.

    The legacy unit speaks the older ``VLLM_TP_*`` schema and cannot pass
    JSON args (so no ``--speculative-config``). Writing the new ``VLLM_*``
    schema against it would silently load nothing.
    """


@dataclass(frozen=True)
class ManagedRestartResult:
    """Outcome of :meth:`VLLMSystemdAdapter.restart_with_spec`."""

    env_path: Path
    env_body: str
    ready: bool
    error: str | None = None


class VLLMSystemdAdapter:
    """Manage the ``vllm-tp.service`` systemd unit by writing env + restarting.

    Construction is intentionally side-effect free. Methods are the only
    surface that touches the filesystem or invokes ``systemctl``.

    Args:
        config: Managed-unit configuration; provides unit name, env file
            path, legacy-unit marker, and default port. When omitted,
            a default :class:`ManagedUnitConfig` is used (matches the
            ``vllm-tp`` posture on yannik-desktop).
        env_file_path: Optional explicit override for the env file path,
            takes precedence over the config-resolved path. Kept for
            test ergonomics and ad-hoc CLI use.
        systemctl: Injected runner for tests; falls back to a real one.
        clock: Injected monotonic clock for tests.
        sleep: Injected sleep function for tests.
        http_get: Injected HTTP getter for tests; defaults to
            ``urllib.request.urlopen``.
    """

    def __init__(
        self,
        config: ManagedUnitConfig | None = None,
        *,
        env_file_path: Path | None = None,
        systemctl: SystemctlRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        http_get: Callable[[str, float], object] | None = None,
        pre_start_hooks: list[LifecycleHook] | None = None,
        post_start_hooks: list[LifecycleHook] | None = None,
    ) -> None:
        self.config = config or ManagedUnitConfig(unit_name="vllm-tp", default_port=8003)
        self.env_file_path = (
            Path(env_file_path).expanduser()
            if env_file_path is not None
            else self.config.resolve_env_file()
        )
        self.unit_name = self.config.unit_name
        self.systemctl = systemctl or SystemctlRunner()
        self._clock = clock
        self._sleep = sleep
        self._http_get = http_get or _default_http_get
        # Hooks are deliberately list-typed (not single callables) so
        # multiple integrations (hermes verify, harbor preflight, custom
        # notifiers) can stack without an aggregator pattern. Exceptions
        # from hooks abort the lifecycle; integrations should swallow
        # their own non-fatal warnings internally.
        self.pre_start_hooks: list[LifecycleHook] = list(pre_start_hooks or [])
        self.post_start_hooks: list[LifecycleHook] = list(post_start_hooks or [])

    def ensure_launcher_unit(self) -> None:
        """Raise :class:`LegacyUnitError` against the legacy ExecStart unit.

        The marker substring checked in ``systemctl cat`` output comes
        from :attr:`ManagedUnitConfig.launcher_marker`. Set ``launcher_marker``
        to ``None`` in config to disable this guard entirely (useful when
        you've installed a custom launcher with a different filename).

        Skipped silently when ``systemctl`` is unavailable (e.g. container
        without systemd), since there's no unit to validate in that case.
        """
        if self.config.launcher_marker is None:
            return
        if not self.systemctl.available():
            return
        body = self.systemctl.cat(self.unit_name)
        if not body:
            return  # unit not installed; the start call will surface the error
        if self.config.launcher_marker not in body:
            raise LegacyUnitError(
                f"{self.unit_name}.service ExecStart does not contain "
                f"{self.config.launcher_marker!r}. llmctl writes the VLLM_* "
                f"schema consumed by the launcher script; running against a "
                f"different launcher would silently load nothing or pass the "
                f"wrong args. Install the launcher-based unit first, or set "
                f"managed_units.<role>.launcher_marker: null in settings.yaml "
                f"to bypass this guard."
            )

    def write_env(self, spec: VLLMLaunchSpec) -> tuple[Path, str]:
        """Render ``spec`` and write it to :attr:`env_file_path`.

        Returns the (path, body) pair so callers can assert byte-equality
        in tests without re-reading from disk.
        """
        body = render_vllm_env(spec)
        self.env_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_file_path.write_text(body)
        return self.env_file_path, body

    def stop(self) -> bool:
        """Stop the managed unit if it's active. Returns ``True`` on stop."""
        return self.systemctl.try_stop(self.unit_name)

    def is_active(self) -> bool:
        """Proxy to :meth:`SystemctlRunner.is_active`."""
        return self.systemctl.is_active(self.unit_name)

    def restart_with_spec(
        self,
        spec: VLLMLaunchSpec,
        *,
        wait_for_ready: bool = True,
        timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
    ) -> ManagedRestartResult:
        """Write env, restart the unit, optionally wait for readiness.

        Refuses against a legacy unit (see :meth:`ensure_launcher_unit`).
        The 300s default timeout covers slow cold-starts (Qwen3-Next-80B
        with CUDA-graph capture needs ~3min on 2x5090 — verified
        empirically; values shorter than that gave up before the server
        was up even though it eventually started cleanly).

        Args:
            wait_for_ready: When ``True``, poll ``/v1/models`` until
                reachable or timeout. When ``False``, return immediately
                after the restart call returns.
            timeout_s: Readiness poll timeout.
            poll_interval_s: Seconds between readiness polls.

        Returns:
            A :class:`ManagedRestartResult` carrying the written env path,
            the body string (for tests), readiness flag, and any error.
        """
        self.ensure_launcher_unit()
        for hook in self.pre_start_hooks:
            hook(spec)
        env_path, body = self.write_env(spec)

        restart_result = self.systemctl.restart(self.unit_name)
        if not restart_result.ok:
            return ManagedRestartResult(
                env_path=env_path,
                env_body=body,
                ready=False,
                error=(
                    f"systemctl restart {self.unit_name} failed "
                    f"(exit {restart_result.returncode}): "
                    f"{restart_result.stderr.strip() or restart_result.stdout.strip()}"
                ),
            )

        if not wait_for_ready:
            for hook in self.post_start_hooks:
                hook(spec)
            return ManagedRestartResult(env_path=env_path, env_body=body, ready=True)

        ready = self._wait_for_ready(
            spec.port,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        # Post-start hooks fire only when ready — verification against a
        # half-started service produces misleading diagnostics.
        if ready:
            for hook in self.post_start_hooks:
                hook(spec)
        return ManagedRestartResult(
            env_path=env_path,
            env_body=body,
            ready=ready,
            error=None if ready else f"vLLM did not become ready within {timeout_s:.0f}s",
        )

    def _wait_for_ready(
        self,
        port: int,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> bool:
        """Poll ``http://localhost:<port>/v1/models`` until it responds."""
        url = f"http://localhost:{port}/v1/models"
        deadline = self._clock() + timeout_s
        # Initial pause matches gpu-models: gives the worker time to
        # bind the port before the first request, avoiding a noisy
        # ConnectionRefused on the first try.
        self._sleep(5.0)
        while self._clock() < deadline:
            try:
                self._http_get(url, 5.0)
                return True
            except (urllib.error.URLError, OSError):
                self._sleep(poll_interval_s)
        return False


def _default_http_get(url: str, timeout: float) -> object:
    """Real HTTP GET used in production; pulled out so tests can patch it."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only
