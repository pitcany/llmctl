"""Injectable wrapper around ``llm_models_config.load_all``.

Why a shim
----------
``llm_models_config.load_all()`` takes no arguments — it resolves the
preset directory from ``user_config_dir()`` at call time, which itself
reads ``$XDG_CONFIG_HOME`` at call time. That works fine in production
but is hostile to tests: there's no way to point one test at a temp
directory without monkeypatching env vars *and* reloading the module to
clear its internal caches. We've already been bitten by that — the TUI
test flaked because a sibling test's reload left stale state visible.

This module hides ``load_all`` behind a typed :class:`PresetStore` so
the rest of llmctl can be fully injection-friendly:

* Production callers do nothing different — :func:`default_store`
  returns a :class:`PresetStore` that wraps the real ``load_all``.
* Tests construct an :class:`InMemoryPresetStore` (or any object
  implementing :class:`PresetStore`) and pass it to whichever service
  takes one.

Phase 7c will replace this with a forked-and-cleaned schema that
takes a ``config_dir`` argument natively, at which point this shim
becomes redundant. Until then, the shim is the only place that
touches ``llm_models_config.load_all`` — refactoring is a one-line
change there, not a hunt across the codebase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from llm_models_config import load_all as _real_load_all
from llm_models_config.schema import Model


class PresetStore(Protocol):
    """Anything that can hand back the configured ``{alias: Model}`` mapping.

    Production uses :class:`DefaultPresetStore` (wraps the real
    ``load_all``). Tests use :class:`InMemoryPresetStore` or a custom
    object — anything with ``.load()`` works.
    """

    def load(self) -> dict[str, Model]:
        """Return all presets keyed by alias."""


class DefaultPresetStore:
    """Production store — calls the real ``llm_models_config.load_all``.

    Accepts an optional ``loader`` callable for *very* targeted
    test injections that still want to exercise the real
    :class:`llm_models_config.schema.Model` parsing.
    """

    def __init__(self, loader: Callable[[], dict[str, Model]] | None = None) -> None:
        self._loader = loader or _real_load_all

    def load(self) -> dict[str, Model]:
        """Delegate to the configured loader."""
        return self._loader()


class InMemoryPresetStore:
    """Test store — return whatever the test handed in.

    Construct with ``InMemoryPresetStore({"alias": Model(...)})`` and
    pass to whichever service takes a :class:`PresetStore`. No
    monkeypatching, no module reloads.
    """

    def __init__(self, presets: dict[str, Model]) -> None:
        self._presets = dict(presets)

    def load(self) -> dict[str, Model]:
        """Return a shallow copy so callers can mutate without disturbing state."""
        return dict(self._presets)


def default_store() -> PresetStore:
    """Factory for the production default — kept as a function so the
    indirection layer (rather than a module-level constant) gives
    test fixtures a single import point to patch when needed."""
    return DefaultPresetStore()
