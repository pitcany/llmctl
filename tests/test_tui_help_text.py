"""The help modal must not promise a global key that this screen rebinds."""

from __future__ import annotations

from textual.binding import Binding

from llmctl.tui.app import help_text
from llmctl.tui.screens_benchmarks import BenchmarksScreen


class _Plain:
    BINDINGS: list = []


def test_everywhere_lists_r_refresh_on_a_screen_that_keeps_it() -> None:
    assert "refresh" in help_text(_Plain()).split("This screen")[0]


def test_everywhere_omits_r_refresh_where_the_screen_rebinds_r() -> None:
    # Benchmarks binds `r` to a live re-run; "r refresh" there is wrong.
    assert any(isinstance(b, Binding) and b.key == "r" for b in BenchmarksScreen.BINDINGS)
    text = help_text(BenchmarksScreen.__new__(BenchmarksScreen))
    everywhere, own = text.split("This screen")
    assert "refresh" not in everywhere, everywhere
    assert "Re-run" in own, own
