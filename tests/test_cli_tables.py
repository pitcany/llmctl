from __future__ import annotations

import io

from rich.console import Console

from llmctl.cli_tables import whole_id_table


def _render(table, width: int = 80) -> list[str]:
    buf = io.StringIO()
    Console(file=buf, width=width, color_system=None).print(table)
    return buf.getvalue().splitlines()


def test_identifier_whole_and_description_ellipsised() -> None:
    ident = "k" * 40
    console = Console(width=80)
    table = whole_id_table(console, ("Name", "Desc"), [(ident, "d" * 200)], ids={"Name"})
    lines = _render(table)
    data = [ln for ln in lines if ident in ln]
    assert len(data) == 1
    assert "…" in data[0]
    assert all(len(ln) <= 80 for ln in lines)
    assert len(lines) == 5  # top, header, rule, row, bottom


def test_cells_are_escaped() -> None:
    console = Console(width=80)
    table = whole_id_table(console, ("Name", "Desc"), [("a", "[red]x[/red]")], ids={"Name"})
    assert "[red]x[/red]" in "\n".join(_render(table))


def test_markup_opt_in() -> None:
    console = Console(width=80)
    table = whole_id_table(
        console, ("Name", "State"), [("a", "[green]up[/green]")], ids={"Name"}, markup={"State"}
    )
    out = "\n".join(_render(table))
    assert "up" in out and "[green]" not in out


def test_narrow_console_drops_descriptions_not_identifier_characters() -> None:
    """Rich squeezes every column once a table is too wide; identifiers must not pay."""
    a, b = "alias-" + "a" * 24, "target-" + "t" * 23  # 30 cells each
    headers = ("A", "B", "Why", "Status")
    rows = [(a, b, "w" * 40, "s" * 20)]
    console = Console(width=72)  # ids + chrome fit; ids + 2 descriptions x 4 do not
    lines = _render(whole_id_table(console, headers, rows, ids={"A", "B"}), width=72)
    data = [ln for ln in lines if a in ln]
    assert len(data) == 1 and b in data[0], "\n".join(lines)
    assert "Status" not in "\n".join(lines)  # the rightmost description went first
    assert all(len(ln) <= 72 for ln in lines)
