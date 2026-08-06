"""No console.print in cli.py may interpolate untrusted text unescaped.

`console.print` runs Rich console markup over everything it is handed. Text that
carries a path, an exception, or a process message is bracket-rich in practice:
``[/INST]`` chat tokens raise ``MarkupError`` (a traceback instead of output),
and ``[rank0]:`` is silently deleted (output that differs from reality).

#167 fixed the sites that were crashing in the field. This is the sweep and the
guard for the rest: rather than trusting a one-time pass, the check walks the
AST of ``cli.py`` and fails on any *new* unescaped interpolation of a
risky-looking expression. That makes the property hold for code not yet written.

Intentional markup — a colour or style name substituted into a tag — is not
untrusted and must not be escaped, so it is excluded by name below.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI_PATH = Path(__file__).resolve().parents[1] / "llmctl" / "cli.py"

#: Substrings that mark an interpolated expression as carrying text from
#: outside the program: a filesystem path, an exception, a subprocess or
#: adapter message, or a registry/user-supplied name.
RISKY = (
    "exc",
    ".message",
    ".error",
    ".name",
    "path",
    "detail",
    "remediation",
    "endpoint",
    "url",
    "preset",
    "unit",
    "served",
    "warning",
    "reason",
    "action",
    "['state']",
    "version",
    "['name']",
    "['message']",
    "['display_name']",
)

#: Expressions that *are* markup by design (a style or colour name going into a
#: tag), or are structurally incapable of carrying brackets.
SAFE_EXPRS = {
    "style",
    "color",
    "ok_color",
    "success_color",
    "mode_color",
    "verdict",
    "header",
    "mark",
}


def _print_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``console.print(...)`` / ``table.add_row(...)`` call node."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"print", "add_row"}:
            calls.append(node)
    return calls


def _is_escaped(expr: ast.AST) -> bool:
    """True when the interpolated expression is wrapped in ``escape(...)``."""
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "escape"
    )


def _offenders(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []
    for call in _print_calls(tree):
        # `markup=False` prints verbatim; nothing to escape.
        if any(kw.arg == "markup" for kw in call.keywords):
            continue
        for arg in call.args:
            for piece in ast.walk(arg):
                if not isinstance(piece, ast.FormattedValue):
                    continue
                text = ast.unparse(piece.value)
                if text in SAFE_EXPRS or _is_escaped(piece.value):
                    continue
                # Counts and lengths are ints; they cannot carry markup even
                # when the thing being counted has a risky-looking name.
                if text.startswith("len("):
                    continue
                if any(token in text.lower() for token in RISKY):
                    found.append((piece.lineno, text))
    return found


def test_no_unescaped_untrusted_interpolation_in_cli() -> None:
    offenders = _offenders(CLI_PATH.read_text())

    assert not offenders, (
        "Unescaped interpolation of untrusted text in cli.py — wrap in "
        "escape(), or print with markup=False:\n"
        + "\n".join(f"  cli.py:{line}: {expr}" for line, expr in offenders)
    )


def test_the_guard_actually_catches_something() -> None:
    """A guard that cannot fail is worse than no guard."""
    offenders = _offenders(
        'console.print(f"[red]Stop refused:[/red] {exc}")\n'
    )
    assert offenders, "the AST guard failed to flag a known-bad line"


def test_the_guard_accepts_the_escaped_form() -> None:
    offenders = _offenders(
        'console.print(f"[red]Stop refused:[/red] {escape(str(exc))}")\n'
    )
    assert not offenders


def test_the_guard_accepts_markup_false() -> None:
    offenders = _offenders('console.print(content, markup=False)\n')
    assert not offenders
