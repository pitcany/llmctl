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
    # Bare identifiers too, not just attribute/subscript forms: a local named
    # `message` carries exactly the same subprocess text as `result.message`.
    "message",
    "name",
    "output",
    "stderr",
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
        # Only `markup=False` prints verbatim. A call passing `markup=True`
        # (Rich's default) must still be checked -- skipping on the mere
        # presence of the keyword would let an explicit `markup=True` through.
        if any(
            kw.arg == "markup"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
            for kw in call.keywords
        ):
            continue
        for arg in call.args:
            # A direct argument is rendered as markup just like an f-string
            # piece: `console.print(str(exc))` and `add_row(entry["name"])`
            # were invisible to an f-string-only walk.
            # An argument already wrapped in escape() is done: descending into
            # it would demand escaping its innards too, and escaping twice
            # renders a literal backslash.
            if _is_escaped(arg):
                continue
            candidates: list[tuple[int, ast.AST]] = []
            if not isinstance(arg, ast.JoinedStr | ast.Constant):
                candidates.append((getattr(arg, "lineno", 0), arg))
            candidates += [
                (piece.lineno, piece.value)
                for piece in ast.walk(arg)
                if isinstance(piece, ast.FormattedValue)
            ]
            for lineno, node in candidates:
                text = ast.unparse(node)
                if text in SAFE_EXPRS or _is_escaped(node):
                    continue
                # Counts and lengths are ints; they cannot carry markup even
                # when the thing being counted has a risky-looking name.
                if text.startswith("len("):
                    continue
                if any(token in text.lower() for token in RISKY):
                    found.append((lineno, text))
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


# --- gaps the review found in this guard --------------------------------------
#
# The first version of this check walked only f-string pieces and skipped any
# call carrying a `markup=` keyword regardless of its value. Both let real
# unescaped output through; these pin the fixes.


def test_a_bare_direct_argument_is_checked() -> None:
    """`console.print(str(exc))` has no f-string to walk into."""
    assert _offenders("console.print(str(exc))\n")


def test_a_direct_add_row_argument_is_checked() -> None:
    assert _offenders('table.add_row(entry.get("name", ""))\n')


def test_markup_true_is_not_a_free_pass() -> None:
    """Only `markup=False` prints verbatim; `markup=True` still renders."""
    assert _offenders('console.print(f"{exc}", markup=True)\n')


def test_a_bare_message_identifier_is_checked() -> None:
    """A local named `message` carries the same subprocess text as `.message`."""
    assert _offenders('console.print(f"  [cyan]{message}[/cyan]")\n')


def test_an_already_escaped_argument_is_not_descended_into() -> None:
    """Escaping twice renders a literal backslash, so stop at the wrapper."""
    assert not _offenders('table.add_row("Model", escape(f"{model.name} "))\n')
