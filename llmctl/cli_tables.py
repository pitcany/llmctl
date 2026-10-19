"""Tables whose identifiers stay whole while descriptions give way."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text


def _plain_len(cell: str, is_markup: bool) -> int:
    return Text.from_markup(cell).cell_len if is_markup else cell_len(cell)


def whole_id_table(
    console: Console,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    ids: Collection[str],
    title: str | None = None,
    caption: str | None = None,
    justify: Mapping[str, str] | None = None,
    styles: Mapping[str, str] | None = None,
    markup: Collection[str] = (),
) -> Table:
    """Build a table where identifier columns are never folded or cut.

    A folded identifier cannot be copied (the pasted text has line breaks in
    it), and a cut identifier is unusable (it no longer names anything). So
    every column named in ``ids`` is ``no_wrap`` with ``min_width`` equal to
    its longest cell. The other, descriptive columns are ``no_wrap`` with an
    ellipsis and share whatever width is left, so each row stays on one line.

    Cells are plain strings and are escaped here, so text such as ``[red]x[/red]``
    prints literally. Columns named in ``markup`` are the explicit opt-in: their
    cells are passed through as Rich markup (e.g. a coloured status).
    ``justify`` and ``styles`` map header to justification / column style.
    """
    justify = justify or {}
    styles = styles or {}
    table = Table(title=title, title_justify="left", caption=caption)
    widths = []
    for i, header in enumerate(headers):
        is_markup = header in markup
        widths.append(
            max([cell_len(header)] + [_plain_len(row[i], is_markup) for row in rows])
        )
    chrome = 3 * len(headers) + 1  # a border and two padding cells per column
    desc = [i for i, h in enumerate(headers) if h not in ids]
    leftover = console.width - chrome - sum(w for i, w in enumerate(widths) if i not in desc)
    # Share the leftover out: narrow descriptive columns keep their natural
    # width, and the rest split what remains.
    limits: dict[int, int] = {}
    pending = sorted(desc, key=lambda i: widths[i])
    for n, i in enumerate(pending):
        fair = max(4, leftover // (len(pending) - n))
        limits[i] = min(widths[i], fair)
        leftover -= limits[i]
    for i, header in enumerate(headers):
        kwargs: dict[str, object] = {"justify": justify.get(header, "left")}
        if header in styles:
            kwargs["style"] = styles[header]
        if i in limits:
            table.add_column(
                header, no_wrap=True, overflow="ellipsis", max_width=limits[i], **kwargs
            )
        else:
            table.add_column(header, no_wrap=True, min_width=widths[i], **kwargs)
    for row in rows:
        table.add_row(
            *(cell if h in markup else escape(cell) for h, cell in zip(headers, row, strict=True))
        )
    return table
