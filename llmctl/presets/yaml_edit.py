"""Line-based editing of preset YAML that keeps comments and untouched keys."""

from __future__ import annotations

import re
from typing import Any

import yaml

_KEY_RE = re.compile(r"^([A-Za-z_][\w.-]*):(?:[ \t]|$)")


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def dump_mapping(data: dict[str, Any]) -> str:
    """Full dump of ``data`` (the fallback and the new-file format)."""
    if not data:
        return ""
    return yaml.safe_dump(_norm(data), sort_keys=False, default_flow_style=False, width=10**6)


def _spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Map top-level key -> [start, end) line range (key line + continuation)."""
    starts: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = _KEY_RE.match(line)
        if m:
            starts.append((m.group(1), i))
    spans: dict[str, tuple[int, int]] = {}
    for n, (key, start) in enumerate(starts):
        limit = starts[n + 1][1] if n + 1 < len(starts) else len(lines)
        end = start + 1
        for j in range(start + 1, limit):
            line = lines[j]
            if line[:1] in (" ", "\t") and line.strip() and not line.lstrip().startswith("#"):
                end = j + 1
            elif line.startswith("- ") or line.rstrip() == "-":
                end = j + 1
            elif not line.strip() or line.lstrip().startswith("#"):
                continue
            else:
                break
        spans[key] = (start, end)
    return spans


def _trailing_comment(line: str, key: str, old_value: Any) -> str:
    body = line.rstrip("\r\n")
    pos = body.find(" #")
    while pos != -1:
        try:
            if _norm(yaml.safe_load(body[:pos].split(":", 1)[1])) == _norm(old_value):
                start = pos
                while start > 0 and body[start - 1] in " \t":
                    start -= 1
                return body[start:]
        except yaml.YAMLError:
            pass
        pos = body.find(" #", pos + 1)
    return ""


def edit_preset_text(text: str, desired: dict[str, Any]) -> str:
    """Return ``text`` edited so it loads to ``desired``; comments are kept.

    Raises ``ValueError`` when the line edit cannot reproduce ``desired``
    (the caller then falls back to a full dump).
    """
    old = yaml.safe_load(text) or {}
    if not isinstance(old, dict):
        raise ValueError("not a mapping")
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    spans = _spans(lines)
    replacements: list[tuple[int, int, list[str]]] = []
    for key, (start, end) in spans.items():
        if key not in desired:
            replacements.append((start, end, []))
            continue
        new = _norm(desired[key])
        same_bool = isinstance(old.get(key), bool) == isinstance(new, bool)
        if key in old and _norm(old[key]) == new and same_bool:
            continue
        dumped = dump_mapping({key: desired[key]}).splitlines(keepends=True)
        if len(dumped) == 1 and end - start == 1:
            comment = _trailing_comment(lines[start], key, old.get(key))
            eol = "\r\n" if lines[start].endswith("\r\n") else "\n"
            dumped = [dumped[0].rstrip("\n") + comment + eol]
        replacements.append((start, end, dumped))
    for start, end, new_lines in sorted(replacements, reverse=True):
        lines[start:end] = new_lines
    added = [k for k in desired if k not in spans]
    if added:
        lines.extend(dump_mapping({k: desired[k] for k in added}).splitlines(keepends=True))
    result = "".join(lines)
    loaded = yaml.safe_load(result) or {}
    if _norm(loaded) != _norm(desired) or set(loaded) != set(desired):
        raise ValueError("line edit did not reproduce the mapping")
    return result
