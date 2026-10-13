"""Entry-level rewriting of a committed coefficient set, byte-preserving elsewhere.

The sets are hand-ordered YAML with comments, so a writer that owns a few entries must
replace exactly those and pass every other byte through. An entry is identified by its
NAME and its SCOPE LINE, because one name legitimately appears once per part (and, for a
model-scoped entry, once per model).
"""

from __future__ import annotations

import re

ENTRY = re.compile(r"^  - ([a-z_0-9]+):$")
SCOPE = re.compile(r"^      scope: (\{.*\})$")


def split(text: str) -> list[list[str]]:
    """The file as blocks: each entry with its indented body, and every other line alone."""
    out: list[list[str]] = []
    for line in text.split("\n"):
        if ENTRY.match(line):
            out.append([line])
        elif out and ENTRY.match(out[-1][0]) and line.startswith("    "):
            out[-1].append(line)
        else:
            out.append([line])
    return out


def join(blocks: list[list[str]]) -> str:
    return "\n".join(line for b in blocks for line in b)


def name_of(block: list[str]) -> str | None:
    m = ENTRY.match(block[0])
    return m.group(1) if m else None


def scope_of(block: list[str]) -> str | None:
    """The entry's scope exactly as written, e.g. '{hardware: [h200]}'."""
    for line in block:
        m = SCOPE.match(line)
        if m:
            return m.group(1)
    return None


def find(blocks: list[list[str]], name: str, scope: str) -> int | None:
    for j, b in enumerate(blocks):
        if name_of(b) == name and scope_of(b) == scope:
            return j
    return None


def upsert(blocks: list[list[str]], entry: list[str], after) -> None:
    """Replace the entry with this (name, scope) in place, or insert it after the last
    block `after(block)` accepts. Refuses rather than appending somewhere arbitrary."""
    name, scope = name_of(entry), scope_of(entry)
    if name is None or scope is None:
        raise SystemExit(f"not a scoped entry: {entry[0]!r}")
    j = find(blocks, name, scope)
    if j is not None:
        blocks[j] = entry
        return
    anchors = [k for k, b in enumerate(blocks) if after(b)]
    if not anchors:
        raise SystemExit(f"{name} {scope}: no anchor entry to place it after")
    blocks.insert(anchors[-1] + 1, entry)
