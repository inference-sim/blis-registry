"""Tests for the documentation hook, run against the committed coefficient sets.

The hook's job is to show the data exactly. These tests check the two ways it could fail
to: by leaving an entry out of a page, and by showing an entry differently from how it is
committed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import registry

ROOT = Path(__file__).resolve().parent.parent

# Loaded at import, because the parametrized tests below enumerate the sets.
registry.on_config({"config_file_path": str(ROOT / "mkdocs.yml")})


def _chomp(x):
    """Drop a block scalar's final newline, which depends on whether the excerpt ends the
    file and so carries no content."""
    if isinstance(x, str):
        return x.removesuffix("\n")
    if isinstance(x, list):
        return [_chomp(v) for v in x]
    if isinstance(x, dict):
        return {k: _chomp(v) for k, v in x.items()}
    return x


def all_sets():
    return list(registry._state["sets"].values())


def test_every_committed_set_is_loaded():
    on_disk = {yaml.safe_load(p.read_text())["name"] for p in (ROOT / "coefficients").glob("*.yaml")}
    assert on_disk == set(registry._state["sets"])


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_entry_appears_in_its_matrix_and_only_under_its_own_parts(s):
    name = s.name
    rows, _ = registry.matrix(s)
    placed = {}
    for row, cells in rows.items():
        for part, entries in cells.items():
            for e in entries:
                assert part in e.hardware, f"{e.name} placed under {part}, outside its scope"
                placed.setdefault(id(e), []).append((row, part))
    for e in s.entries:
        assert id(e) in placed, f"{name}:{e.line} {e.name} is missing from the matrix"
        if registry._fold(e)[0] != e.name:
            # a name carrying its part fills exactly that part's cell
            assert len(placed[id(e)]) == 1, f"{e.name} fills {placed[id(e)]}"


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_quoted_entry_parses_back_to_the_committed_entry(s):
    for e in s.entries:
        part = e.hardware[0]
        matches = [x for x in s.entries if x.name == e.name and part in x.hardware]
        if len(matches) != 1:
            continue  # the directive refuses ambiguous quotes; nothing to compare
        html = registry.render_entry(f"{s.name} {e.name} {part}")
        quoted = re.search(r"```yaml[^\n]*\n(.*?)\n```", html, re.S).group(1)
        assert _chomp(yaml.safe_load(quoted)) == _chomp([{e.name: e.body}]), f"{s.name}:{e.line}"


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_rendered_value_reads_back_as_the_committed_number(s):
    for e in s.entries:
        shown = registry._fmt_value(e.body["value"], e.value_text)
        assert float(shown.replace(",", "")) == float(e.body["value"]), f"{s.name}:{e.line}"
        assert shown.replace(",", "") == e.value_text, f"{s.name}:{e.line}"


def test_census_totals_match_the_files():
    total = sum(len(s.entries) for s in all_sets())
    assert registry._stat("total") == f"{total:,}"
    html = registry.render_census("")
    assert f"<strong>{total:,}</strong>" in html


@pytest.mark.parametrize("cite, expected", [
    ("NVIDIA AISimulate systems/data/h200_sxm/attention/vllm/0.25.0/generation_attention_perf.parquet",
     "AISimulate sweep, vllm lane"),
    ("NVIDIA AISimulate systems/data/a100_sxm/comm/nccl/2.27.3/nccl_perf.parquet (NCCL 2.27.3)",
     "AISimulate sweep, nccl lane"),
    ("NVIDIA AISimulate systems/b200_sxm.yaml", "AISimulate system descriptor"),
    ("Extrapolated from this part's own measured (2, 4, 8) floors in NVIDIA AISimulate systems/data/gb300/comm/nccl/",
     "derived from other registry entries"),
])
def test_source_classification(cite, expected):
    e = registry.Entry("x", {"sources": [{"kind": "model", "cite": cite, "role": "primary"}]}, 1)
    assert registry.classify_source(e) == expected


def test_every_entry_has_a_source_class():
    for s in all_sets():
        for e in s.entries:
            assert registry.classify_source(e)
