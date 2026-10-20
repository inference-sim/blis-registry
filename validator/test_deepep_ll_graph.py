"""The DeepEP low-latency graph-mode pair is what its writer derives from CollectiveX.

Runs `fit_deepep_ll_graph.py --check` against $COLLECTIVEX_DATA, the directory of InferenceX
CollectiveX run bodies the entries cite; skipped when that data is not present, since it is
fetched, not vendored.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = os.environ.get("COLLECTIVEX_DATA", "")


@pytest.mark.skipif(not DATA or not any(Path(DATA).glob("cx_run_*.body")),
                    reason="CollectiveX run bodies not present ($COLLECTIVEX_DATA)")
def test_graph_mode_pair_reproduces_from_collectivex():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fit_deepep_ll_graph.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
        env=dict(os.environ, COLLECTIVEX_DATA=DATA),
    )
    assert r.returncode == 0, (
        "the committed moe_a2a_*_deepep_low_latency_graph entries are not what "
        f"fit_deepep_ll_graph.py derives.\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


@pytest.mark.skipif(not DATA or not any(Path(DATA).glob("cx_run_*.body")),
                    reason="CollectiveX run bodies not present ($COLLECTIVEX_DATA)")
def test_host_link_reproduces_from_collectivex():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fit_host_link.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
        env=dict(os.environ, COLLECTIVEX_DATA=DATA))
    assert r.returncode == 0, (
        "the committed host_link_bandwidth entries are not what fit_host_link.py derives.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
