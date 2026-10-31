"""Tests for Track G helpers (no QEMU needed). The kernel itself is verified by `usertests` under QEMU (see REPORT.md)."""
import os, sys, subprocess
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trackG"))
from run import parse, metrics, jain

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "..", "trackG", "xv6_patch.diff")

def test_parse_and_prop_metrics():
    txt = ("RESULT wl=prop role=0 pid=4 tickets=30 iters=10 cputicks=50 sched=50\n"
           "RESULT wl=prop role=1 pid=5 tickets=20 iters=7 cputicks=33 sched=33\n"
           "RESULT wl=prop role=2 pid=6 tickets=10 iters=3 cputicks=17 sched=17\n")
    m = metrics(parse(txt))
    assert abs(m["prop_share_t30"] - 0.5) < 1e-9 and abs(m["prop_share_t10"] - 0.17) < 1e-9
    assert m["prop_L1_error_vs_tickets"] < 0.01

def test_jain_bounds():
    assert abs(jain([5, 5, 5, 5]) - 1.0) < 1e-12
    assert abs(jain([1, 0, 0, 0]) - 0.25) < 1e-12

def test_kernel_prng_replica_matches_ticket_ratio():
    """Replica of the kernel's xorshift32 + 'counter > winner' draw: long-run shares must match tickets."""
    state, tix, wins = 2463534242, [30, 20, 10], [0, 0, 0]
    for _ in range(60000):
        state ^= (state << 13) & 0xFFFFFFFF; state ^= state >> 17; state ^= (state << 5) & 0xFFFFFFFF
        w, c = state % 60, 0
        for i, t in enumerate(tix):
            c += t
            if c > w: wins[i] += 1; break
    sh = np.array(wins) / sum(wins)
    assert np.abs(sh - np.array([30, 20, 10]) / 60).max() < 0.01

def test_patch_present_and_touches_scheduler():
    d = open(PATCH).read()
    assert "SCHED_LOTTERY" in d and "settickets" in d and "user/schedbench.c" in d

def test_patch_applies_to_pristine_xv6():
    xv6 = os.environ.get("XV6_PRISTINE", "/home/claude/work/xv6-riscv")
    if not os.path.isdir(os.path.join(xv6, ".git")):
        pytest.skip("pristine xv6-riscv clone not found (set XV6_PRISTINE)")
    r = subprocess.run(["git", "apply", "--check", os.path.abspath(PATCH)], cwd=xv6)
    assert r.returncode == 0
