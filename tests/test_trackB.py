"""Tests for Track B — discrete optimization.
Run from the repo root: python -m pytest tests/test_trackB.py -v
"""
import pytest, json, math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackB'))
from run import (
    # knapsack
    knapsack_greedy, knapsack_dp, knapsack_bnb, knapsack_ilp,
    _validate_knapsack,
    # scheduling
    sched_greedy_eft, sched_dp, _validate_sched,
    # max-cut
    cut_value, _validate_cut, maxcut_brute, maxcut_greedy, maxcut_sa,
    # QUBO/Ising
    build_qubo_maxcut, qubo_energy, qubo_to_ising,
    ising_energy_hJ, qubo_brute, qubo_sa,
)

# ── shared small instances ────────────────────────────────────────
np.random.seed(7)
W5, V5, CAP = np.array([3,4,5,2,6]), np.array([4,5,3,7,2]), 8

# 3x3 adjacency (triangle, all weight 1)
W3 = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=int)


class TestKnapsack:
    def test_dp_exact_small(self):
        """Known optimum for W5=[3,4,5,2,6] V5=[4,5,3,7,2] CAP=8:
        Best is items 1,3 (w=4+2=6, v=5+7=12). Verified by enumeration."""
        _, v = knapsack_dp(W5, V5, CAP)
        assert v == 12

    def test_bnb_matches_dp(self):
        """B&B must match DP on the small instance."""
        _, vd = knapsack_dp(W5, V5, CAP)
        _, vb = knapsack_bnb(W5, V5, CAP)
        assert int(vb) == vd

    def test_ilp_matches_dp(self):
        _, vd = knapsack_dp(W5, V5, CAP)
        _, vi = knapsack_ilp(W5, V5, CAP)
        assert vi == vd

    def test_greedy_feasible(self):
        xg = knapsack_greedy(W5, V5, CAP)
        _validate_knapsack(xg, W5, V5, CAP)   # raises on infeasibility

    def test_all_methods_agree_on_larger_instance(self):
        """On the B1 instance all exact methods must agree."""
        np.random.seed(0)
        wts = np.random.randint(1, 15, 25)
        vals= np.random.randint(1, 20, 25)
        cap = 50
        _, vd = knapsack_dp(wts, vals, cap)
        _, vb = knapsack_bnb(wts, vals, cap)
        _, vi = knapsack_ilp(wts, vals, cap)
        assert int(vb) == vd
        assert vi      == vd

    def test_greedy_cannot_exceed_optimal(self):
        """Greedy value ≤ optimal (DP)."""
        xg = knapsack_greedy(W5, V5, CAP)
        _, vd = knapsack_dp(W5, V5, CAP)
        assert _validate_knapsack(xg, W5, V5, CAP) <= vd

    def test_dp_zero_capacity(self):
        _, v = knapsack_dp(W5, V5, 0)
        assert v == 0

    def test_dp_single_item(self):
        wts = np.array([3]); vals = np.array([5])
        _, v = knapsack_dp(wts, vals, 3)
        assert v == 5
        _, v = knapsack_dp(wts, vals, 2)
        assert v == 0


class TestScheduling:
    def test_dp_feasible(self):
        np.random.seed(1)
        s = np.random.randint(0, 20, 15)
        l = np.random.randint(1,  8, 15)
        e = s + l
        w = np.random.randint(1, 10, 15)
        sel, _ = sched_dp(s, e, w)
        _validate_sched(sel, s, e, w)   # raises on overlap

    def test_greedy_feasible(self):
        np.random.seed(1)
        s = np.random.randint(0, 20, 15)
        l = np.random.randint(1,  8, 15)
        e = s + l
        w = np.random.randint(1, 10, 15)
        sel = sched_greedy_eft(s, e, w)
        _validate_sched(sel, s, e, w)

    def test_dp_known_optimum(self):
        """Two non-overlapping jobs [0,3) and [5,8) with weights 10 and 1.
        Both fit so DP picks both: total = 11."""
        s = np.array([0, 5])
        e = np.array([3, 8])
        w = np.array([10, 1])
        sel, v = sched_dp(s, e, w)
        assert v == 11   # both jobs fit (non-overlapping)
        assert 0 in sel and 1 in sel

    def test_dp_cannot_exceed_total_weight(self):
        np.random.seed(1)
        s = np.random.randint(0, 20, 15)
        e = s + np.random.randint(1, 8, 15)
        w = np.random.randint(1, 10, 15)
        _, v = sched_dp(s, e, w)
        assert v <= int(w.sum())


class TestMaxCut:
    def test_brute_force_triangle(self):
        """Triangle with all-1 weights: max cut = 2 (one node vs two)."""
        _, v = maxcut_brute(W3)
        assert v == 2

    def test_cut_symmetry(self):
        x = np.array([1, 0, 1])
        _validate_cut(x, W3)   # should not raise

    def test_greedy_feasible(self):
        x = maxcut_greedy(W3)
        assert x.shape == (3,)
        assert set(x.tolist()).issubset({0, 1})

    def test_sa_matches_brute_on_triangle(self):
        """SA with enough iterations should find the optimum on a 3-node graph."""
        _, v = maxcut_sa(W3, seed=0, n_iter=5000)
        _, v_opt = maxcut_brute(W3)
        assert v == v_opt

    def test_cut_value_empty(self):
        """All nodes in S → cut = 0."""
        x = np.ones(3, int)
        assert cut_value(x, W3) == 0

    def test_cut_value_manual(self):
        """Node 0 alone in S, triangle weight 1 each: cut should be 2 (edges 0-1, 0-2)."""
        x = np.array([1, 0, 0])
        assert cut_value(x, W3) == 2


class TestQUBO:
    def test_qubo_energy_match_cut(self):
        """QUBO energy at x* must equal -max_cut for unconstrained Max-Cut."""
        _, v_bf = maxcut_brute(W3)
        Q = build_qubo_maxcut(W3)
        xbq, vbq = qubo_brute(Q)
        # QUBO min energy = -max_cut
        assert abs(vbq - (-v_bf)) < 1e-9

    def test_ising_conversion_consistent(self):
        """Ising energy via (h,J) + const must equal QUBO energy for any x."""
        Q = build_qubo_maxcut(W3)
        h, J, c = qubo_to_ising(Q)
        rng = np.random.default_rng(99)
        for _ in range(20):
            x = rng.integers(0, 2, 3)
            s = 1 - 2 * x
            assert abs(ising_energy_hJ(h, J, s) + c - qubo_energy(Q, x)) < 1e-9

    def test_penalty_forces_cardinality(self):
        """With large lambda the constrained QUBO brute-force must satisfy |S|=k."""
        Q_con = build_qubo_maxcut(W3, lam=100.0, k=1)
        xbqc, _ = qubo_brute(Q_con)
        assert int(xbqc.sum()) == 1

    def test_qubo_sa_feasible(self):
        """SA result is a binary vector."""
        Q = build_qubo_maxcut(W3)
        x, _ = qubo_sa(Q, seed=0, n_iter=1000)
        assert set(x.tolist()).issubset({0, 1})

    def test_qubo_brute_matches_brute_cut_on_7node(self):
        """The QUBO brute-force on B3 graph must find the same cut as Max-Cut brute-force."""
        edges = [(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(5,6),(0,5),(1,6)]
        rng = np.random.default_rng(2)
        W7 = np.zeros((7,7), dtype=int)
        for i,j in edges:
            w = int(rng.integers(1,6))
            W7[i,j]=w; W7[j,i]=w
        _, v_mc = maxcut_brute(W7)
        Q7 = build_qubo_maxcut(W7)
        xbq7, _ = qubo_brute(Q7)
        qubo_cut = cut_value(xbq7, W7)
        assert qubo_cut == v_mc or cut_value(1 - xbq7, W7) == v_mc
