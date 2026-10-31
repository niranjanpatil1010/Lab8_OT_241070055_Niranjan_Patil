"""Tests for Track J — Software Engineering Optimization.
Run from repo root: python -m pytest tests/test_trackJ.py -v
"""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackJ'))
from run import (
    solve_minimization_greedy, solve_minimization_ilp,
    solve_budget_greedy, solve_budget_ilp,
    TEST_SUITE, CAPABILITIES
)

class TestTrackJSoftwareOptimization:
    def test_minimization_ilp_covers_all_capabilities(self):
        n_reqs = len(CAPABILITIES)
        sel, tot_time, cov = solve_minimization_ilp(TEST_SUITE, n_reqs)
        assert cov == n_reqs  # 100% capability retention
        assert tot_time < sum(t[1] for t in TEST_SUITE)

    def test_minimization_ilp_beats_or_ties_greedy(self):
        n_reqs = len(CAPABILITIES)
        _, time_g, _ = solve_minimization_greedy(TEST_SUITE, n_reqs)
        _, time_ilp, _ = solve_minimization_ilp(TEST_SUITE, n_reqs)
        # ILP is exact, must be <= greedy
        assert time_ilp <= time_g + 1e-6

    def test_budget_prioritization_respects_time_budget(self):
        n_reqs = len(CAPABILITIES)
        budget = 0.50
        sel_g, spent_g, cov_g = solve_budget_greedy(TEST_SUITE, budget)
        sel_ilp, spent_ilp, cov_ilp = solve_budget_ilp(TEST_SUITE, budget, n_reqs)

        assert spent_g <= budget + 1e-5
        assert spent_ilp <= budget + 1e-5
        # ILP coverage should be >= greedy coverage
        assert cov_ilp >= cov_g

    def test_synthetic_toy_minimization_exact_match(self):
        # 3 requirements, 3 tests
        # Test 0: covers 0, 1 in 0.1s
        # Test 1: covers 1, 2 in 0.1s
        # Test 2: covers 0, 1, 2 in 0.3s
        toy_tests = [
            ("t0", 0.1, [0, 1]),
            ("t1", 0.1, [1, 2]),
            ("t2", 0.3, [0, 1, 2])
        ]
        sel, time_res, cov = solve_minimization_ilp(toy_tests, 3)
        assert cov == 3
        # Optimal is picking t0 and t1 (0.2s) rather than t2 (0.3s)
        assert set(sel) == {0, 1}
        assert abs(time_res - 0.2) < 1e-5
