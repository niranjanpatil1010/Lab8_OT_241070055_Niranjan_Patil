"""Tests for Track D — Submodular, Parallel, and Distributed Optimization.
Run from repo root: python -m pytest tests/test_trackD.py -v
"""
import pytest, math
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackD'))
from run import (
    coverage, check_submodularity, greedy_coverage, brute_force_coverage,
    lpt_makespan, random_makespan, simulate_work_stealing,
    cpu_task, local_sgd, fedavg, centralized_sgd,
    sigmoid, mt_forward, mt_grads, mgda_weights, train_mt,
    SETS, UNIVERSE_SIZE, BUDGET_K
)

class TestSubmodularCoverage:
    def test_coverage_empty(self):
        assert coverage([]) == 0

    def test_coverage_monotone(self):
        # Adding a set cannot decrease coverage
        c1 = coverage([0])
        c2 = coverage([0, 1])
        assert c2 >= c1

    def test_greedy_within_guarantee(self):
        g_sel, g_val, gains = greedy_coverage(BUDGET_K)
        opt_sel, opt_val = brute_force_coverage(BUDGET_K)
        assert opt_val > 0
        approx_ratio = g_val / opt_val
        bound = 1.0 - 1.0 / math.e
        assert approx_ratio >= bound - 1e-9

    def test_submodularity_diminishing_returns(self):
        n_trials, n_violated = check_submodularity(SETS, n_trials=300, seed=123)
        assert n_violated == 0


class TestParallelScheduling:
    def test_lpt_makespan_bound(self):
        # LPT must achieve <= 4/3 * OPT (or lower bound sum/m)
        jt = [10, 8, 7, 6, 5, 4, 3, 2]
        m = 3
        ms, loads = lpt_makespan(jt, m)
        assert sum(loads) == sum(jt)
        lb = math.ceil(sum(jt) / m)
        assert ms >= lb

    def test_work_stealing_feasible(self):
        jt = [5, 12, 7, 3, 9, 4]
        m = 2
        ms = simulate_work_stealing(jt, m, seed=42)
        assert ms >= math.ceil(sum(jt) / m)

    def test_cpu_task_exact_output(self):
        # sin^2(x) + cos^2(x) == 1, so sum == chunk_size
        size = 1000
        val = cpu_task((size, 42))
        assert abs(val - size) < 1e-5


class TestDistributedFedAvg:
    def test_local_sgd_step(self):
        w0 = np.zeros(3)
        X = np.array([[1.0, 0.0, 0.0]])
        y = np.array([2.0])
        w = local_sgd(w0, X, y, lr=0.1, steps=1, seed=0)
        # grad = X[0] * (0 - 2) = -2 * [1, 0, 0]
        # w = 0 - 0.1 * (-2) = 0.2
        assert abs(w[0] - 0.2) < 1e-6

    def test_fedavg_convergence(self):
        w0 = np.zeros(10)
        w_5 = fedavg(w0, rounds=5, local_steps=5, lr=0.01)
        w_20 = fedavg(w0, rounds=20, local_steps=5, lr=0.01)
        # More rounds should reduce error on average
        from run import X_fed, y_fed, opt_mse_f
        mse_5 = np.mean((X_fed @ w_5 - y_fed)**2) - opt_mse_f
        mse_20 = np.mean((X_fed @ w_20 - y_fed)**2) - opt_mse_f
        assert mse_20 < mse_5


class TestMultiTaskOptimization:
    def test_mgda_analytical_weights(self):
        # Two orthogonal gradients
        g1 = np.array([1.0, 0.0])
        g2 = np.array([0.0, 1.0])
        a1, a2 = mgda_weights(g1, g2)
        assert abs(a1 - 0.5) < 1e-6
        assert abs(a2 - 0.5) < 1e-6

    def test_mgda_collinear_opposite(self):
        # Conflicting opposite gradients
        g1 = np.array([1.0, 0.0])
        g2 = np.array([-1.0, 0.0])
        a1, a2 = mgda_weights(g1, g2)
        assert abs(a1 - 0.5) < 1e-6

    def test_train_mt_reproducibility(self):
        mse1, acc1 = train_mt("equal", n_iter=50, seed=42)
        mse2, acc2 = train_mt("equal", n_iter=50, seed=42)
        assert abs(mse1 - mse2) < 1e-9
        assert abs(acc1 - acc2) < 1e-9
