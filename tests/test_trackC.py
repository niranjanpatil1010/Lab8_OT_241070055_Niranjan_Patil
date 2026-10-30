"""Tests for Track C — Quantum Optimization.
Run from repo root: python -m pytest tests/test_trackC.py -v
"""
import pytest, math
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackC'))
from run import (
    run_c1_qaoa, run_c2_vqe, run_c3_optimizers, run_c4_qnn,
    run_c5_finite_shots, run_c6_hardware_noise, run_c7_barren_plateaus,
    WMAT, CLASSICAL_OPT, N_QUBITS
)

class TestTrackCQAOA:
    def test_qaoa_structure_and_bounds(self):
        # Expected cut must be non-negative and <= total graph weight
        res = run_c1_qaoa()
        total_w = sum(WMAT[i, j] for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS))
        cut_p1 = res["qaoa_p1"]["expected_cut"]
        cut_p2 = res["qaoa_p2"]["expected_cut"]
        assert 0.0 < cut_p1 <= total_w
        assert 0.0 < cut_p2 <= total_w
        # Depth 2 should have at least comparable or better expected cut than depth 1
        assert cut_p2 >= cut_p1 - 0.5

    def test_qaoa_classical_comparison(self):
        res = run_c1_qaoa()
        # Verify classical baseline matches Track B brute force
        assert res["classical_optimal_cut"] == 33
        assert res["comparison_with_classical"]["classical_simulated_annealing"] == 33.0


class TestTrackCVQE:
    def test_vqe_approaches_exact_ground_state(self):
        res = run_c2_vqe()
        e_exact = res["exact_ground_state_energy"]
        e_vqe = res["vqe_final_energy"]
        # Variational principle: VQE energy must be >= exact ground state energy (up to numerical float precision)
        assert e_vqe >= e_exact - 1e-4
        # Relative error should be within reasonable bound (< 15%)
        assert res["relative_error_pct"] < 15.0


class TestTrackCOptimizers:
    def test_optimizers_converge_towards_minimum(self):
        res = run_c3_optimizers()
        e_min = res["exact_min_energy"]
        # All optimizers should reduce energy towards e_min = -2.0
        assert res["parameter_shift"]["final_cost"] <= 0.0
        assert res["spsa"]["final_cost"] <= 0.0
        assert res["qng"]["final_cost"] <= 0.0
        # QNG achieves exact ground state
        assert abs(res["qng"]["final_cost"] - e_min) < 1e-3


class TestTrackCFiniteShotsAndNoise:
    def test_shot_noise_variance_decreases(self):
        res = run_c5_finite_shots()
        std_50 = res["shots_50"]["std"]
        std_1000 = res["shots_1000"]["std"]
        # Higher shots must reduce measurement variance
        assert std_1000 < std_50
        assert res["shots_infinite_analytic"]["std"] == 0.0

    def test_hardware_noise_degrades_fidelity(self):
        res = run_c6_hardware_noise()
        val_ideal = res["depolarizing_p_0.0"]["expval_Z0_Z1"]
        val_noisy = res["depolarizing_p_0.2"]["expval_Z0_Z1"]
        assert abs(val_ideal - 1.0) < 1e-5
        # Noise strictly degrades expectation value
        assert val_noisy < val_ideal
        assert res["depolarizing_p_0.2"]["fidelity_degradation_pct"] > 30.0


class TestTrackCBarrenPlateaus:
    def test_gradient_variance_monotonic_decay(self):
        # As qubits grow from 2 to 8, gradient variance should strictly decay
        res = run_c7_barren_plateaus()
        var_2 = res["gradient_variances"]["qubits_2"]["variance"]
        var_4 = res["gradient_variances"]["qubits_4"]["variance"]
        var_8 = res["gradient_variances"]["qubits_8"]["variance"]
        assert var_2 > var_4 > var_8
