"""Tests for Track I — Quantum Algorithms and QML Advantage.
Run from repo root: python -m pytest tests/test_trackI.py -v
"""
import pytest, math
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackI'))
from run import (
    run_i1_deutsch_jozsa, run_i2_bernstein_vazirani, run_i3_grover_vs_linear,
    run_i4_qft, run_i5_qpe, run_i6_shor, run_i7_quantum_annealing,
    run_i8_noise_model, run_i9_quantum_kernel_vs_classical
)

class TestTrackIAlgorithms:
    def test_deutsch_jozsa_distinction(self):
        res = run_i1_deutsch_jozsa()
        # Constant oracle must give all zeros; balanced must give 0 prob of all zeros
        assert res["constant_oracle"]["verdict"] == "Constant"
        assert res["balanced_oracle"]["verdict"] == "Balanced"
        assert res["queries_required"] == 1

    def test_bernstein_vazirani_recovery(self):
        res = run_i2_bernstein_vazirani()
        assert res["recovered_secret"] == res["target_secret"]
        assert res["success_probability"] == 1.0

    def test_grover_vs_linear_query_advantage(self):
        res = run_i3_grover_vs_linear()
        # For N=64, Grover takes 6 queries while classical linear search takes 62
        res_64 = res["N_64"]
        assert res_64["grover_queries"] < res_64["classical_queries"]
        assert res_64["query_speedup_ratio"] > 5.0
        # Honest negative result check: classical CPU execution is faster than statevector simulation
        assert res_64["classical_linear_time_s"] < res_64["grover_simulator_time_s"]

    def test_qft_unitary_match(self):
        res = run_i4_qft()
        # Hand-written QFT circuit matrix must exactly match classical discrete Fourier transform matrix
        assert res["max_absolute_error_vs_classical_fft"] < 1e-6
        assert abs(res["fidelity"] - 1.0) < 1e-6

    def test_qpe_exact_phase_recovery(self):
        res = run_i5_qpe()
        assert res["estimated_phase"] == res["true_phase"]
        assert res["peak_probability"] == 1.0
        assert res["accuracy_error"] == 0.0

    def test_shor_factoring_correctness(self):
        res = run_i6_shor()
        assert res["factoring_successful"] is True
        assert set(res["gcd_factors"]) == {3, 5}

    def test_quantum_annealing_ground_state(self):
        res = run_i7_quantum_annealing()
        assert res["found_ground_state"] is True
        assert res["annealed_found_energy"] == res["ground_state_energy_exact"]

    def test_hardware_noise_reduces_bv_fidelity(self):
        res = run_i8_noise_model()
        p0 = res["depolarizing_p_0.0"]["success_probability"]
        p2 = res["depolarizing_p_0.2"]["success_probability"]
        assert p0 == 1.0
        assert p2 < p0
        assert res["depolarizing_p_0.2"]["degradation_pct"] > 40.0

    def test_quantum_kernel_feasibility(self):
        res = run_i9_quantum_kernel_vs_classical()
        # Both models must attain valid non-trivial accuracy (> 65%)
        assert res["quantum_kernel_svm"]["accuracy"] >= 0.70
        assert res["classical_rbf_svm"]["accuracy"] >= 0.85
