"""Tests for Track A. trackA/run.py is a script (it runs + writes results at import), so the tests do NOT import it;
they validate its saved output (results/trackA_results.json) against independently computed facts."""
import json, os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "trackA_results.json")
R = json.load(open(RES))
OPTS = ["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "Newton"]

def test_all_required_optimizers_reported_on_convex_and_nonconvex():
    for prob in ("convex_quadratic", "rosenbrock_nonconvex"):
        assert set(OPTS) <= set(R["deterministic"][prob])

def test_newton_solves_quadratic_in_one_step_independent_check():
    A = np.diag([1.0, 20.0]); x = np.array([5.0, 5.0])
    x_new = x - np.linalg.solve(A, A @ x)          # exact Newton step on 0.5 x'Ax
    assert np.allclose(x_new, 0, atol=1e-12)
    assert R["deterministic"]["convex_quadratic"]["Newton"]["iters"] <= 2

def test_rosenbrock_optimum_is_1_1_and_adam_newton_reach_it():
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    assert f(np.array([1.0, 1.0])) == 0.0
    for o in ("Adam", "Newton", "Momentum"):
        assert R["deterministic"]["rosenbrock_nonconvex"][o]["final_f"] < 1e-6

def test_plain_gd_is_slow_on_ill_conditioned_valley():
    assert R["deterministic"]["rosenbrock_nonconvex"]["GD"]["final_f"] > 1e-3   # honest negative result

def test_backprop_gradient_matches_finite_differences():
    assert R["backprop_mlp"]["max_relative_gradient_check_error"] < 1e-6
    # independent tiny check: logistic-regression gradient vs central differences
    rng = np.random.default_rng(0); X = rng.normal(size=(20, 3)); y = (rng.random(20) < .5).astype(float); w = rng.normal(size=3)
    L = lambda w: -np.mean(y * np.log(1 / (1 + np.exp(-X @ w))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-X @ w))))
    g = X.T @ (1 / (1 + np.exp(-X @ w)) - y) / 20
    num = np.array([(L(w + 1e-6 * e) - L(w - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
    assert np.allclose(g, num, atol=1e-7)

def test_hessian_vector_product_and_power_iteration():
    assert R["curvature"]["hvp_vs_exact_max_abs_err"] < 1e-8
    assert abs(R["curvature"]["top_eig_power_iter_via_HVP"] - R["curvature"]["top_eig_exact"]) < 1e-6

def test_saddle_behaviour_measured():
    L = R["landscape"]
    assert np.allclose(L["saddle_GD_from_exact_saddle"], 0, atol=1e-9)          # GD stuck at exact saddle
    assert np.linalg.norm(L["saddle_GD_with_noise"]) > 0.5                      # noise escapes it

def test_sgd_noise_floor_and_decay_help():
    n = R["noisy_objective_gap_to_optimum_loss"]
    assert n["SGD(bs=1) + lr decay"]["mean_gap"] < n["SGD(bs=1)"]["mean_gap"]
