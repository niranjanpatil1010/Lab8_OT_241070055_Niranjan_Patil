"""
Track J — Software-engineering optimization: test-suite minimization & prioritization (Chapter-6 ILP / greedy).
Codebase under optimization: THIS repository (the 'optimize-anything' source in track*/ and its pytest suite).
All inputs are MEASURED: per-test wall time and per-test function coverage come from trackJ/profile_suite.py
(results/trackJ_profile.json); the selected subsets are then RE-RUN and re-measured independently.

Decision variables  x_i in {0,1}: run test i.
Problem A (minimization): min sum t_i x_i  s.t. every source function covered by the full suite stays covered.
Problem B (CI budget)   : max #functions covered  s.t. sum t_i x_i <= budget.
Methods: ILP (scipy.optimize.milp / HiGHS), greedy set cover (ratio rule), naive baselines (run all / random order).
Run from repo root after profiling:  python3 trackJ/profile_suite.py && python3 trackJ/run.py
"""
import os, sys, time, json, statistics, subprocess
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE = os.path.join(ROOT, "results", "trackJ_profile.json")

def load_profile(path=PROFILE):
    """-> (TEST_SUITE=[(id, seconds, [requirement idx])], CAPABILITIES=[function names], raw profile)"""
    prof = json.load(open(path))
    req = prof["requirements"]; idx = {r: i for i, r in enumerate(req)}
    suite = [(t["id"], t["median_s"], sorted(idx[f] for f in t["functions"])) for t in prof["tests"]]
    # a test with 0 timed cost would break the ratio rule; clamp to 1 ms (timer resolution of junit)
    suite = [(n, max(d, 1e-3), r) for n, d, r in suite]
    return suite, req, prof

TEST_SUITE, CAPABILITIES, _PROFILE = load_profile()

# =================================================================
# 2. Problem A: Test-Suite Minimization (Minimum Cost Set Cover)
# =================================================================
def solve_minimization_greedy(tests, n_reqs):
    """Greedy Set Cover heuristic: pick test with highest uncovered_reqs / runtime."""
    covered = set()
    selected = []
    tot_time = 0.0

    while len(covered) < n_reqs:
        best_ratio = -1.0
        best_idx = -1
        for idx, (name, dur, reqs) in enumerate(tests):
            if idx in selected: continue
            new_cov = len(set(reqs) - covered)
            if new_cov > 0:
                ratio = new_cov / dur
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = idx

        if best_idx == -1:
            break  # cannot cover any more
        selected.append(best_idx)
        covered |= set(tests[best_idx][2])
        tot_time += tests[best_idx][1]

    return selected, tot_time, len(covered)

def solve_minimization_ilp(tests, n_reqs):
    """
    Exact Minimum-Cost Set Cover via ILP (scipy.optimize.milp):
      Min sum t_i * x_i
      s.t. sum_{i: j in R_i} x_i >= 1 for each requirement j in {0..M-1}
      x_i in {0, 1}
    """
    n_tests = len(tests)
    c = np.array([t[1] for t in tests])  # runtimes

    # Constraint matrix A: M x N
    A = np.zeros((n_reqs, n_tests))
    for i, (_, _, reqs) in enumerate(tests):
        for r in reqs:
            A[r, i] = 1.0

    lhs = np.ones(n_reqs)
    rhs = np.full(n_reqs, np.inf)
    constraints = LinearConstraint(A, lhs, rhs)
    integrality = np.ones(n_tests)
    bounds = Bounds(0, 1)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success:
        sol = np.round(res.x).astype(int)
        selected = [i for i in range(n_tests) if sol[i] == 1]
        tot_time = sum(tests[i][1] for i in selected)
        covered = set().union(*[tests[i][2] for i in selected])
        return selected, tot_time, len(covered)
    return None, None, None

# =================================================================
# 3. Problem B: Test Prioritization under Strict Time Budget (Knapsack)
# =================================================================
def solve_budget_greedy(tests, budget_s):
    # Sort by coverage / duration
    sorted_idx = sorted(range(len(tests)), key=lambda i: -(len(tests[i][2]) / tests[i][1]))
    selected = []
    spent = 0.0
    covered = set()
    for idx in sorted_idx:
        dur = tests[idx][1]
        if spent + dur <= budget_s:
            selected.append(idx)
            spent += dur
            covered |= set(tests[idx][2])
    return selected, spent, len(covered)

def solve_budget_ilp(tests, budget_s, n_reqs):
    """
    Maximize unique requirements covered within time budget:
      Max sum y_j
      s.t. y_j <= sum_{i: j in R_i} x_i  (y_j can only be 1 if at least one covering test selected)
           sum t_i x_i <= budget_s
      x_i in {0, 1}, y_j in {0, 1}
    """
    n_tests = len(tests)
    num_vars = n_tests + n_reqs  # x_i (0..N-1) + y_j (N..N+M-1)

    # Maximize sum y_j <=> Minimize -sum y_j
    c = np.zeros(num_vars)
    for j in range(n_reqs):
        c[n_tests + j] = -1.0

    # Constraints:
    # 1. Budget: sum t_i x_i <= budget_s
    A_budget = np.zeros((1, num_vars))
    for i in range(n_tests):
        A_budget[0, i] = tests[i][1]
    lhs_budget = np.array([0.0])
    rhs_budget = np.array([budget_s])

    # 2. Coverage link: y_j - sum_{i: j in R_i} x_i <= 0
    A_link = np.zeros((n_reqs, num_vars))
    for j in range(n_reqs):
        A_link[j, n_tests + j] = 1.0  # +y_j
        for i, (_, _, reqs) in enumerate(tests):
            if j in reqs:
                A_link[j, i] = -1.0    # -x_i
    lhs_link = np.full(n_reqs, -np.inf)
    rhs_link = np.zeros(n_reqs)

    A_full = np.vstack([A_budget, A_link])
    lhs = np.concatenate([lhs_budget, lhs_link])
    rhs = np.concatenate([rhs_budget, rhs_link])

    constraints = LinearConstraint(A_full, lhs, rhs)
    integrality = np.ones(num_vars)
    bounds = Bounds(0, 1)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success:
        sol = np.round(res.x).astype(int)
        selected = [i for i in range(n_tests) if sol[i] == 1]
        spent = sum(tests[i][1] for i in selected)
        covered = set().union(*[tests[i][2] for i in selected]) if selected else set()
        return selected, spent, len(covered)
    return None, None, None

# =================================================================
# Main Execution
# =================================================================
def wall(node_ids, repeats=3):
    """Real wall-clock of `pytest <ids>` (process start-up included), median of repeats."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--ignore=tests/test_trackJ.py"]
                       + node_ids, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts

def run_trackJ():
    sys.path.insert(0, os.path.dirname(__file__))
    from profile_suite import covered_functions
    n_reqs = len(CAPABILITIES); T = TEST_SUITE
    all_ids = [t[0] for t in T]; total = sum(t[1] for t in T)
    print(f"codebase: {_PROFILE['loc_source']} source lines; {len(T)} tests; {n_reqs} functions covered; sum(test time)={total:.2f}s")

    # ---- Problem A
    sel_g, tg, cg = solve_minimization_greedy(T, n_reqs)
    t0 = time.perf_counter(); sel_i, ti, ci = solve_minimization_ilp(T, n_reqs); ilp_solve_s = time.perf_counter() - t0
    full_wall, full_runs = wall([ "tests" ])
    res = {"codebase": {"name": "optimize-anything (this repo)", "source_lines_track_dirs": _PROFILE["loc_source"],
                        "tests_profiled": len(T), "functions_covered_by_full_suite": n_reqs,
                        "sum_of_test_times_s": round(total, 3), "full_suite_wall_s": round(full_wall, 3),
                        "full_suite_wall_runs_s": [round(x, 3) for x in full_runs]}}
    A = {"naive_full_suite": {"tests": len(T), "predicted_s": round(total, 3), "wall_s": round(full_wall, 3), "coverage": 1.0}}
    for name, sel, pred in (("greedy_set_cover", sel_g, tg), ("ilp_exact_set_cover", sel_i, ti)):
        ids = [T[i][0] for i in sel]
        w, runs = wall(ids)
        cov = covered_functions(ids)                      # independent validation by coverage.py
        A[name] = {"tests": len(sel), "predicted_s": round(pred, 3), "wall_s": round(w, 3),
                   "wall_runs_s": [round(x, 3) for x in runs],
                   "measured_speedup_vs_full": round(full_wall / w, 2),
                   "measured_time_reduction_pct": round(100 * (full_wall - w) / full_wall, 1),
                   "functions_covered_independent": len(cov), "functions_required": n_reqs,
                   "coverage_retained": len(cov & set(CAPABILITIES)) / n_reqs, "selected": ids}
    A["ilp_solver_time_s"] = round(ilp_solve_s, 4)
    res["problem_a_minimization"] = A

    # ---- Problem B: budget = 10% of suite time
    budget = 0.10 * total
    rnd = []
    for s in range(10):
        rng = np.random.default_rng(s); sp, cv = 0.0, set()
        for i in rng.permutation(len(T)):
            if sp + T[i][1] <= budget:
                sp += T[i][1]; cv |= set(T[i][2])
        rnd.append(len(cv))
    sg, spg, cvg = solve_budget_greedy(T, budget)
    si, spi, cvi = solve_budget_ilp(T, budget, n_reqs)
    ver = {}
    for name, sel in (("greedy", sg), ("ilp", si)):
        ids = [T[i][0] for i in sel]
        ver[name] = len(covered_functions(ids) & set(CAPABILITIES)) if ids else 0
    res["problem_b_budget"] = {"budget_s": round(budget, 3), "budget_fraction_of_suite": 0.10,
        "random_10seeds_functions_mean": round(float(np.mean(rnd)), 2), "random_std": round(float(np.std(rnd)), 2),
        "greedy": {"functions": cvg, "spent_s": round(spg, 3), "independent_check": ver["greedy"], "tests": len(sg)},
        "ilp": {"functions": cvi, "spent_s": round(spi, 3), "independent_check": ver["ilp"], "tests": len(si)},
        "of_total_functions": n_reqs}
    res["cost_analysis"] = {
        "correctness_risk": "Function-level coverage retention does NOT imply equal fault detection: dropped tests "
                            "assert different properties on the same functions (not measured here: no mutation testing).",
        "readability": "No test code changed; adds a profile + selection script.",
        "maintainability": "Profile (timing + coverage) must be regenerated when code/tests change; stale profile = silent coverage loss."}
    return res

if __name__ == "__main__":
    print("=== Track J — Software Engineering Optimization (measured) ===")
    results = run_trackJ()
    out_path = os.path.join(ROOT, "results", "trackJ_results.json")
    json.dump(results, open(out_path, "w"), indent=1, default=float)
    print(json.dumps(results, indent=1, default=float)); print("[OK] saved", out_path)
