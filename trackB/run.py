"""
Track B -- Discrete Optimization
numpy + scipy only, seeded. Writes ../results/trackB_results.json.

Problems covered:
  B1. 0/1 Knapsack (allocation)  -- Greedy, DP (exact), own B&B (exact), ILP (scipy.milp)
  B2. Weighted Interval Scheduling -- Greedy (earliest-finish), DP (exact)
  B3. Max-Cut on 7-node graph     -- Greedy, Brute-force (exact), SA (heuristic, 10 seeds)
  B4. QUBO / Ising formulation of Max-Cut -- classical SA on QUBO; spin variables;
      penalty for a cardinality constraint; verified against brute-force baseline.
"""
# -*- coding: utf-8 -*-
import json, math, time, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

RNG = np.random.default_rng(42)
R   = {}           # top-level results dict

# =================================================================
# B1. 0/1 KNAPSACK  (n=25 items, capacity W=50)
# =================================================================
np.random.seed(0)
N_K, W_K = 25, 50
weights = np.random.randint(1, 15, N_K)
values  = np.random.randint(1, 20, N_K)


def _validate_knapsack(x, wts, vals, W):
    """Check feasibility and return value; raises on violation."""
    x = np.asarray(x, dtype=int)
    assert x.shape == (len(wts),)
    assert np.all((x == 0) | (x == 1)), "binary constraint violated"
    assert wts @ x <= W, f"capacity violated: {wts @ x} > {W}"
    return int(vals @ x)


# B1-a  Greedy (by value/weight ratio)
def knapsack_greedy(wts, vals, W):
    idx = np.argsort(-vals / wts)   # descending ratio
    x, cap = np.zeros(len(wts), int), W
    for i in idx:
        if wts[i] <= cap:
            x[i] = 1
            cap -= wts[i]
    return x


# B1-b  DP (exact, O(nW) pseudo-polynomial)
def knapsack_dp(wts, vals, W):
    n = len(wts)
    dp = np.zeros((n + 1, W + 1), dtype=int)
    for i in range(1, n + 1):
        w, v = int(wts[i - 1]), int(vals[i - 1])
        dp[i] = dp[i - 1].copy()
        if w <= W:
            dp[i, w:] = np.maximum(dp[i, w:], dp[i - 1, :W - w + 1] + v)
    # back-track
    x = np.zeros(n, int)
    cap = W
    for i in range(n, 0, -1):
        if dp[i, cap] != dp[i - 1, cap]:
            x[i - 1] = 1
            cap -= int(wts[i - 1])
    return x, int(dp[n, W])


# B1-c  Branch-and-Bound (exact, best-first with LP relaxation upper bound)
def knapsack_bnb(wts, vals, W):
    n   = len(wts)
    idx = np.argsort(-vals / wts)
    ws  = wts[idx].astype(float)
    vs  = vals[idx].astype(float)
    best_val = [0]
    best_x   = [np.zeros(n, int)]

    def upper_bound(depth, remaining_cap, current_val):
        ub  = current_val
        cap = remaining_cap
        for j in range(depth, n):
            if ws[j] <= cap:
                cap -= ws[j]; ub += vs[j]
            else:
                ub += vs[j] * cap / ws[j]; break
        return ub

    def bnb(depth, remaining_cap, current_val, current_x):
        if depth == n or remaining_cap == 0:
            if current_val > best_val[0]:
                best_val[0] = current_val
                best_x[0]   = current_x.copy()
            return
        if upper_bound(depth, remaining_cap, current_val) <= best_val[0]:
            return
        if ws[depth] <= remaining_cap:
            current_x[depth] = 1
            bnb(depth + 1, remaining_cap - ws[depth],
                current_val + vs[depth], current_x)
            current_x[depth] = 0
        bnb(depth + 1, remaining_cap, current_val, current_x)

    bnb(0, float(W), 0.0, np.zeros(n, int))
    x_orig = np.zeros(n, int)
    for j in range(n):
        x_orig[idx[j]] = best_x[0][j]
    return x_orig, best_val[0]


# B1-d  ILP via scipy.milp (exact, HiGHS)
def knapsack_ilp(wts, vals, W):
    n    = len(wts)
    c    = -vals.astype(float)          # minimise -value
    bnds = Bounds(0, 1)
    A    = wts[None, :].astype(float)
    cons = LinearConstraint(A, -np.inf, float(W))
    res  = milp(c, constraints=cons, integrality=np.ones(n), bounds=bnds)
    if res.success:
        x = np.round(res.x).astype(int)
        return x, int(vals @ x)
    return None, None


print("=== B1. 0/1 Knapsack ===")
t0 = time.perf_counter(); xg = knapsack_greedy(weights, values, W_K); tg = time.perf_counter()-t0
t0 = time.perf_counter(); xd, vd = knapsack_dp(weights, values, W_K);  td = time.perf_counter()-t0
t0 = time.perf_counter(); xb, vb = knapsack_bnb(weights, values, W_K); tb = time.perf_counter()-t0
t0 = time.perf_counter(); xi, vi = knapsack_ilp(weights, values, W_K); ti = time.perf_counter()-t0

vg = _validate_knapsack(xg, weights, values, W_K)
_validate_knapsack(xd, weights, values, W_K)
_validate_knapsack(xb, weights, values, W_K)
if xi is not None: _validate_knapsack(xi, weights, values, W_K)

# independent cross-check: DP is exact, so B&B and ILP must match it
assert int(vb) == vd, f"B&B value {vb} != DP exact {vd}"
assert vi      == vd, f"ILP value {vi} != DP exact {vd}"

R["B1_knapsack"] = {
    "n_items": N_K, "capacity": W_K,
    "greedy":   {"value": vg, "time_s": round(tg, 5),
                 "gap_from_optimal_pct": round(100*(vd-vg)/vd, 2)},
    "dp_exact": {"value": vd, "time_s": round(td, 5)},
    "bnb_exact":{"value": int(vb), "time_s": round(tb, 5)},
    "ilp_exact":{"value": vi, "time_s": round(ti, 5)},
    "validation": "DP=B&B=ILP (all exact); greedy feasible; all verified by independent validator"
}
print(f"  Greedy={vg}  DP={vd}  B&B={int(vb)}  ILP={vi}")
print(f"  Times: greedy={tg:.4f}s  dp={td:.4f}s  bnb={tb:.4f}s  ilp={ti:.4f}s")


# =================================================================
# B2. WEIGHTED INTERVAL SCHEDULING  (n=15 jobs)
# =================================================================
np.random.seed(1)
N_J = 15
starts  = np.random.randint(0, 20, N_J)
lengths = np.random.randint(1,  8, N_J)
ends    = starts + lengths
wt_j    = np.random.randint(1, 10, N_J)


def _validate_sched(sel, s, e, wts):
    """Check no overlaps and return total weight."""
    for a, b in itertools.combinations(sel, 2):
        assert not (s[a] < e[b] and s[b] < e[a]), f"jobs {a} and {b} overlap"
    return int(sum(wts[i] for i in sel))


# B2-a  Greedy by earliest finish time (maximises count, not necessarily weight)
def sched_greedy_eft(s, e, wts):
    idx = np.argsort(e)
    sel, last_end = [], -1
    for i in idx:
        if s[i] >= last_end:
            sel.append(int(i)); last_end = e[i]
    return sel


# B2-b  DP (exact weighted interval scheduling, O(n log n))
def sched_dp(s, e, wts):
    n   = len(s)
    idx = np.argsort(e)
    s_s, s_e, s_w = s[idx], e[idx], wts[idx]
    # p[j] = index of latest job that ends before job j starts (0-indexed, -1 if none)
    p = np.searchsorted(s_e, s_s, side='left') - 1

    dp = np.zeros(n + 1, int)
    for j in range(1, n + 1):
        dp[j] = max(dp[j - 1], int(s_w[j - 1]) + (dp[p[j - 1] + 1] if p[j-1] >= 0 else 0))

    sel, j = [], n
    while j >= 1:
        val_incl = int(s_w[j-1]) + (dp[p[j-1]+1] if p[j-1] >= 0 else 0)
        if val_incl > dp[j - 1]:
            sel.append(int(idx[j - 1])); j = p[j - 1] + 1
        else:
            j -= 1
    return sel[::-1], int(dp[n])


print("\n=== B2. Weighted Interval Scheduling ===")
sg      = sched_greedy_eft(starts, ends, wt_j)
sd, vsd = sched_dp(starts, ends, wt_j)
vsg     = _validate_sched(sg, starts, ends, wt_j)
_validate_sched(sd, starts, ends, wt_j)

gap_pct = round(100 * (vsd - vsg) / vsd, 2) if vsd else 0.0
R["B2_scheduling"] = {
    "n_jobs": N_J,
    "greedy_eft": {"selected": sg, "total_weight": vsg,
                   "note": "Greedy EFT maximises job count, not weight; may be suboptimal"},
    "dp_exact":   {"selected": sd, "total_weight": vsd},
    "greedy_gap_pct": gap_pct,
    "validation": "DP exact; both verified by overlap checker"
}
print(f"  Greedy-EFT weight={vsg}  DP weight={vsd}  gap={gap_pct}%")


# =================================================================
# B3. MAX-CUT on 7-node graph
# =================================================================
# Build a connected weighted graph via a fixed adjacency list + random weights
N_G = 7
# Fixed edges (ensure connectivity)
EDGES_RAW = [(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(5,6),(0,5),(1,6)]
_rng2 = np.random.default_rng(2)
WMAT  = np.zeros((N_G, N_G), dtype=int)
for (i, j) in EDGES_RAW:
    w = int(_rng2.integers(1, 6))
    WMAT[i, j] = w
    WMAT[j, i] = w


def cut_value(x, W):
    """Cut value = sum W[i,j] for i in S (x=1) and j not in S (x=0)."""
    x = np.asarray(x)
    val = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if x[i] == 1 and x[j] == 0:
                val += W[i, j]
    return int(val)


def _validate_cut(x, W):
    """Cut is symmetric: S and complement S give the same cut value."""
    v  = cut_value(x, W)
    vc = cut_value(1 - x, W)
    assert v == vc, f"cut symmetry broken: {v} vs {vc}"
    return v


# B3-a  Greedy (add each node to the side that increases cut more)
def maxcut_greedy(W):
    n = len(W)
    x = np.zeros(n, int)
    for i in range(n):
        gain0 = sum(W[i, j] * (x[j] == 1) for j in range(n))
        gain1 = sum(W[i, j] * (x[j] == 0) for j in range(n))
        x[i]  = 1 if gain1 > gain0 else 0
    return x


# B3-b  Brute-force (exact, 2^7 = 128 partitions)
def maxcut_brute(W):
    n = len(W)
    best_val, best_x = 0, np.zeros(n, int)
    for bits in range(1, 2**n):
        x = np.array([(bits >> i) & 1 for i in range(n)])
        v = cut_value(x, W)
        if v > best_val:
            best_val = v; best_x = x.copy()
    return best_x, best_val


# B3-c  Simulated Annealing (heuristic, 10 seeds)
def maxcut_sa(W, seed=0, n_iter=2000, T0=5.0, alpha=0.995):
    n   = len(W)
    rng = np.random.default_rng(seed)
    x   = rng.integers(0, 2, n)
    val = cut_value(x, W)
    best_x, best_val = x.copy(), val
    T = T0
    for _ in range(n_iter):
        i = int(rng.integers(n))
        x[i] ^= 1
        nv    = cut_value(x, W)
        delta = nv - val
        if delta > 0 or rng.random() < math.exp(delta / T):
            val = nv
            if val > best_val: best_val = val; best_x = x.copy()
        else:
            x[i] ^= 1
        T *= alpha
    return best_x, best_val


print("\n=== B3. Max-Cut (7 nodes) ===")
t0  = time.perf_counter(); xg3       = maxcut_greedy(WMAT); tg3  = time.perf_counter() - t0
t0  = time.perf_counter(); xb3, vb3  = maxcut_brute(WMAT);  tb3  = time.perf_counter() - t0
sa_results = [maxcut_sa(WMAT, seed=s) for s in range(10)]
sa_vals    = [v for _, v in sa_results]

vg3 = _validate_cut(xg3, WMAT)
_validate_cut(xb3, WMAT)
for xs, _ in sa_results: _validate_cut(xs, WMAT)

print(f"  Edges={len(EDGES_RAW)}  Greedy={vg3}  Brute-force={vb3}")
print(f"  SA 10 runs: mean={np.mean(sa_vals):.1f}  std={np.std(sa_vals):.1f}")

R["B3_maxcut"] = {
    "n_nodes": N_G, "n_edges": len(EDGES_RAW),
    "weight_matrix": WMAT.tolist(),
    "greedy":      {"cut": vg3, "time_s": round(tg3, 5),
                    "gap_from_optimal_pct": round(100*(vb3-vg3)/vb3, 2) if vb3 else 0},
    "brute_force": {"cut": vb3, "time_s": round(tb3, 5)},
    "sa_10runs":   {"mean": round(float(np.mean(sa_vals)), 2),
                    "std":  round(float(np.std(sa_vals)),  2),
                    "min":  int(min(sa_vals)), "max": int(max(sa_vals))},
    "validation":  "All cuts verified by independent cut_value(); brute-force exact over 2^7 partitions"
}


# =================================================================
# B4. QUBO / ISING FORMULATION of Max-Cut
#
# Max-Cut QUBO: maximise Σ_{i<j} W[i,j]*(x_i + x_j - 2*x_i*x_j)
# Equivalent minimisation: minimise x^T Q x where:
#   Q[i,i] = -Σ_j W[i,j]      (linear terms after x_i^2 = x_i substitution)
#   Q[i,j] =  2*W[i,j]        (i < j, upper triangle)
#
# Binary -> Spin: x_i = (1 - s_i)/2,  s_i in {-1, +1}
# Ising: H = Σ h_i*s_i + Σ_{i<j} J_ij*s_i*s_j + const
#
# Penalty for cardinality constraint Σ x_i = k:
#   P(x) = lambda*(Σ x_i - k)^2
#         = lambda*(Σ x_i^2 + 2 Σ_{i<j} x_i*x_j - 2k Σ x_i + k^2)
#   => QUBO diagonal += lambda*(1 - 2k)
#      QUBO off-diag += 2*lambda  (upper triangle)
#      constant += lambda*k^2 (ignored in argmin)
# =================================================================

def build_qubo_maxcut(W, lam=0.0, k=None):
    """Return Q (n x n upper-triangular) for QUBO: minimise x^T Q x."""
    n = len(W)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] -= float(sum(W[i, j] for j in range(n)))
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * W[i, j]
    if lam > 0 and k is not None:
        for i in range(n):
            Q[i, i] += lam * (1 - 2 * k)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * lam
    return Q


def qubo_energy(Q, x):
    """Evaluate x^T Q x (upper-triangular Q convention)."""
    x = np.asarray(x)
    n = len(x)
    e = 0.0
    for i in range(n):
        e += Q[i, i] * x[i]
        for j in range(i + 1, n):
            e += Q[i, j] * x[i] * x[j]
    return float(e)


def qubo_to_ising(Q):
    """
    Convert QUBO Q (upper-triangular) to Ising (h, J, const).
    x_i = (1 - s_i)/2  =>  H_Ising = Σ h_i*s_i + Σ_{i<j} J_ij*s_i*s_j + const
    """
    n     = len(Q)
    h     = np.zeros(n)
    J     = np.zeros((n, n))
    const = 0.0
    for i in range(n):
        h[i]  += -Q[i, i] / 2.0
        const +=  Q[i, i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            q        = Q[i, j]
            const   +=  q / 4.0
            h[i]    += -q / 4.0
            h[j]    += -q / 4.0
            J[i, j] +=  q / 4.0
    return h, J, const


def ising_energy_hJ(h, J, s):
    """Evaluate H = Σ h_i*s_i + Σ_{i<j} J_ij*s_i*s_j."""
    s = np.asarray(s, dtype=float)
    n = len(s)
    e = float(h @ s)
    for i in range(n):
        for j in range(i + 1, n):
            e += J[i, j] * s[i] * s[j]
    return e


def ising_energy_direct(W, s):
    """H_Ising = -1/4 * Σ W[i,j]*(1 - s_i*s_j)."""
    n = len(s)
    h = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            h -= W[i, j] * (1 - s[i] * s[j]) / 4.0
    return float(h)


def qubo_sa(Q, seed=0, n_iter=3000, T0=10.0, alpha=0.997):
    """SA on QUBO: minimise x^T Q x."""
    n   = len(Q)
    rng = np.random.default_rng(seed)
    x   = rng.integers(0, 2, n)
    val = qubo_energy(Q, x)
    best_x, best_val = x.copy(), val
    T = T0
    for _ in range(n_iter):
        i = int(rng.integers(n))
        x[i] ^= 1
        nv    = qubo_energy(Q, x)
        delta = nv - val
        if delta < 0 or rng.random() < math.exp(-delta / T):
            val = nv
            if val < best_val: best_val = val; best_x = x.copy()
        else:
            x[i] ^= 1
        T *= alpha
    return best_x, best_val


def qubo_brute(Q):
    """Exact minimisation of QUBO by exhaustive search (2^n cases)."""
    n = len(Q)
    best_val, best_x = float('inf'), None
    for bits in range(2**n):
        x = np.array([(bits >> i) & 1 for i in range(n)])
        v = qubo_energy(Q, x)
        if v < best_val:
            best_val = v; best_x = x.copy()
    return best_x, best_val


print("\n=== B4. QUBO / Ising formulation ===")
# -- Unconstrained QUBO (pure Max-Cut) --
Q_unc       = build_qubo_maxcut(WMAT, lam=0.0)
h, J, const = qubo_to_ising(Q_unc)

xb_q, vb_q = qubo_brute(Q_unc)
sa_q_res    = [qubo_sa(Q_unc, seed=s) for s in range(10)]
sa_q_vals   = [v for _, v in sa_q_res]

# Verify: QUBO optimum cut must match Max-Cut brute-force (for x or its complement)
qubo_cut = cut_value(xb_q, WMAT)
assert qubo_cut == vb3 or cut_value(1 - xb_q, WMAT) == vb3, \
    f"QUBO brute cut {qubo_cut} (compl={cut_value(1-xb_q,WMAT)}) != Max-Cut exact {vb3}"

# Spin verification: QUBO energy must equal Ising (h,J)+const
s_best          = 1 - 2 * xb_q.astype(int)   # x=1 => s=-1, x=0 => s=+1
e_ising_via_hJ  = ising_energy_hJ(h, J, s_best) + const
e_ising_direct_val = ising_energy_direct(WMAT, s_best)
assert abs(e_ising_via_hJ - vb_q) < 1e-9, \
    f"Ising(h,J)+const={e_ising_via_hJ} != QUBO energy={vb_q}"

# -- Constrained QUBO: penalty for |S|=3 --
K_CONSTRAINT = 3
LAM          = 10.0
Q_con        = build_qubo_maxcut(WMAT, lam=LAM, k=K_CONSTRAINT)
xb_qc, vb_qc = qubo_brute(Q_con)
sa_qc_res     = [qubo_sa(Q_con, seed=s) for s in range(10)]
sa_qc_vals    = [v for _, v in sa_qc_res]

assert int(xb_qc.sum()) == K_CONSTRAINT, \
    f"Constrained brute-force violates |S|=3: got |S|={int(xb_qc.sum())}"

print(f"  QUBO unconstrained: brute-force energy={vb_q:.2f}  cut={cut_value(xb_q,WMAT)}")
print(f"  SA unconstrained 10 runs: mean={np.mean(sa_q_vals):.2f}  std={np.std(sa_q_vals):.2f}")
print(f"  QUBO constrained (|S|=3, lam={LAM}): brute-force energy={vb_qc:.2f}  cut={cut_value(xb_qc,WMAT)}")
print(f"  Ising(h,J)+const = {e_ising_via_hJ:.4f}  QUBO energy = {vb_q:.4f}  (match OK)")

R["B4_qubo_ising"] = {
    "n_nodes": N_G,
    "description": (
        "Max-Cut as QUBO: minimise x^T Q x. "
        "Binary->Spin: x_i=(1-s_i)/2. "
        "Penalty P=lambda*(sum(x_i)-k)^2 for cardinality constraint."
    ),
    "unconstrained": {
        "qubo_brute_energy": round(vb_q, 4),
        "qubo_brute_cut":    int(cut_value(xb_q, WMAT)),
        "qubo_brute_x":      xb_q.tolist(),
        "ising_h":           h.round(4).tolist(),
        "ising_J_nonzero":   {f"{i},{j}": round(J[i,j], 4)
                               for i in range(N_G) for j in range(i+1, N_G) if J[i,j] != 0},
        "ising_const":       round(const, 4),
        "sa_10runs_energy":  {"mean": round(float(np.mean(sa_q_vals)), 4),
                               "std":  round(float(np.std(sa_q_vals)),  4)},
        "verification":      "Ising(h,J)+const == QUBO energy: PASS"
    },
    "constrained_penalty": {
        "lambda": LAM, "k": K_CONSTRAINT,
        "qubo_brute_energy": round(vb_qc, 4),
        "qubo_brute_cut":    int(cut_value(xb_qc, WMAT)),
        "qubo_brute_x":      xb_qc.tolist(),
        "sa_10runs_energy":  {"mean": round(float(np.mean(sa_qc_vals)), 4),
                               "std":  round(float(np.std(sa_qc_vals)),  4)},
        "verification":      f"|S|={int(xb_qc.sum())} == k=3: PASS"
    },
    "spin_conversion_check": {
        "ising_energy_direct":        round(e_ising_direct_val, 4),
        "ising_energy_via_hJ_plus_c": round(e_ising_via_hJ, 4),
        "qubo_energy":                round(vb_q, 4),
        "all_match":                  abs(e_ising_via_hJ - vb_q) < 1e-9
    }
}

# =================================================================
# Save results (only when run as a script, not when imported by tests)
# =================================================================
if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "trackB_results.json")
    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print("\nResults saved to", out_path)
