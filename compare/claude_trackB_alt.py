"""Fresh Claude (claude.ai chat, Claude Sonnet 5.5) solution of the Track B task, written WITHOUT looking at
trackB/run.py. Task given: 0/1 knapsack (greedy, DP, B&B, ILP) + Max-Cut as QUBO and Ising. Used only for the
tool comparison (README section 'Tool comparison')."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def knap_greedy(w, v, W):
    x = np.zeros(len(w), int); cap = W
    for i in np.argsort(-(v / w)):
        if w[i] <= cap: x[i] = 1; cap -= w[i]
    return x

def knap_dp(w, v, W):
    n = len(w); dp = np.zeros((n + 1, W + 1), int)
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if w[i - 1] <= W:
            dp[i, w[i - 1]:] = np.maximum(dp[i - 1, w[i - 1]:], dp[i - 1, :W - w[i - 1] + 1] + v[i - 1])
    x = np.zeros(n, int); c = W
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]: x[i - 1] = 1; c -= w[i - 1]
    return x

def knap_bnb(w, v, W):
    n = len(w); order = np.argsort(-(v / w)); ws, vs = w[order], v[order]
    best = {"val": -1, "x": None}
    def bound(i, cap, val):                       # fractional (LP) upper bound
        for j in range(i, n):
            if ws[j] <= cap: cap -= ws[j]; val += vs[j]
            else: return val + vs[j] * cap / ws[j]
        return val
    def rec(i, cap, val, x):
        if val > best["val"]: best["val"], best["x"] = val, x.copy()
        if i == n or bound(i, cap, val) <= best["val"]: return
        if ws[i] <= cap: x[i] = 1; rec(i + 1, cap - ws[i], val + vs[i], x); x[i] = 0
        rec(i + 1, cap, val, x)
    rec(0, W, 0, np.zeros(n, int))
    out = np.zeros(n, int); out[order] = best["x"]; return out

def knap_ilp(w, v, W):
    r = milp(-v.astype(float), constraints=LinearConstraint(w[None, :].astype(float), -np.inf, W),
             integrality=np.ones(len(w)), bounds=Bounds(0, 1))
    return np.round(r.x).astype(int)

def maxcut_qubo(A):
    """minimise x^T Q x, x in {0,1}^n, cut(x) = -x^T Q x.  Q symmetric full matrix."""
    n = len(A); Q = A.astype(float).copy()
    for i in range(n): Q[i, i] = -A[i].sum()
    return Q * np.where(np.eye(n) == 1, 1.0, 1.0)   # off-diagonal: +A_ij (counted twice in x^TQx -> 2*A_ij*xi*xj)

def qubo_to_ising(Q):
    """x=(1-s)/2 (s=+1 <-> x=0): returns h, J (J_ij for i<j), const so that x^T Q x = sum h_i s_i + sum J_ij s_i s_j + const."""
    n = len(Q); S = (Q + Q.T) / 2
    J = np.triu(S, 1) / 2 * 2 / 2 * 2 / 2           # placeholder replaced below
    J = np.zeros((n, n)); h = np.zeros(n); c = 0.0
    for i in range(n):
        c += S[i, i] / 2; h[i] += -S[i, i] / 2
        for j in range(i + 1, n):
            q = 2 * S[i, j]                          # coefficient of x_i x_j
            c += q / 4; h[i] += -q / 4; h[j] += -q / 4; J[i, j] += q / 4
    return h, J, c

def ising_energy(h, J, c, s): return float(h @ s + s @ J @ s + c)
def cut_value(A, x): return float(sum(A[i, j] for i in range(len(A)) for j in range(i + 1, len(A)) if x[i] != x[j]))
