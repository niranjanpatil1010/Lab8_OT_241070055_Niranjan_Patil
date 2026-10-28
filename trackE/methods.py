import time, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from model import objective
from data import W1, W2, W3


def _rooms(inst, e):
    return [r for r, c in enumerate(inst["caps"]) if c >= inst["size"][e]]


def _can_place(inst, sol, e, t, r):
    for f, (tt, rr) in sol.items():
        if f == e:
            continue
        if tt == t and (rr == r or inst["C"][e][f]):
            return False
    return True


# ---------- 1. Random baseline ----------
def random_assign(inst, T, seed):
    rng = random.Random(seed)
    for _ in range(200):                       # retry until a feasible assignment appears
        order = list(range(inst["n"])); rng.shuffle(order)
        sol, ok = {}, True
        for e in order:
            opts = [(t, r) for t in range(T) for r in _rooms(inst, e) if _can_place(inst, sol, e, t, r)]
            if not opts:
                ok = False; break
            sol[e] = rng.choice(opts)
        if ok:
            return sol
    return None


# ---------- 2. Greedy: largest-degree first, earliest slot, best-fit (smallest) room ----------
def greedy(inst, T=None):
    C = inst["C"]
    order = sorted(range(inst["n"]), key=lambda e: (-(C[e] > 0).sum(), -inst["size"][e]))
    sol = {}
    for e in order:
        placed = False
        for t in range(T if T is not None else 10 ** 4):
            for r in sorted(_rooms(inst, e), key=lambda r: inst["caps"][r]):
                if _can_place(inst, sol, e, t, r):
                    sol[e] = (t, r); placed = True; break
            if placed:
                break
        if not placed:
            return None
    return sol


# ---------- 3. Greedy + local search (move one exam to its best feasible cell, repeat) ----------
def local_search(inst, T, start, max_pass=20):
    sol = dict(start); best, _ = objective(inst, sol)
    for _ in range(max_pass):
        improved = False
        for e in range(inst["n"]):
            keep = sol[e]
            for t in range(T):
                for r in _rooms(inst, e):
                    if (t, r) == sol[e] or not _can_place(inst, sol, e, t, r):
                        continue
                    old = sol[e]; sol[e] = (t, r)
                    val, _ = objective(inst, sol)
                    if val < best - 1e-9:
                        best = val; improved = True
                    else:
                        sol[e] = old
        if not improved:
            break
    return sol


# ---------- 4. ILP via HiGHS (branch-and-cut), scipy.optimize.milp ----------
def ilp(inst, T, time_limit=60):
    n, C, size, caps = inst["n"], inst["C"], inst["size"], inst["caps"]
    idx = {}
    def var(key):
        idx[key] = len(idx); return idx[key]
    for e in range(n):
        for t in range(T):
            for r in _rooms(inst, e):
                var(("x", e, t, r))
    for t in range(T):
        var(("y", t))
    edges = [(e, f) for e in range(n) for f in range(e + 1, n) if C[e][f]]
    for (e, f) in edges:
        for t in range(T - 1):
            var(("b", e, f, t))
    N = len(idx)
    c = np.zeros(N)
    for k, i in idx.items():
        if k[0] == "x": c[i] = W3 * (caps[k[3]] - size[k[1]])
        elif k[0] == "y": c[i] = W1
        else: c[i] = W2 * C[k[1]][k[2]]
    rows, lo, hi = [], [], []
    def add(coefs, l, u): rows.append(coefs); lo.append(l); hi.append(u)
    def s(e, t): return [(idx[("x", e, t, r)], 1) for r in _rooms(inst, e)]   # "exam e is in slot t"
    for e in range(n):                                   # (1) each exam exactly once
        add([(idx[("x", e, t, r)], 1) for t in range(T) for r in _rooms(inst, e)], 1, 1)
    for t in range(T):
        for r in range(len(caps)):                       # (2) one exam per (slot, room)
            cells = [(idx[("x", e, t, r)], 1) for e in range(n) if ("x", e, t, r) in idx]
            if cells: add(cells, 0, 1)
        for e in range(n):                               # (3) y[t] >= exam e in slot t
            add(s(e, t) + [(idx[("y", t)], -1)], -np.inf, 0)
        if t < T - 1:                                    # symmetry breaking: use early slots first
            add([(idx[("y", t)], -1), (idx[("y", t + 1)], 1)], -np.inf, 0)
    for (e, f) in edges:
        for t in range(T):                               # (4) student clash
            add(s(e, t) + s(f, t), -np.inf, 1)
        for t in range(T - 1):                           # (5) back-to-back linearisation
            bi = idx[("b", e, f, t)]
            add(s(e, t) + s(f, t + 1) + [(bi, -1)], -np.inf, 1)
            add(s(f, t) + s(e, t + 1) + [(bi, -1)], -np.inf, 1)
    A = lil_matrix((len(rows), N))
    for i, co in enumerate(rows):
        for j, v in co: A[i, j] += v
    integ = np.zeros(N)
    for k, i in idx.items():
        integ[i] = 0 if k[0] == "b" else 1               # b is continuous (tight at optimum)
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ,
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
    info = dict(status=str(res.message), runtime=time.time() - t0, gap=getattr(res, "mip_gap", None),
                nodes=getattr(res, "mip_node_count", None), bound=getattr(res, "mip_dual_bound", None))
    if res.x is None:
        return None, info
    sol = {k[1]: (k[2], k[3]) for k, i in idx.items() if k[0] == "x" and res.x[i] > 0.5}
    return sol, info


# ---------- 5. Hand-written Branch-and-Bound (tiny instances; the one you explain in viva) ----------
def _partial(inst, sol):
    """Cost of a partial assignment. All 3 terms only grow when exams are added -> valid lower bound."""
    C, size, caps = inst["C"], inst["size"], inst["caps"]
    es = list(sol)
    used = len({t for t, _ in sol.values()})
    b2b = sum(C[e][f] for i, e in enumerate(es) for f in es[i + 1:]
              if C[e][f] and abs(sol[e][0] - sol[f][0]) == 1)
    unused = sum(caps[r] - size[e] for e, (t, r) in sol.items())
    return W1 * used + W2 * b2b + W3 * unused


def branch_and_bound(inst, T, ub=float("inf")):
    n = inst["n"]
    order = sorted(range(n), key=lambda e: -(inst["C"][e] > 0).sum())
    best = {"val": ub + 1e-6, "sol": None, "nodes": 0}
    def rec(i, sol):
        best["nodes"] += 1
        lb = _partial(inst, sol)
        if lb >= best["val"] - 1e-9:                     # prune: cannot beat incumbent
            return
        if i == n:
            best["val"], best["sol"] = lb, dict(sol); return
        e = order[i]
        for t in range(T):                               # NOTE: no slot-symmetry pruning: back-to-back
                                                         # cost depends on slot order (see failure log #1)
            for r in _rooms(inst, e):
                if _can_place(inst, sol, e, t, r):
                    sol[e] = (t, r); rec(i + 1, sol); del sol[e]
    rec(0, {})
    return best["sol"], dict(nodes=best["nodes"])
