"""Same task, two tools, same independent validators. Truth = brute force. Writes ../results/tool_comparison.json"""
import sys, io, json, time, itertools, contextlib, importlib.util
import numpy as np
sys.path.insert(0, "."); import claude_trackB_alt as C
spec = importlib.util.spec_from_file_location("agB", "../trackB/run.py"); AG = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(AG)       # Antigravity's Track B

def brute_knap(w, v, W):
    n = len(w); X = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    ok = X @ w <= W; return int((X @ v)[ok].max())
def feasible_val(x, w, v, W): return (int(v @ x) if int(w @ x) <= W else None)

rng = np.random.default_rng(123); N = 40; res = {"claude_chat": {}, "antigravity": {}}
impls = {"claude_chat": dict(greedy=C.knap_greedy, dp=C.knap_dp, bnb=C.knap_bnb, ilp=C.knap_ilp),
         "antigravity": dict(greedy=AG.knapsack_greedy, dp=lambda w, v, W: AG.knapsack_dp(w, v, W)[0],
                             bnb=lambda w, v, W: (lambda r: r[0] if isinstance(r, tuple) else r)(AG.knapsack_bnb(w, v, W)),
                             ilp=lambda w, v, W: AG.knapsack_ilp(w, v, W)[0])}
stats = {t: {m: dict(infeasible=0, optimal=0, t=0.0) for m in impls[t]} for t in impls}
for _ in range(N):
    n = 15; w = rng.integers(1, 15, n); v = rng.integers(1, 20, n); W = int(w.sum() * 0.4); opt = brute_knap(w, v, W)
    for t, fs in impls.items():
        for m, f in fs.items():
            t0 = time.perf_counter(); x = np.asarray(f(w, v, W)); dt = time.perf_counter() - t0
            val = feasible_val(x, w, v, W); s = stats[t][m]; s["t"] += dt
            if val is None: s["infeasible"] += 1
            elif val == opt: s["optimal"] += 1
# Max-Cut QUBO / Ising identities on random graphs
def maxcut_brute(A):
    n = len(A); return max(C.cut_value(A, [(b >> i) & 1 for i in range(n)]) for b in range(2 ** n))
mc = {"claude_chat": dict(qubo_min_equals_maxcut=0, ising_identity_ok=0), "antigravity": dict(qubo_min_equals_maxcut=0, ising_identity_ok=0)}
G = 20
for _ in range(G):
    n = 8; A = np.triu(rng.integers(0, 2, (n, n)) * rng.integers(1, 5, (n, n)), 1); A = A + A.T; best = maxcut_brute(A)
    Q = C.maxcut_qubo(A)
    qmin = min(float(np.array([(b >> i) & 1 for i in range(n)]) @ Q @ np.array([(b >> i) & 1 for i in range(n)])) for b in range(2 ** n))
    h, J, c = C.qubo_to_ising(Q); ok = True
    for b in range(2 ** n):
        x = np.array([(b >> i) & 1 for i in range(n)]); s = 1 - 2 * x
        ok &= abs(x @ Q @ x - C.ising_energy(h, J, c, s)) < 1e-9
    mc["claude_chat"]["qubo_min_equals_maxcut"] += abs(-qmin - best) < 1e-9; mc["claude_chat"]["ising_identity_ok"] += ok
    Qa = AG.build_qubo_maxcut(A); qa = AG.qubo_brute(Qa)[1]
    mc["antigravity"]["qubo_min_equals_maxcut"] += abs(-qa - best) < 1e-9
    ha, Ja, ca = AG.qubo_to_ising(Qa)[:3] if len(AG.qubo_to_ising(Qa)) >= 3 else (None, None, 0)
    okA = True
    try:
        for b in range(2 ** n):
            x = np.array([(b >> i) & 1 for i in range(n)]); s = 1 - 2 * x
            okA &= abs(AG.qubo_energy(Qa, x) - (AG.ising_energy_hJ(ha, Ja, s) + ca)) < 1e-9
    except Exception as e: okA = f"error: {type(e).__name__}"
    mc["antigravity"]["ising_identity_ok"] += (False if isinstance(okA, str) else bool(okA))
out = dict(knapsack_instances=N, maxcut_graphs=G, knapsack={t: {m: dict(infeasible=s["infeasible"], optimal=s["optimal"], mean_ms=1000 * s["t"] / N) for m, s in d.items()} for t, d in stats.items()}, maxcut=mc)
json.dump(out, open("../results/tool_comparison.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
