"""Run all methods on each instance size; every output is checked by validate()."""
import sys, time, json
import numpy as np
from data import generate
from model import objective, validate
import methods as M

SIZES = [int(a) for a in sys.argv[1:]] or [20, 40, 60, 100]
TL = 60          # ILP time limit (seconds)
out = []
for n in SIZES:
    inst = generate(n, seed=0)
    g0 = M.greedy(inst)
    T = max(t for t, _ in g0.values()) + 1 + 2           # slot budget = greedy slots + 2 spare
    row = dict(n=n, T=T, edges=int((inst["C"] > 0).sum() // 2))

    vals, t0 = [], time.time()
    for sd in range(10):                                 # stochastic baseline: 10 seeds
        s = M.random_assign(inst, T, sd)
        if s:
            assert not validate(inst, s, T)
            vals.append(objective(inst, s)[0])
    row["random"] = dict(mean=float(np.mean(vals)) if vals else None,
                         std=float(np.std(vals)) if vals else None,
                         feasible_runs=len(vals), time=(time.time() - t0) / 10)

    t0 = time.time(); g = M.greedy(inst, T); tg = time.time() - t0
    assert not validate(inst, g, T)
    row["greedy"] = dict(obj=objective(inst, g)[0], time=tg)

    t0 = time.time(); ls = M.local_search(inst, T, g); tl = time.time() - t0
    assert not validate(inst, ls, T)
    row["greedy+LS"] = dict(obj=objective(inst, ls)[0], time=tl)

    sol, info = M.ilp(inst, T, TL)
    if sol:
        errs = validate(inst, sol, T)
        assert not errs, errs[:3]
        row["ilp"] = dict(obj=objective(inst, sol)[0], **info)
    else:
        row["ilp"] = dict(obj=None, **info)
    print(json.dumps(row, default=str), flush=True)
    out.append(row)

json.dump(out, open("../results/trackE_results.json", "w"), default=str, indent=1)
