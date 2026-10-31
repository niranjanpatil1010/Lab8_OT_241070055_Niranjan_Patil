"""Track G driver: boot xv6 (round-robin baseline vs lottery) under QEMU and run schedbench.
Usage: python3 trackG/run.py [runs] [dur_ticks]     -> results/trackG_raw.txt, results/trackG_results.json
Build kernels first: see trackG/README_G.md. Everything here is MEASURED under QEMU (1 vCPU, TCG emulation)."""
import os, sys, re, json, subprocess, threading
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import xv6run

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
KERNELS = ["rr", "lottery"]

def commands(dur, seed):
    return [f"schedbench prop {dur} {seed}", f"schedbench pingpong {dur} {seed}",
            f"schedbench fair 8 {dur} {seed}", f"schedbench fair 1 {dur} {seed}",
            f"schedbench forkstorm {dur} {seed}"]

def parse(text):
    rows = []
    for m in re.finditer(r"RESULT (.*)", text):
        d = dict(kv.split("=") for kv in m.group(1).split())
        rows.append({k: (v if k == "wl" else int(v)) for k, v in d.items()})
    return rows

def jain(x):
    x = np.asarray(x, float); return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if x.sum() > 0 else 0.0

def metrics(rows):
    m = {}
    pr = sorted([r for r in rows if r["wl"] == "prop"], key=lambda r: r["role"])
    if len(pr) == 3:
        ct = np.array([r["cputicks"] for r in pr], float); sh = ct / ct.sum()
        exp = np.array([30, 20, 10]) / 60.0
        m["prop_share_t30"], m["prop_share_t20"], m["prop_share_t10"] = map(float, sh)
        m["prop_L1_error_vs_tickets"] = float(np.abs(sh - exp).sum())
        m["prop_total_iters"] = sum(r["iters"] for r in pr)
    pp = [r for r in rows if r["wl"] == "pingpong"]
    if len(pp) == 5:
        m["pingpong_roundtrips"] = [r for r in pp if r["role"] == 0][0]["iters"]
        m["pingpong_hog_iters"] = sum(r["iters"] for r in pp if r["role"] == 1)
    fa = [r for r in rows if r["wl"] == "fair"]
    if len(fa) == 8:
        m["fair8_jain_cputicks"] = jain([r["cputicks"] for r in fa])
        m["fair8_total_iters"] = sum(r["iters"] for r in fa)
    so = [r for r in rows if r["wl"] == "solo"]
    if len(so) == 1:
        m["solo_iters"] = so[0]["iters"]
    fs = [r for r in rows if r["wl"] == "forkstorm"]
    if len(fs) == 4:
        m["forkstorm_total_forks"] = sum(r["iters"] for r in fs)
    return m

def one(kernel, seed, dur, out):
    boot, outs = xv6run.run(os.path.join(HERE, "build", kernel), commands(dur, seed), timeout=1500)
    out[kernel] = "\n".join(outs)

if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dur = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    raw, per = [], {k: [] for k in KERNELS}
    for i in range(runs):
        seed = i + 1
        out = {}
        ts = [threading.Thread(target=one, args=(k, seed, dur, out)) for k in KERNELS]
        [t.start() for t in ts]; [t.join() for t in ts]
        for k in KERNELS:
            raw.append(f"##### kernel={k} run={i} seed={seed} dur={dur}\n{out[k]}")
            per[k].append(metrics(parse(out[k])))
        print(f"run {i} done", flush=True)
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "trackG_raw.txt"), "w").write("\n".join(raw))
    agg = {}
    for k in KERNELS:
        keys = sorted({kk for m in per[k] for kk in m})
        agg[k] = {kk: {"mean": float(np.mean([m[kk] for m in per[k] if kk in m])),
                       "std": float(np.std([m[kk] for m in per[k] if kk in m])),
                       "n": sum(kk in m for m in per[k])} for kk in keys}
    json.dump({"setup": {"qemu": "qemu-system-riscv64 8.2.2, 1 vCPU, TCG", "xv6": "mit-pdos/xv6-riscv e93dabe",
                         "runs": runs, "duration_ticks": dur, "status": "MEASURED (simulated machine, not bare metal)"},
               "aggregate": agg, "per_run": per}, open(os.path.join(RES, "trackG_results.json"), "w"), indent=1)
    print(json.dumps(agg, indent=1))
