"""Track D additions for Chapter 6 topics: data-locality optimization (D5) and synchronization /
communication optimization (D6). Measured on THIS machine (see 'cpus'); results -> ../results/trackD_locality_sync.json"""
import json, os, time, statistics as st
import numpy as np
import multiprocessing as mp

def med(f, rep=5):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return st.median(ts)

# ---------------- D5: data locality ----------------
def d5():
    out = {}
    a = np.random.default_rng(0).random((4000, 4000))                    # C-order: rows contiguous
    def by_rows(): return sum(a[i, :].sum() for i in range(a.shape[0]))
    def by_cols(): return sum(a[:, j].sum() for j in range(a.shape[1]))   # stride = 32 KB between elements
    assert abs(by_rows() - by_cols()) < 1e-6 * abs(by_rows())               # same result, different access order
    out["sum_rows_s"] = med(by_rows); out["sum_cols_s"] = med(by_cols)
    out["col_over_row_slowdown"] = out["sum_cols_s"] / out["sum_rows_s"]
    b = np.random.default_rng(1).random((3000, 3000)); n = b.shape[0]
    def transpose_tiled(T):
        o = np.empty_like(b)
        for i in range(0, n, T):
            for j in range(0, n, T): o[j:j + T, i:i + T] = b[i:i + T, j:j + T].T
        return o
    assert np.array_equal(transpose_tiled(64), b.T)                          # correctness check
    out["transpose_by_tile_s"] = {str(T): med(lambda T=T: transpose_tiled(T), rep=3) for T in [8, 16, 32, 64, 128, 256, 1024, 3000]}
    out["transpose_best_tile"] = min(out["transpose_by_tile_s"], key=out["transpose_by_tile_s"].get)
    return out

# ---------------- D6: synchronization & communication ----------------
def _work_locked(args):
    lock, counter, lo, hi = args
    for i in range(lo, hi):
        with lock: counter.value += i                       # synchronize on EVERY element
def _work_local(args):
    lo, hi = args; return sum(range(lo, hi))                # no sharing; one combine at the end
def _tiny(x): return x * 2

def d6(nproc):
    out = {"nproc": nproc}; N = 200_000; ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    chunks = [(k * N // nproc, (k + 1) * N // nproc) for k in range(nproc)]; expect = sum(range(N))
    def run_locked():
        lock, counter = ctx.Lock(), ctx.Value("q", 0, lock=False)
        ps = [ctx.Process(target=_work_locked, args=((lock, counter, lo, hi),)) for lo, hi in chunks]
        for p in ps: p.start()
        for p in ps: p.join()
        assert counter.value == expect; return counter.value
    def run_local():
        with ctx.Pool(nproc) as pool: r = sum(pool.map(_work_local, chunks))
        assert r == expect; return r
    def run_serial(): assert sum(range(N)) == expect
    out["serial_s"] = med(run_serial); out["lock_per_element_s"] = med(run_locked, 3); out["local_partial_sums_s"] = med(run_local, 3)
    out["lock_over_local_slowdown"] = out["lock_per_element_s"] / out["local_partial_sums_s"]
    # communication granularity: 20k tiny tasks, different chunksize
    items = list(range(20_000)); gran = {}
    with ctx.Pool(nproc) as pool:
        for cs in [1, 10, 100, 1000, 5000]:
            gran[str(cs)] = med(lambda cs=cs: pool.map(_tiny, items, chunksize=cs), 3)
    out["pool_map_chunksize_s"] = gran; out["best_chunksize"] = min(gran, key=gran.get)
    out["serial_tiny_tasks_s"] = med(lambda: [_tiny(i) for i in items])
    return out

if __name__ == "__main__":
    n = os.cpu_count()
    R = {"cpus": n, "D5_data_locality": d5(), "D6_sync_comm": d6(min(4, n))}
    json.dump(R, open("../results/trackD_locality_sync.json", "w"), indent=1); print(json.dumps(R, indent=1))
