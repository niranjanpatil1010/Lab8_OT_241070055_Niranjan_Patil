# Failure log (manual needs >= 3; only add cases that REALLY happened to you)

## #1 Invalid symmetry-breaking in hand-written Branch-and-Bound (Track E) — REAL
- What: B&B only allowed opening "the next new slot" (slots interchangeable assumption). On a 8-exam instance it returned 136.2 while the ILP returned 121.2, so B&B was not exact.
- Caught by: `tests/test_trackE.py::test_bnb_equals_ilp_tiny` (cross-check B&B vs ILP).
- Cause: the back-to-back term depends on slot ORDER, so slots are not symmetric. (The ILP's `y[t] >= y[t+1]` rule is valid because it only forces used slots to be a prefix, which is a shift, not a permutation.)
- Fix: removed the pruning; B&B now explores every slot. Cost: pure-Python B&B only scales to ~4-5 exams (140,935 nodes for 4 exams), so tests use n=4.
- Lesson: symmetry-breaking must be checked against the objective, not just the constraints.

## #2 Untested claim hard-coded in my own results (Track A) — REAL
- What: the first version of `trackA/run.py` stored a text note "Newton is attracted to saddles… goes to (0,0)" without running anything to check it.
- Caught by: review of the output before writing the report (the field held a string, not a measurement).
- Fix: added `newton_sad()`; real run from (0.3,0.5) converges to (0,0) — claim now measured.
- Lesson: report only what an experiment produced (manual §10).

## #3 Exact B&B did not scale — inefficiency — REAL
- What: after removing the invalid pruning (#1) the pure-Python B&B ran >5 minutes on 6-8 exams and hit the tool time limit.
- Fix: B&B test restricted to a 4-exam instance (140,935 nodes, 2.6 s); documented as non-scalable.
- Lesson: weak lower bound (partial cost only) -> use solver B&B (HiGHS) for real sizes.

## #4 Shell command written for the wrong shell — REAL (tooling)
- What: AI-generated `mkdir -p dir/{a,b}` ran in `/bin/sh`, which has no brace expansion; it created literal wrong directories and the following `cd` failed, so no files were written.
- Fix: re-ran with `bash -c`. Lesson: check which shell executes generated commands.

## #5 Hand-computed "known optima" wrong in Track B tests — REAL
- What: `test_dp_exact_small` asserted the knapsack optimum was 16 (items 0,1,3 with weights 3+4+2=9 > capacity 8 — infeasible!). `test_dp_known_optimum` asserted value 10 for two non-overlapping jobs but both jobs fit so true optimum is 11.
- Caught by: pytest failures on `assert 12 == 16` and `assert 11 == 10`.
- Fix: corrected the expected values to 12 (items 1,3: w=4+2=6, v=5+7=12) and 11 (both jobs non-overlapping). The algorithm was correct all along.
- Lesson: always compute "known answers" for tests programmatically (e.g. by brute-force) rather than by hand-arithmetic.

## #6 Graph construction for Max-Cut produced all-zero weight matrix — REAL
- What: the first version used `((_mask + _mask.T) / 2 > 0.5).astype(int)` to symmetrize an ER random adjacency mask; because the mask had 0 on the diagonal and the off-diagonal entries were either 0 or 1 (never both), the sum divided by 2 is always <=0.5, so the adjacency was all-zero. Brute-force then reported Max-Cut=0 correctly (for an empty graph) — masking the bug until the assertion `qubo_cut == vb3` would have caught it.
- Caught by: visual inspection of the printed "Brute-force=0" output.
- Fix: replaced the probabilistic mask with a hard-coded edge list that guarantees connectivity, then applied random weights from `default_rng(2)`.
- Lesson: check the data you generate (print adjacency, count edges) before running the experiment.

## #7 Analytical MGDA gradient formula sign flip (Track D) — REAL
- What: `mgda_weights()` calculated $\alpha = \text{clip}((g_2 \cdot dg) / \|dg\|^2, 0, 1)$ where $dg = g_1 - g_2$. Since $d/d\alpha \|\alpha(g_1 - g_2) + g_2\|^2 = 2\alpha \|dg\|^2 + 2 dg \cdot g_2 = 0$, the correct unclipped root is $\alpha = -(dg \cdot g_2)/\|dg\|^2 = ((g_2 - g_1) \cdot g_2)/\|dg\|^2$. The missing negative sign caused $\alpha = -0.5 \to 0.0$ on orthogonal gradients $(1,0)$ and $(0,1)$ instead of $0.5$.
- Caught by: `tests/test_trackD.py::TestMultiTaskOptimization::test_mgda_analytical_weights`.
- Fix: changed numerator to `((g2 - g1) @ g2)`.
- Lesson: unit-test optimization sub-routines against simple analytic vectors (orthogonal and opposite gradients) before training models with them.

## #8 Windows multiprocessing spawn hang in top-level script (Track D) — REAL
- What: On Windows, executing `mp.Pool()` inside a module-level execution without strict separation caused process re-initialization lockups.
- Caught by: command hanging when executing `trackD/run.py`.
- Fix: guarded all pool execution strictly under `if __name__ == '__main__':` and modularized functions so tests can import without executing pools.
- Lesson: on Windows, multiprocessing must always be protected under `if __name__ == '__main__':`.

## #9 Autograd ArrayBox incompatible with numpy.array() in QNN loss (Track C) — REAL
- What: In `trackC/run.py`, the QNN cost function used `np.array([qnn_circuit(weights, x) for x in X])` followed by `np.mean((preds - y)**2)`. When PennyLane's Autograd passed `weights` as an `ArrayBox` during backward gradient tracing, numpy threw `ValueError: setting an array element with a sequence` / `TypeError: float() argument must be a string or a real number, not 'ArrayBox'`.
- Caught by: runtime crash during execution of `trackC/run.py` at `C4_pqc_qnn`.
- Fix: accumulated scalar loss iteratively using native Python addition (`loss = loss + (pred - yi)**2 / len(y)`), which preserves Autograd computational graph tracing without numpy coercion.
- Lesson: never wrap quantum circuit outputs in standard `numpy.array()` inside functions differentiated by Autograd/PennyLane; use scalar arithmetic or PennyLane's `pnp.stack`.

## #10 Quantum Phase Estimation (QPE) qubit power reversal (Track I) — REAL
- What: In `run_i5_qpe()`, controlled-$U^{2^j}$ was applied such that counting wire 0 received $U^{2^0}$, wire 1 received $U^{2^1}$, and wire 2 received $U^{2^2}$. However, in PennyLane's `QFT(wires=[0, 1, 2])`, wire 0 is the most significant bit (weight $2^2 = 4$) and wire 2 is the least significant bit (weight $2^0 = 1$). This reversal caused the estimated phase for a $T$-gate ($\phi = 0.125$) to be measured as $0.5000$ (bit-reversed state $|100\rangle$ instead of $|001\rangle$).
- Caught by: reviewing output `estimated_phase: 0.5` vs `true_phase: 0.125`.
- Fix: applied controlled-$U$ with powers $2^2 = 4$ on wire 0, $2^1 = 2$ on wire 1, and $2^0 = 1$ on wire 2.
- Lesson: always align the control unitary powers with the endianness convention (MSB vs LSB) of the subsequent inverse QFT routine.

## #11 VM interpreter evaluated constant PRINT arguments as variable names (Track F) — REAL
- What: In `trackF/compiler.py`, the constant folding pass reduced `d = 45; PRINT d` to `PRINT 45`. The VM interpreter executed `PRINT` via `val = self.env.get(instr.arg1, 0)`. Because `instr.arg1` was the integer `45` rather than a string identifier, `self.env.get(45, 0)` looked for key `45` in the variable dictionary and returned default `0`. This caused the unoptimized output `[45]` to mismatch the optimized output `[0]`.
- Caught by: `assert out_unopt == out_opt` in `trackF/run.py` raising `AssertionError: [45] != [0]`.
- Fix: updated the VM interpreter to handle literal values: `val = instr.arg1 if isinstance(instr.arg1, int) else self.env.get(instr.arg1, 0)`.
- Lesson: when compiler passes fold variable references into immediate literals, all downstream VM instructions and codegen backends must support immediate operands.

## #12 Non-deterministic SQL query tuple order caused false discrepancy without ORDER BY (Track H) — REAL
- What: In `trackH/run.py`, the 3-way join query `Q5` did not specify an `ORDER BY` clause. When executed without secondary indexes, SQLite executed a full table scan and returned rows in disk storage insertion order. When executed with secondary indexes (`idx_orders_cust`), the query planner used an index scan, returning the exact same set of tuples but in index traversal order. The exact equality check `assert base_results == opt_results` failed with `AssertionError: Query result discrepancy detected in Q5_3Way_Join!`.
- Caught by: the correctness validator in `trackH/run.py`.
- Fix: added explicit deterministic `ORDER BY` clauses to all benchmark queries (e.g. `ORDER BY c.customer_id, o.order_id, l.item_id`).
- Lesson: in relational databases, relation outputs have undefined order unless an explicit `ORDER BY` clause is present; changing query execution plans via indexing alters output iteration order.

## #13 Track J "measured" test durations were hard-coded, not measured — REAL (misleading result)
- What: the first `trackJ/run.py` embedded a `TEST_SUITE` literal of 20 test names with durations (e.g. 0.045 s, 2.150 s) and coverage sets, described as "profiled/measured". The names (`test_trackA_quadratic_gd`, ...) do not exist in `tests/`, and no profiler had been run. The reported "46.9 % reduction / 1.88x" was therefore computed on invented inputs, which violates rule "never invent numbers". It also claimed "4,137 lines" without a measurement.
- Caught by: review during final check — grepped the real test names against `tests/*.py` and found no match.
- Fix: rewrote Track J. `trackJ/profile_suite.py` runs the repo's real pytest suite (junit timings, median of 3) and coverage.py with per-test dynamic contexts; `trackJ/run.py` solves ILP/greedy on that measured matrix and RE-RUNS the selected subsets to measure real wall time and independently re-check coverage. Old numbers discarded.
- Lesson: an optimizer is only as honest as its input data; a formulation on made-up data proves nothing. Validate that the inputs exist and come from a tool run.

## #14 AI-written xv6 code used the wrong `copyout` signature (Track G) — REAL
- What: `getpinfo()` called `copyout(pagetable, addr, src, len)` (classic xv6). This xv6-riscv revision has `copyout(pagetable, uint64 srcva_size_limit, dst, src, len)` (extra size argument), and also uses `pause`/`kfork`/`sleep_prepare`, i.e. it differs from the textbook xv6.
- Caught by: compiler error (`-Werror`, "too few arguments to function copyout").
- Fix: read the real prototype in `kernel/defs.h` and passed `myproc()->sz`.
- Lesson: AI models recall a different xv6 version; always check prototypes in the checked-out tree.

## #15 Benchmark start-up starvation: tickets set inside the child (Track G) — REAL
- What: `schedbench` children called `settickets(30/20/10)` after they first ran; children inherit 1 ticket, so the first-run child jumped to 30 tickets while siblings still held 1. In a smoke run the 20-ticket process got `sched=1, cputicks=0` in 30 ticks and the 30-ticket one got 29/30 of the CPU — a wrong "proportional share" picture caused by the benchmark, not the kernel.
- Caught by: inspecting the smoke-run output (impossible under a fair lottery: P ≈ 3e-6 for 20/60 tickets never winning in 30 draws).
- Fix: parent sets its own tickets before `fork()` (children inherit) and restores itself to 1 (`fork_with()`); re-ran, shares now ~50/33/17 %.
- Lesson: when a result is statistically impossible, suspect the experiment design before the kernel.

## #16 Two QEMU instances on one disk image (Track G) — REAL (tooling)
- What: launching `usertests` and a smoke test on the same `fs.img` failed with "Failed to get write lock".
- Fix: `xv6run.py` now boots from a private temp copy of the image on every run.
- Lesson: QEMU raw images are exclusively locked; also keeps benchmark state from leaking between runs.

## #17 Test suite could not run as one session: every track's module is called `run` (cross-track) — REAL
- What: each `tests/test_trackX.py` does `sys.path.insert(0, trackX)` + `from run import ...`. All ten tracks have a `run.py`, and Python caches a single `sys.modules['run']`. Per-track "N passed" claims in CHECKLIST.md were each true when the file was run alone, but `pytest tests` (the command in README) failed at collection (`cannot import name 'run_c1_qaoa' from run (trackB/run.py)`). `tests/test_trackE.py` additionally used a cwd-relative `sys.path.insert(0, "../trackE")`, so it only worked when launched from inside `tests/`.
- Caught by: running the whole suite in one session for the Track J profiler (it must execute every test).
- Fix: `tests/conftest.py` binds `tests/test_trackX.py` to `trackX/run.py` (loaded by path) at collection and again while its tests execute (needed for the lazy `from run import` in test_trackD); Track E path made `__file__`-relative. My FIRST conftest (cache 'run' per module as collected) passed the full suite but failed when the Track J profiler re-ran a subset of node ids in a different order (trackD got trackC's `run`) — a second AI-written bug caught by the same verification step; replaced with the deterministic path-based binding. Full suite: 72 passed; reversed 40-test subset: 40 passed.
- Lesson: "each file passes alone" is not "the suite passes"; run the exact command you tell the reader to run.

## #18 Two bugs in my own comparison harness, caught before they became a false claim (tool comparison) — REAL
- What: the first run of `compare/run_comparison.py` reported **0/20** for Antigravity's QUBO→Ising identity. Two harness bugs of mine: (1) I wrote `qubo − ising − c` but the identity is `qubo = ising + c` (constant sign), (2) `okA &= (…)` produced a `numpy.bool_`, and `okA is True` is always False for numpy booleans. A third: `json.dump` failed on `numpy.int64`.
- Caught by: the result contradicted that tool's own passing test (`tests/test_trackB.py::test_ising_conversion_consistent`), so I read its docstring/test before concluding anything. After the fixes both tools score 20/20.
- Lesson: when a comparison says the other tool is wrong, check your harness first.

## #19 Expected benefit not observed: cache-blocking (tiling) in numpy (Track D5) — REAL (misleading expectation)
- What: textbook says blocked transpose improves locality; the measured times got *worse* for small tiles and best at "no tiling" (see REPORT D5). Reported as a negative result instead of tuning until tiling "won".
- Lesson: locality gains need compiled loops; in numpy, Python-level overhead hides them.

