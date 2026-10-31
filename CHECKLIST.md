# Chapter 6 coverage checklist
Syllabus text (student's course outline, Unit 6, 8 h, CO4): *Optimization in Machine Learning and Parallel Programs: Continuous Optimization, Discrete optimization, Submodular Functions, Load balancing techniques, data Locality optimization, synchronization and communication optimization, Optimization strategy selection.* The assignment manual (§5) additionally lists quantum topics, so both are covered below.

## Syllabus topic → where it is covered
| Syllabus topic | Track / file | Status |
|---|---|---|
| Continuous optimization | trackA/run.py | ✅ |
| Discrete optimization | trackB/run.py, trackE/ | ✅ |
| Submodular functions | trackD/run.py (D1) | ✅ |
| Load balancing techniques | trackD/run.py (D2: LPT, work stealing) | ✅ |
| Data locality optimization | trackD/run_locality_sync.py (D5) | ✅ (negative result for tiling) |
| Synchronization and communication optimization | trackD/run_locality_sync.py (D6) | ✅ |
| Optimization strategy selection | REPORT.md D7 + Track E table | ✅ |
| ML / parallel / quantum basics (manual §5) | trackA, trackC, trackD, trackI | ✅ |

## Detailed checklist
| Track | Item | Experiment / file | Status |
|---|---|---|---|
| A | GD, SGD, mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam | trackA/run.py §1-2; tests/test_trackA.py (8 passed, validates saved results) | ✅ |
| A | Newton, Gauss-Newton, Fisher/diag/low-rank, Hessian-vector products | trackA/run.py §1,3,5 | ✅ |
| A | Backprop on small ML model; convex vs non-convex (saddle, local minima) | trackA/run.py §4,6 | ✅ |
| B | Greedy, ILP, Branch-and-Bound, DP, QUBO/Ising (Max-Cut 6-8 nodes) | trackB/run.py (B1-B4); tests/test_trackB.py (23 passed) | ✅ |
| C | VQE, QAOA (solve B's QUBO), PQC/QNN, parameter-shift, SPSA, QNG, shots/noise/barren plateau | trackC/run.py (C1-C7); tests/test_trackC.py (7 passed) | ✅ |
| D | Submodular greedy; load balancing, work stealing, data/model/pipeline parallelism, speedup, Amdahl/Gustafson; FedAvg; MGDA; GradNorm | trackD/run.py (D1-D4); tests/test_trackD.py (12 passed); D5/D6: trackD/run_locality_sync.py, tests/test_trackD_locality.py (5 passed) | ✅ |
| E | Exam timetabling: random, greedy, greedy+LS, ILP, B&B | trackE/ | ✅ |
| F | Mini-compiler, >=4 optimizations, one decision as optimization | trackF/compiler.py, trackF/run.py; tests/test_trackF.py (8 passed) | ✅ |
| G | xv6 lottery scheduler (settickets/getpinfo/schedseed) + 4 workloads (prop, pingpong, fair, forkstorm), usertests pass | trackG/xv6_patch.diff, trackG/run.py, user/schedbench.c; tests/test_trackG.py (5 passed); MEASURED under QEMU; 2nd feature NOT done | ✅ (minimum scope) |
| H | DBMS index/join-order/view selection (greedy vs ILP/DP vs default optimizer) | trackH/run.py; tests/test_trackH.py (5 passed) | ✅ |
| I | All quantum algorithms (DJ, BV, Grover, QFT, QPE, Shor, annealing), QML vs classical | trackI/run.py (I1-I9); tests/test_trackI.py (9 passed) | ✅ |
| J | Test-suite minimization (ILP vs greedy vs naive) + budget prioritization on this repo (3130 source lines, 72 tests), measured + re-verified | trackJ/profile_suite.py, trackJ/run.py; tests/test_trackJ.py (4 passed) | ✅ |

Note: the whole suite (`pytest tests`) = 84 tests, all passing in a single session (conftest.py fixes the cross-track `run` module collision, FAILURE_LOG #17).
