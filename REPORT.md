# Technical Report — Optimize Anything

## Track E — Exam timetabling (main case study)
**Process & evidence.** Assigning exams to (slot, room) so no student has two exams at once, while using few slots, avoiding back-to-back exams and wasting few seats. Baseline: random feasible assignment.

**Formulation.** x[e,t,r]∈{0,1}; y[t]∈{0,1} slot used; b[e,f,t]∈[0,1] back-to-back linearisation.
min W1·Σy[t] + W2·Σ C[e][f]·b[e,f,t] + W3·Σ x[e,t,r]·(cap_r − size_e), with W1=10, W2=1, W3=0.1.
s.t. (1) each exam placed once; (2) one exam per (slot, room); (3) room capacity ≥ enrolment (variables only exist for fitting rooms); (4) exams sharing a student never share a slot; (5) y[t] ≥ x[e,t,r]; (6) b ≥ s[e,t]+s[f,t+1]−1 (both orders); (7) y[t] ≥ y[t+1] symmetry breaking.
Data: synthetic, seed 0, 3 students per exam, each takes 4 exams from own branch + 1 elective, 6 rooms (20/30/40/60/80/120). Slot budget T = greedy slots + 2.

**Strategy-selection table**
| Variable type | Size | Exact/heuristic | Hardware | Parallelism |
|---|---|---|---|---|
| binary | 20–100 exams (≈10³–10⁴ binaries) | ILP (exact, time-limited) vs greedy/local search (heuristic) | 1 CPU | none (that is Track D) |

**Results** (ILP limit 60 s; lower is better)
| Exams | Slots T | Random (mean±std) | Greedy | Greedy+LS | ILP obj | ILP gap | ILP bound |
|---|---|---|---|---|---|---|---|
| 20 | 13 | 296.4±11.9 | 244.0 | 205.0 | 217.0 | 46% | 117.3 |
| 40 | 15 | 482.7±23.9 | 411.6 | 299.6 | 462.6 | 69% | 155.7 |
| 60 | 15 | 640.3±15.9 | 638.9 | 481.9 | 608.9 | 100% | 0.0 |
| 100 | 19 | 939.6±19.0 | 1002.0 | 799.0 | 810.0 | 100% | 0.0 |

![Track E plot](results/trackE_plot.png)

**Findings (what the numbers actually support)**
- Greedy+local search was best or tied-best at every size and took ≤ ~3 s.
- At n=100 plain greedy (1002) is *worse* than the random mean (939.6): it packs exams into the earliest slots, which creates many back-to-back exams. It minimises slots (W1 term) but ignores W2.
- The ILP did **not** reach optimality in 60 s at any size; the gaps (46%→100%) mean the solver's bound was weak (bound 0 at n=60/100 = no useful bound found). At n=40 and 60 the time-limited ILP incumbent is worse than greedy+LS. So the claim "ILP gives the optimum" is **not** supported here; only "ILP gives a valid solution plus a bound, with quality limited by the weak back-to-back linearisation".
- ILP objective at n=20 varied between runs (208 vs 217) because time-limited HiGHS is not deterministic under CPU load; report ranges, not single values.
- Own B&B (exact, verified against ILP on a 4-exam instance) explores 140,935 nodes at n=4 → does not scale; it exists to explain the algorithm.

**Limitations/next steps.** Single instance per size (add several seeds); ILP warm-start with greedy solution; tighter formulation (clique constraints, day structure); longer time limits; CP-SAT as comparison.


## Track A — Continuous optimization
Code: `trackA/run.py` (numpy only, seeded) · raw results: `results/trackA_results.json`. Stop rule: f < 1e-8 or 2000 iterations; each method's lr chosen from the same 9-value grid.

**A1. Deterministic optimizers** (convex quadratic, condition number 20; non-convex Rosenbrock)
| Problem | Method | best lr | final f | iters | lr values diverged (of 9) |
|---|---|---|---|---|---|
| convex_quadratic | GD | 0.03 | 9.91e-09 | 344 | 2 |
| convex_quadratic | Momentum | 0.01 | 2.99e-09 | 167 | 2 |
| convex_quadratic | NAG | 0.03 | 8.73e-11 | 87 | 3 |
| convex_quadratic | AdaGrad | 1.0 | 8.58e-09 | 106 | 0 |
| convex_quadratic | RMSProp | 1.0 | 3.60e-09 | 9 | 0 |
| convex_quadratic | Adam | 1.0 | 8.32e-09 | 182 | 0 |
| convex_quadratic | Newton | 1.0 | 1.31e-15 | 1 | 0 |
| rosenbrock_nonconvex | GD | 0.001 | 7.82e-02 | 2000 | 5 |
| rosenbrock_nonconvex | Momentum | 0.003 | 9.94e-09 | 617 | 5 |
| rosenbrock_nonconvex | NAG | 0.001 | 9.96e-09 | 1982 | 6 |
| rosenbrock_nonconvex | AdaGrad | 1.0 | 5.86e-03 | 2000 | 0 |
| rosenbrock_nonconvex | RMSProp | 0.003 | 5.09e-03 | 2000 | 0 |
| rosenbrock_nonconvex | Adam | 0.3 | 9.76e-09 | 820 | 0 |
| rosenbrock_nonconvex | Newton | 1.0 | 1.86e-11 | 5 | 0 |

- On the quadratic, Newton solves it in 1 step (exact for quadratics). NAG/Momentum beat plain GD. RMSProp/Adam/AdaGrad tolerated large lr without diverging but "iters" depends strongly on the grid.
- On Rosenbrock, plain GD did not converge in 2000 iterations (f=0.078 at best lr); Momentum, NAG, Adam and Newton converged; AdaGrad and RMSProp reached ~5e-3 only. Single start point only → ranking is indicative, not general.

**A2. Noisy objective** (linear regression, n=2000, 20 epochs, 10 seeds)
| Method | mean gap to optimal loss | std (10 seeds) |
|---|---|---|
| GD(full batch) | 5.19e-02 | 1.0e-16 |
| SGD(bs=1) | 1.13e-02 | 6.5e-03 |
| minibatch(bs=32) | 1.17e-03 | 6.8e-04 |
| SGD(bs=1) + lr decay | 8.08e-07 | 2.8e-07 |
Fixed-lr SGD plateaus at a noise floor; lr decay removes it. Full-batch GD's gap (0.052) is larger only because lr=0.1 and 20 epochs = 20 updates — it is *not* a statement that GD is worse; update counts differ (stated, not hidden).

**A3. Gauss–Newton vs GD** (fit y=a·exp(bx), true a=2, b=0.7): GN reaches cost 0.0389 in 7 iterations (a=1.997, b=0.699), but its first step overshoots (cost 72,964 > start) — GN has no step-size safeguard. GD with lr=1e-3 was still at 2.71 after 10 iterations (lr not tuned, so this comparison is weak).

**A4. Backpropagation.** Hand-written 2-layer tanh MLP with L2 regularisation on an XOR-like task. Analytic gradient vs central finite differences: max relative error 2.1e-9 → backprop correct. GD and Adam both reach 98% accuracy (loss 0.2196 vs 0.2173).

**A5. Curvature.** Hessian-vector product by finite-difference of gradients matches the exact Hessian (max error 9.4e-12) without forming H. Power iteration with HVPs finds the top eigenvalue 0.28221 = exact 0.28221 (low-rank curvature). Empirical Fisher diagonal (0.208–0.242) is close to the Hessian diagonal (0.198–0.240) for logistic regression, as theory suggests near the data distribution.

**A6. Landscapes.** f=x²−y²+0.25y⁴: GD started exactly at the saddle (0,0) stays there; with small noise it escapes to a minimum (−0.03, 1.40) (true minima at y=±√2≈1.414). Newton from (0.3,0.5) converges to the saddle (0,0) — second-order methods are attracted to saddles. f=x⁴−3x²+x has two minima: GD from −2/−0.5 reaches x=−1.30 (f=−3.51, global), from 0.5/2 reaches x=1.13 (f=−1.07, local) → the result depends on the start. Convex case (A1 quadratic) has one minimum reachable from anywhere.

## Tracks B–D, F–J
TODO (one section each, linked to code/results).

## Final questions (Section 11) — answer after all tracks are finished.
