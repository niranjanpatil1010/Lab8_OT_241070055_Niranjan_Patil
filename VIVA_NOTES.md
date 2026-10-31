# Viva notes — be able to explain and modify these WITHOUT AI
(Where a formulation is quoted it is the one written in REPORT.md; open the code file next to it.)

| Piece | File | What you must be able to say |
|---|---|---|
| Timetabling ILP (E) | trackE/methods.py `ilp()` | variables x[e,t,r], y[t]; each exam once; one exam per (slot,room); capacity via variable existence; clash: s[e,t]+s[f,t] ≤ 1 for exams sharing a student; back-to-back b ≥ s[e,t]+s[f,t+1]−1; `y[t] ≥ y[t+1]` symmetry breaking valid, but NOT slot permutation (B&B bug, FAILURE_LOG #1). Why ILP gap stayed large. |
| Greedy vs greedy+LS (E) | trackE/methods.py | greedy = largest-degree first, earliest slot, best-fit room; why it loses to random at n=100 (packs early slots → back-to-back). |
| Optimizers + backprop (A) | trackA/run.py | update rules of Momentum/NAG/AdaGrad/RMSProp/Adam; Newton step; Gauss–Newton `(JᵀJ)⁻¹Jᵀr`; why a finite-difference gradient check validates backprop; HVP without forming H; saddle: GD stays, noise escapes, Newton is attracted. |
| QUBO → Ising (B) | trackB/run.py `build_qubo_maxcut`, `qubo_to_ising` | x=(1−s)/2; cut = −xᵀQx; penalty λ(Σx−k)²; energy identity `QUBO = Ising + const`. |
| QAOA / VQE (C) | trackC/run.py | cost + mixer layers, p, expectation of the Ising Hamiltonian, parameter-shift rule, SPSA (2 evals per step), QNG (metric tensor), why shots add noise, barren plateau (gradient variance ∝ 2⁻ⁿ). |
| Submodular, parallel, FedAvg, MGDA (D) | trackD/run.py | diminishing returns definition and the (1−1/e) greedy guarantee for monotone submodular under a cardinality constraint; LPT vs work stealing; Amdahl S=1/((1−p)+p/N) vs Gustafson; why speedup < 1 on tiny tasks. |
| Locality, sync, communication (D5/D6) | trackD/run_locality_sync.py | row vs column traversal, cache lines, strides; why tiling did not help in numpy; lock per element vs private partial sums; chunksize = communication granularity. |
| Compiler passes + register allocation ILP (F) | trackF/compiler.py | lexer→parser→3-address IR→VM; the 5 passes; ILP: x[v,c], spill s[v], Σc x[v,c]+s[v]=1, interfering variables get different registers, minimize spills. |
| Index selection knapsack (H) | trackH/run.py | max Σ b_i x_i s.t. Σ c_i x_i ≤ B; DP vs ILP vs greedy ratio; EXPLAIN QUERY PLAN before/after; identical results; why default optimizer = no secondary indexes. |
| Test-suite minimization (J) | trackJ/run.py | weighted set cover: min Σ t_i x_i s.t. every function covered ≥ 1; budget version with y_f; ILP vs greedy (ln n); why measured gain (7 %) < model gain (27 %): fixed start-up cost not in the model. |
| xv6 lottery scheduler (G) | trackG/xv6_patch.diff, user/schedbench.c | tickets per process; pass 1 sums tickets of RUNNABLE processes, random winner, pass 2 finds holder, re-check state under the process lock; new syscalls settickets/getpinfo/schedseed; timer-tick accounting; why shares are only approximately 50/33/17; why equal tickets are less even than round-robin (Jain 0.94 vs 1.00); why 1 vCPU only. |
| Quantum algorithms (I) | trackI/run.py | circuits for DJ, BV, Grover (oracle + diffusion, ≈π/4·√N iterations), QFT, QPE, Shor's order finding; Grover query advantage but slower wall-clock on a simulator (2ⁿ state vector). |

Likely modifications an examiner may ask: change a weight in the timetabling objective, change tickets in the scheduler, add a constraint to an ILP, change the number of QAOA layers, change the knapsack budget.
