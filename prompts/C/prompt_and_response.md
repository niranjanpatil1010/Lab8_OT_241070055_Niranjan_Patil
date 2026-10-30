# Track C Prompt Log

## Original prompt (excerpt)
```
Track C — Quantum optimization:
simulator (Qiskit or PennyLane): VQE, QAOA on the Track B QUBO, PQC/QNN,
parameter-shift, SPSA, quantum natural gradient; compare with the classical baseline;
show finite-shot effect, hardware-noise model effect and barren plateaus as qubits grow.
State clearly: simulated, not real hardware.
```

## AI Response / Implementation Approach
- Implemented in `trackC/run.py` using PennyLane simulator (`default.qubit` and `default.mixed`):
  - C1: QAOA with $p=1$ and $p=2$ on the exact 7-node Max-Cut QUBO from Track B. Compared with Track B's classical brute-force (33) and simulated annealing (33.0).
  - C2: VQE for Transverse-Field Ising Model ($J=1.0, g=0.8$) on 4 qubits, compared with exact classical matrix diagonalization (-4.1456 vs -3.8347).
  - C3: Convergence comparison of Parameter-Shift Rule (analytic gradients, 12 evals/step), SPSA (stochastic, 2 evals/step), and QNG (quantum natural gradient via Fubini-Study tensor, exact convergence).
  - C4: 2-qubit Parameterized Quantum Circuit (PQC / QNN) trained for non-linear concentric circles classification (71.67% accuracy).
  - C5: Finite-shot noise evaluation on 50, 200, 1000, 10000 shots, verifying $1/\sqrt{N_{\text{shots}}}$ scaling against analytic statevector.
  - C6: Hardware noise model under Depolarizing noise $p \in [0.0, 0.2]$ on Bell state parity.
  - C7: Barren plateaus on $n \in [2, 4, 6, 8]$ qubits, demonstrating exponential variance decay $\sim 2^{-n}$.

## Verification Note
- 7 unit tests in `tests/test_trackC.py` all pass.
- All experiments explicitly marked: **SIMULATED on classical hardware**.
- Honest negative result documented: classical simulated annealing (cut 33) far outperforms shallow QAOA ($p=1$ cut 17.28, $p=2$ cut 18.29) on NISQ-scale graph instances.

## Bugs Found & Logged (see FAILURE_LOG.md #9)
- Autograd ArrayBox conversion error in QNN loss function fixed.
