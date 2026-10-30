# Track I Prompt Log

## Original prompt (excerpt)
```
Track I — Quantum algorithms and QML advantage:
run every quantum algorithm in the course on a simulator:
Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor, quantum annealing;
write >=2 circuits by hand (not only library calls);
one run with a hardware noise model;
variational classifier / QNN / quantum kernel vs equivalent classical model on the same data
(accuracy, training time, parameters, circuit evaluations);
Grover vs linear search by query count AND time. Honest negative results are fine.
```

## AI Response / Implementation Approach
- Implemented in `trackI/run.py` using PennyLane simulator (`default.qubit` / `default.mixed`):
  - I1: Deutsch-Jozsa with hand-written circuit (constant vs balanced, 1 query vs 5 classical).
  - I2: Bernstein-Vazirani with hand-written circuit (recovered "1011" with probability 1.0 in 1 query).
  - I3: Grover's search with hand-written oracle and diffuser across $N \in [8, 16, 32, 64]$; honest comparison showing quadratic query speedup ($10.33\times$) but classical wall-clock CPU victory ($\mu\text{s}$ vs $\text{ms}$).
  - I4: Hand-written 3-qubit QFT circuit verified with zero entry error ($< 10^{-15}$) against analytic classical Discrete Fourier Transform matrix.
  - I5: Quantum Phase Estimation on $T$-gate recovering $\phi = 0.1250$ with probability 1.0000.
  - I6: Shor's algorithm order finding simulation factoring $N=15$ ($a=7, r=4 \implies 3 \times 5$).
  - I7: Simulated quantum annealing on 4-spin frustrated Ising glass recovering ground state ($E = -3.5$, 83.9% probability).
  - I8: Hardware noise model under Depolarizing noise degrading Bernstein-Vazirani fidelity from 100% to 39.9%.
  - I9: Quantum Kernel SVM vs Classical RBF SVM on 2D circles (780 circuit evaluations vs 0; classical RBF achieves 100% accuracy in 1.2 ms vs 80% in 1.47 s).

## Verification Note
- 9 unit tests in `tests/test_trackI.py` all pass.
- All algorithms marked: **SIMULATED on classical hardware**.
- Documented honest negative results: classical linear search is thousands of times faster in wall-clock time on classical CPUs than statevector simulation of Grover, and classical RBF SVM outperforms spatial quantum kernels without quadratic circuit evaluation overhead.

## Bugs Found & Logged (see FAILURE_LOG.md #10)
- QPE endianness bit-order reversal between controlled powers and QFT input.
