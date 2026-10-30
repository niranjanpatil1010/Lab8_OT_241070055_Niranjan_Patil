# Track D Prompt Log

## Original prompt (excerpt)
```
Track D — Submodular, parallel and distributed optimization:
greedy on a submodular objective (test diminishing returns + check approximation-guarantee assumptions);
load balancing, work stealing, data/model/pipeline parallelism with multiprocessing:
measure speedup, efficiency, overhead, interpret with Amdahl and Gustafson;
FedAvg vs centralized (vary local steps and rounds);
MGDA and GradNorm vs equal weighting on a two-task model.
```

## AI Response / Implementation Approach
- Implemented in `trackD/run.py` (numpy, multiprocessing, seeded):
  - D1: Unweighted/weighted coverage function, sampled diminishing returns check (500 trials, 0 violations), greedy vs exact brute-force (approx ratio = 1.0 >= 0.6321).
  - D2: Static LPT vs random assignment vs work-stealing simulation on 16 tasks / 4 machines; measured multiprocessing speedup across 1, 2, 4 cores on Windows; Amdahl/Gustafson analysis of process spawn overhead.
  - D3: FedAvg vs centralized SGD on linear regression across local steps {1, 5, 10} and communication rounds {5, 10, 20} with 4 clients.
  - D4: Shared representation two-task model (MSE regression + BCE classification) comparing equal weighting, analytical MGDA, and GradNorm over 10 seeds.

## Verification Note
- 12 unit tests in `tests/test_trackD.py` all pass.
- Diminishing returns confirmed with 0 violations over 500 random submodular tests.
- Centralized SGD vs FedAvg compared at strictly equal total gradient steps.
- MGDA analytical solution independently tested with orthogonal and collinear opposing gradients.

## Bugs Found & Logged (see FAILURE_LOG.md #7, #8)
- MGDA gradient sign flip caught by unit test.
- Windows multiprocessing pool lockup caught during execution and restructured.
