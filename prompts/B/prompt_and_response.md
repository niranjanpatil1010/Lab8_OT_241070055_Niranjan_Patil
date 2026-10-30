# Track B prompt log

## Original prompt (excerpt)
```
TO DO: Tracks B, C, D, F, G, H, I, J. Work ONE track at a time, in this order: B, D, C, I, F, H, J, G.
B: greedy, ILP, Branch-and-Bound, dynamic programming on selection/scheduling/allocation problems
(exact vs heuristic). Formulate Max-Cut (6-8 nodes) as QUBO/Ising (binary->spin conversion,
penalty terms for a constraint) and solve classically as the baseline for Track C.
```

## AI response / implementation approach
- Antigravity (Claude Sonnet 4.6 Thinking) generated all code in trackB/run.py
- Problems chosen: 0/1 Knapsack (allocation), Weighted Interval Scheduling (scheduling), Max-Cut on 7-node graph (combinatorial)
- QUBO/Ising: Max-Cut → minimise xᵀQx; binary→spin x=(1-s)/2; penalty λ(Σx-k)² for |S|=3

## Verification note
- 23 pytest tests all pass (python -m pytest tests/test_trackB.py -v)
- B&B, DP, ILP cross-verified on knapsack (all return 151)
- QUBO energy == Ising(h,J)+const: error < 1e-9
- Constrained QUBO brute-force returns |S|=3 as required
- Max-Cut brute-force (2^7=128) and QUBO brute-force agree on cut=33

## Bugs found and fixed (see FAILURE_LOG.md #5, #6)
1. Graph adjacency was all-zero due to bad symmetrization → fixed to hard-coded edge list
2. Test expected values for knapsack/scheduling were computed incorrectly by hand → fixed after pytest caught them

## Commit
[AI:antigravity] Track B: discrete opt (knapsack, scheduling, Max-Cut, QUBO/Ising)

## Tool used
Antigravity (Claude Sonnet 4.6 Thinking) — code generation, debugging, test writing, report writing
