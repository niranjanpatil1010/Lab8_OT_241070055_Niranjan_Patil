# Track H Prompt Log

## Original prompt (excerpt)
```
Track H — DBMS optimization:
SQLite or PostgreSQL, documented synthetic or TPC-H dataset (state size).
Formulate index (or view) selection as knapsack/ILP with a storage budget and
compare >=2 methods (greedy vs ILP/DP) against the default optimizer.
Compare EXPLAIN plans, latency, storage; query results must be identical.
```

## AI Response / Implementation Approach
- Implemented in `trackH/run.py` (SQLite 3, scipy.optimize.milp):
  - Synthetic relational dataset: 100,000 rows across `customers` (10k), `orders` (30k), and `lineitems` (60k). Database size: 3.48 MB.
  - Workload of 5 analytical and join queries.
  - Candidate pool of 8 secondary B-tree indexes.
  - Index selection formulated as a 0/1 Knapsack problem with a storage budget $B = 2224.8$ KB.
  - Solved with 3 methods:
    1. Default Optimizer (0 secondary indexes baseline).
    2. Greedy Benefit/Storage ratio heuristic.
    3. Dynamic Programming (Exact Knapsack).
    4. Integer Linear Programming (Exact ILP via `scipy.optimize.milp`).
  - Evaluated workload latency across 10 runs with mean ± std.
  - Extracted and compared `EXPLAIN QUERY PLAN` showing transition from table scans to index searches.
  - Strictly verified query result identity across all queries.

## Verification Note
- 5 unit tests in `tests/test_trackH.py` all pass.
- Verified exact bit-for-bit query result tuple identity before and after indexing.
- Verified plan improvements (e.g. elimination of temporary B-tree for GROUP BY in Q1, index nested-loop join in Q2/Q5).

## Bugs Found & Logged (see FAILURE_LOG.md #12)
- Missing `ORDER BY` caused index scan order to differ from sequential table scan order, caught by equality assertion and resolved.
